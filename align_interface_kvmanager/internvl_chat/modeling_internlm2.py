# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""InternLM models' APIs."""
import copy
from typing import Optional, Union, Tuple

import mindspore
import numpy as np
from mindnlp.transformers import PreTrainedModel
from mindnlp.transformers.modeling_outputs import BaseModelOutputWithPast
from mindspore import nn, ops, Tensor
import mindspore.common.dtype as mstype
from mindspore.common.initializer import initializer, Normal

from .configuration_internlm2 import InternLM2Config
from .internlm_layers import InternLM2DecoderLayer, LlamaRMSNorm, FreqsMgr, CausalMaskForInternLM2
from utils.kvcache_mgr import KVCachePreprocess
from utils.loss import CrossEntropyLoss


class InternLM2PreTrainedModel(PreTrainedModel):
    config_class = InternLM2Config
    base_model_prefix = 'model'
    _no_split_modules = ["InternLM2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Dense):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(initializer(Normal(self.config.initializer_range),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.has_bias:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, self.config.initializer_range, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.set_data(Tensor(weight, cell.weight.dtype))


class InternLM2Model(InternLM2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: InternLM2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.CellList([InternLM2DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.is_first_iteration = True

        head_dim = config.hidden_size //config.num_attention_heads
        self.freqs_mgr = FreqsMgr(
            head_dim=head_dim,
            max_position_embedding=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.casual_mask = CausalMaskForInternLM2(
            seq_length=config.seq_len,
            pad_token_id=config.pad_token_id,
            use_flash_attention=config.use_flash_attention,
        )
        self.kvcache_preprocess = KVCachePreprocess(
            max_seq_length=config.seq_len,
            use_kvcache_op=config.use_kvcache_op,
        )
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_valid_length = attention_mask.sum(axis=-1)  # (bs,)
        if not self.is_first_iteration:
            batch_valid_length -= 1

        # 1. retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            # ignore inputs_embeds if input_ids is not None
            bs, seq_len = input_ids.shape[:2]
            inputs_embeds = self.tok_embeddings(input_ids)
        elif inputs_embeds is not None:
            bs, seq_len = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        hidden_states = inputs_embeds

        # 2. rotary_emb
        if not use_cache:  # full input, usually for training
            freqs_cis = self.freqs_mgr(seq_len)  # cos and sin value with len of max_position_embedding
            attention_mask = self.casual_mask(attention_mask)  # (bs, seq_len) ->(bs, seq, seq), with casual and padding
            attention_mask = self.casual_mask.post_process(attention_mask, hidden_states.dtype) # to -inf-zero format
            kvcache_inputs = None
        else:
            if self.is_first_iteration:
                freqs_cis = self.freqs_mgr(seq_len)
                attention_mask = self.casual_mask(attention_mask)  # (bs, seq_len) ->(bs, seq, seq)
            else:
                freqs_cis = self.freqs_mgr.increment(batch_valid_length, bs) # get the value at index of valid_length
                attention_mask = self.casual_mask.increment(self.kvcache_preprocess.range, batch_valid_length)
            attention_mask = self.casual_mask.post_process(attention_mask, hidden_states.dtype)

            kvcache_inputs = self.kvcache_preprocess(bs, batch_valid_length)

        # 4. hidden_states
        for layer in self.layers:
            hidden_states = layer(hidden_states,
                                  attention_mask=attention_mask,
                                  freqs_cis=freqs_cis,
                                  kvcache_inputs=kvcache_inputs,
                                  use_cache=use_cache)

        # 5. final norm
        hidden_states = self.norm(hidden_states)

        return hidden_states


class InternLM2ForCausalLM(InternLM2PreTrainedModel):
    r"""
    Provide qwen training loss or logits through network.
        Args:
            config (QwenConfig): The config of Qwen model.

        Returns:
            Tensor, the loss or logits of the network.
    """
    _tied_weights_keys = ['output.weight']


    def __init__(self, config: InternLM2Config):
        super().__init__(config)

        self.model = InternLM2Model(config=config)
        self.output = nn.Dense(
            in_channels=config.hidden_size,
            out_channels=config.vocab_size,
            has_bias=False
        )
        self.loss = CrossEntropyLoss()

        self.pad_token_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.is_first_iteration = True

        self.post_init()

    def get_input_embeddings(self):
        return self.model.tok_embeddings

    def set_input_embeddings(self, value):
        self.model.tok_embeddings = value

    def get_output_embeddings(self):
        return self.output

    def set_output_embeddings(self, new_embeddings):
        self.output = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def prepare_inputs_for_generation(self, input_ids, inputs_embeds, attention_mask, **kwargs):
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if self.is_first_iteration and inputs_embeds is not None:
            input_ids = None
        else:
            inputs_embeds = None

        if not self.is_first_iteration and input_ids is not None:
            # keep only final ID
            bs, seq_len = input_ids.shape
            input_ids = input_ids[:, seq_len-1:]

        return {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "attention_mask": mindspore.Tensor(attention_mask, dtype=mindspore.int32),
            'use_cache': kwargs.get('use_cache'),
        }

    def construct(
        self,
        input_ids: Tensor = None,  # (bs, init_seq or 1)
        attention_mask: Optional[Tensor] = None,  # (bs, full_seq)
        position_ids: Optional[Tensor] = None,  # (bs, init_seq or 1)
        inputs_embeds: Optional[Tensor] = None,  # (bs, init_seq) or None
        labels: Optional[Tensor] = None, # for masked language model
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pre_gather = not use_cache or self.is_first_iteration
        if pre_gather: # gather the last token when there is full length output
            batch_valid_length = attention_mask.sum(axis=-1)  # (bs,)
            output = ops.gather(output, batch_valid_length - 1, axis=1, batch_dims=0)  # (bs, 1, vocab_size)
        logits = self.output(output)  # (bs, 1, vocab_size)

        if not self.training:
            logits = ops.cast(logits, mstype.float32)
            return logits

        # below for training
        # TODO revise inputmask and label
        # bs, seqlen = logits.shape[:2]
        # if labels is None:
        #     labels = ops.strided_slice(input_ids, (0, 1), (bs, seqlen), (1, 1))
        # else:
        #     # TODO add training code
        #     pass
        #
        # if logits.ndim > 2:
        #     logits = ops.reshape(logits, (-1, logits.shape[-1]))
        # logits = ops.cast(logits, mstype.float32)
        # labels = ops.reshape(labels, (-1,))
        # input_mask = ops.reshape(input_mask, (-1,))
        # loss = self.loss(logits, labels, input_mask)
        # return loss
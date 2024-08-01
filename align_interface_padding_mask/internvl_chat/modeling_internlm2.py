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
from typing import Optional, Union, Tuple, List

import mindspore
import numpy as np
from mindnlp.transformers import PreTrainedModel
from mindnlp.transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from mindnlp.transformers.modeling_outputs import BaseModelOutputWithPast
from mindspore import nn, ops, Tensor
import mindspore.common.dtype as mstype
from mindspore.common.initializer import initializer, Normal

from .configuration_internlm2 import InternLM2Config
from .internlm_layers import InternLM2DecoderLayer, LlamaRMSNorm, CausalMaskForInternLM2
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
        past_key_values: Optional[List[mindspore.Tensor]] = None,  # None or tuple[(bs, full_seq, hidden_size)]
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

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        # 1. retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            # ignore inputs_embeds if input_ids is not None
            batch_size, seq_length = input_ids.shape[:2]
            inputs_embeds = self.tok_embeddings(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        # 4. hidden_states
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(hidden_states,
                                  attention_mask=attention_mask,
                                  position_ids=position_ids,
                                  past_key_value=past_key_value,
                                  output_attentions=output_attentions,
                                  use_cache=use_cache)
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # 5. final norm
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        # currently only support return_dict=False
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        raise NotImplementedError


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

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]  # (bs, full_seq_len)

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids_cum = attention_mask.cumsum(-1) - 1
            position_ids = position_ids_cum.masked_fill(attention_mask == 0, 1)
            if past_key_values:
                # attention_mask is the same for each batch
                position_ids = position_ids_cum[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def construct(
        self,
        input_ids: Tensor = None,  # (bs, full_seq or 1)
        attention_mask: Optional[Tensor] = None,  # (bs, full_seq)
        position_ids: Optional[Tensor] = None,  # (bs, full_seq or 1)
        past_key_values: Optional[List[mindspore.Tensor]] = None,  # None or tuple[(bs, full_seq, hidden_size)]
        inputs_embeds: Optional[Tensor] = None,  # (bs, full_seq) or None
        labels: Optional[Tensor] = None, # for masked language model
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        # pick the last one
        pre_gather = use_cache and past_key_values is None
        if pre_gather:
            last_valid_pos = attention_mask.sum(-1) - 1
            hidden_states = ops.gather(hidden_states, last_valid_pos, 1)

        logits = self.output(hidden_states)  # (bs, seq_len/1, vocab_size)
        logits = logits.astype(mstype.float32)

        loss = None
        # TODO add loss code
        pass

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

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
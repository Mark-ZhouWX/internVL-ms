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

from mindspore import nn, ParallelMode, ops, Tensor
import mindspore.common.dtype as mstype


from .internlm_config import InternLM2Config
from .internlm_layers import InternLM2DecodeLayer, LlamaRMSNorm, FreqsMgr, CausalMaskForInternLM2, LlamaEmbedding
from .llm.base_llm_model import BaseLLMModel
from .utils.kvcache_mgr import KVCachePreprocess
from .utils.layers import Linear
from .utils.loss import CrossEntropyLoss


class InternLM2Model(BaseLLMModel):
    """transformer"""

    def __init__(self, config):
        # config = InternLM2Config(**config)
        super().__init__(config)
        self.dtype = config.compute_dtype
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_layers
        self.embed_dim = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads
        self.seq_length = config.seq_length
        self.pad_token_id = config.pad_token_id
        self.num_attention_heads = config.num_heads
        self.num_key_value_heads = config.n_kv_heads
        self.use_past = config.use_past
        self.is_dynamic = config.is_dynamic
        self.use_kvcache_op = config.use_kvcache_op
        self.is_flexible_shape = config.is_flexible_shape

        self.is_first_iteration = True
        self.use_flash_attention = config.use_flash_attention

        self.num_patches = 7 * 256
        self.image_start_token_pos = 26

        # 1. wte
        self.tok_embeddings = LlamaEmbedding(
            self.vocab_size, self.embed_dim, param_init_type=config.param_init_type, parallel_optimizer=True
        )

        # 2. drop
        self.drop = nn.Dropout(p=config.emb_dropout_prob)

        # 4. h hidden layers for transformer
        self.layers = nn.CellList()
        for layer_id in range(config.num_layers):
            layer = InternLM2DecodeLayer(
                config.batch_size,
                config.seq_length,
                layer_id,
                dim=config.hidden_size,
                n_heads=config.num_heads,
                n_kv_heads=config.n_kv_heads,
                intermediate_size=config.intermediate_size,
                norm_eps=config.rms_norm_eps,
                compute_dtype=config.compute_dtype,
                layernorm_compute_dtype=config.layernorm_compute_type,
                softmax_compute_dtype=config.softmax_compute_type,
                rotary_dtype=config.rotary_dtype,
                param_init_type=config.param_init_type,
                ln_param_init_type=config.ln_param_init_type,
                qkv_has_bias=config.qkv_has_bias,
                qkv_concat=config.qkv_concat,
                use_past=config.use_past,
                use_flash_attention=config.use_flash_attention,
            )

            self.layers.append(layer)

        self.freqs_mgr = FreqsMgr(
            head_dim=self.head_dim,
            seq_length=self.seq_length,
            max_position_embedding=config.max_position_embedding,
            rotary_dtype=config.rotary_dtype,
            theta=config.theta,
            scaling_factor=config.scaling_factor,
            extend_method=config.extend_method,
            is_dynamic=config.is_dynamic,
        )
        self.casual_mask = CausalMaskForInternLM2(
            seq_length=config.seq_length,
            compute_type=config.compute_dtype,
            is_dynamic=config.is_dynamic,
            pad_token_id=config.pad_token_id,
            use_flash_attention=config.use_flash_attention,
        )
        self.kvcache_preprocess = KVCachePreprocess(
            max_batch_size=config.batch_size,
            max_seq_length=config.seq_length,
            is_dynamic=config.is_dynamic,
            use_kvcache_op=config.use_kvcache_op,
            is_flexible_shape=config.is_flexible_shape,
        )
        # 5. final norm
        self.norm = LlamaRMSNorm(
            self.embed_dim,
            eps=config.rms_norm_eps,
            compute_type=config.layernorm_compute_type,
            param_init_type=config.ln_param_init_type,
        )

        self.shape = ops.Shape()

    def construct(
        self, input_ids: Tensor, init_reset=True, batch_valid_length=None, batch_index=None, zactivate_len=None,
            image_features=None,
    ):
        """construct"""
        if input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])

        # 1. wte
        inputs_embeds = self.tok_embeddings(input_ids)

        if self.is_first_iteration and image_features is not None:
            bs, seq_len = self.shape(input_ids)

            new_input_embeds = []
            for i in range(bs):
                cur_input_embeds = inputs_embeds[i]
                per_cur_image_features = image_features[i]
                assert per_cur_image_features.shape[0] == self.num_patches
                cur_input_embeds = ops.cat(
                    (
                        cur_input_embeds[: self.image_start_token_pos],
                        per_cur_image_features,
                        cur_input_embeds[self.image_start_token_pos + self.num_patches:],
                    ),
                    axis=0,
                )

                new_input_embeds.append(cur_input_embeds)

            hidden_states = ops.stack(new_input_embeds, axis=0)
        else:
            hidden_states = inputs_embeds

        # 2. drop
        hidden_states = self.drop(hidden_states)

        # 2. rotary_emb
        bs, seq_len = self.shape(input_ids)
        if not self.use_past:
            freqs_cis = self.freqs_mgr()
            mask = self.casual_mask(input_ids)  # mask: [bs, seq, seq]
            mask = self.casual_mask.post_process(mask)
            kvcache_inputs = None
        else:
            if self.is_first_iteration:
                freqs_cis = self.freqs_mgr(seq_len)
                mask = self.casual_mask(input_ids)  # mask: [bs, seq, seq]
            else:
                freqs_cis = self.freqs_mgr.increment(batch_valid_length, bs)
                if self.is_dynamic and self.is_flexible_shape and not self.use_kvcache_op:
                    mask = self.casual_mask.increment_slice(
                        self.kvcache_preprocess.range,
                        self.kvcache_preprocess.max_cache_length // bs,
                        batch_valid_length,
                        zactivate_len,
                    )
                else:
                    mask = self.casual_mask.increment(self.kvcache_preprocess.range, batch_valid_length, zactivate_len)
            mask = self.casual_mask.post_process(mask)

            kvcache_inputs = self.kvcache_preprocess(bs, batch_valid_length, batch_index, zactivate_len)

        # 4. hidden_states
        for i in range(self.num_hidden_layers):
            hidden_states = self.layers[i](hidden_states, freqs_cis, mask, kvcache_inputs=kvcache_inputs)

        # 5. final norm
        hidden_states = self.norm(hidden_states)

        return hidden_states


class InternLM2CausalLM(BaseLLMModel):
    r"""
    Provide qwen training loss or logits through network.
        Args:
            config (QwenConfig): The config of Qwen model.

        Returns:
            Tensor, the loss or logits of the network.
    """

    def __init__(self, config=None):
        super().__init__(config)

        self.model = InternLM2Model(config=config)
        self.output = Linear(
            in_channels=config.hidden_size,
            out_channels=config.vocab_size,
            has_bias=False,
            compute_dtype=config.compute_dtype,
            param_init_type=mstype.float16,
            weight_init="normal",
        )
        self.loss = CrossEntropyLoss()

        self.pad_token_id = config.pad_token_id
        self.use_past = config.use_past
        self.ignore_token_id = config.ignore_token_id
        self.seq_length = config.seq_length
        self.vocab_size = config.vocab_size
        self.is_first_iteration = True
        self.not_equal = ops.NotEqual()
        self.cast = ops.Cast()
        self.add = ops.Add()
        self.reshape = ops.Reshape()
        self.ones = ops.Ones()
        self.slice = ops.StridedSlice()
        self.mul = ops.Mul()
        self.sub_batch_valid_len = ops.Sub()
        self.gather = ops.Gather(1)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": Tensor(input_ids, mstype.int32),
            "image_features": kwargs.get("image_features")
        }

    def construct(
        self,
        input_ids,
        labels=None,
        input_position=None,
        position_ids=None,
        attention_mask=None,
        input_embeds=None,
        init_reset=True,
        batch_valid_length=None,
        batch_index=None,
        zactivate_len=None,
        image_features=None,
    ):
        bsz, seqlen = input_ids.shape
        if self.use_past:
            if not isinstance(batch_valid_length, Tensor):
                batch_valid_length = self.ones((bsz,), mstype.int32)
        if self.training:
            tokens = self.slice(input_ids, (0, 0), (bsz, seqlen - 1), (1, 1))
        else:
            tokens = input_ids

        if batch_valid_length is not None:
            batch_valid_length = self.reshape(batch_valid_length, (-1,))
        if not self.is_first_iteration:
            batch_valid_length = self.sub_batch_valid_len(batch_valid_length, 1)

        output = self.model(
            tokens,
            init_reset=init_reset,
            batch_valid_length=batch_valid_length,
            batch_index=batch_index,
            zactivate_len=zactivate_len,
            image_features=image_features,
        )
        pre_gather = (not self.use_past or self.is_first_iteration) and batch_valid_length is not None
        if pre_gather:
            output = self.gather(output, self.sub_batch_valid_len(batch_valid_length, 1), 1)
        logits = self.output(output)

        input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)
        if labels is None:
            labels = self.slice(input_ids, (0, 1), (bsz, seqlen), (1, 1))
        else:
            if labels.ndim > 1:
                if self.training:
                    labels = self.slice(labels, (0, 1), (bsz, seqlen), (1, 1))
                label_mask = self.cast(self.not_equal(labels, self.ignore_token_id), mstype.float32)
                input_mask = self.mul(input_mask, label_mask)

        if not self.training:
            if not pre_gather:
                logits = self.reshape(logits, (bsz, seqlen, -1))
            logits = self.cast(logits, mstype.float32)
            # makes cast effective to avoid allgather issue in Mindspore1.10
            input_mask = self.add(input_mask, 1)
            return logits, tokens, input_mask

        if logits.ndim > 2:
            logits = self.reshape(logits, (-1, logits.shape[-1]))
        logits = self.cast(logits, mstype.float32)
        labels = self.reshape(labels, (-1,))
        input_mask = self.reshape(input_mask, (-1,))
        loss = self.loss(logits, labels, input_mask)
        return loss
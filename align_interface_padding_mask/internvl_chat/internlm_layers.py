import math

from typing import Optional, Tuple

import mindspore
import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
from mindnlp.transformers.activations import ACT2FN
from mindnlp.transformers.models.llama.modeling_llama import LlamaMLP, LlamaDynamicNTKScalingRotaryEmbedding, repeat_kv
from mindspore import Parameter, Tensor, nn, ops
from mindspore.common.initializer import initializer

from .configuration_internlm2 import InternLM2Config


# Copied from transformers.model.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ops.cat((-x2, x1), axis=-1)


# Copied from transformers.model.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRMSNorm(nn.Cell):
    r"""
    A self-defined RMSNorm operation using reduce mean.

        Args:
            dim (int): The shape of the input tensor
            eps (float): The epsilon value of the denominator. Default 1e-5.
            compute_type: The compute type.
            param_init_type: The layer norm param init type.
        Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(batch, seq\_length, hidden\_size)`.

        Outputs:
            Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """

    def __init__(self, dim, eps=1e-6, compute_type=mstype.float32, param_init_type=mstype.float32):
        super(LlamaRMSNorm, self).__init__()
        self.eps = eps
        self.compute_type = compute_type
        self.weight = Parameter(initializer("ones", (dim,), dtype=param_init_type), parallel_optimizer=False)

        self.norm = ops.RmsNorm(eps)

        if ms.get_context("device_target") == "Ascend":
            self.rms_norm = self._rms_norm
        else:
            self.rms_norm = self._self_norm

    def _self_norm(self, x):
        original_type = x.dtype
        norm_factor = ops.square(ops.cast(x, self.compute_type))
        norm_factor = ops.mean(norm_factor, -1, keep_dims=True)
        norm_factor = ops.add(norm_factor, self.eps)
        norm_factor = ops.rsqrt(norm_factor)
        output = ops.mul(x, ops.cast(norm_factor, original_type))
        output = ops.mul(output, ops.cast(self.weight, original_type))
        return output

    def _rms_norm(self, x):
        original_type = x.dtype
        return self.norm(x, ops.cast(self.weight, original_type))[0]

    def construct(self, x):
        """Forward of RMSNorm."""
        return self.rms_norm(x)


class InternLM2Attention(nn.Cell):
    r"""
    This is an implementation of multi head attention in LLaMA.

    Args:
            - **batch_size** (int): The batch size of the input tensor when do incremental prediction. Should be a
                positive value.
                When do training or prediction, the argument will not work and the user can just pass None to the
                argument.
            - **src_seq_length** (int): The sequence length of the query vector.
            - **tgt_seq_length** (int): The sequence length of the key and value vector.
            - **dim** (int): The hidden size of the input.
            - **head_dim** (int): The dim of head.
            - **n_heads** (int): The number of the heads.
            - **compute_dtype** (dtype.Number): The computation type of dense. Default mstype.float16.
                Should be mstype.float32 or mstype.float16.
            - **softmax_compute_type** (dtype.Number): The type of softmax computation module. Default mstype.float32.
                Should be mstype.float32 or mstype.float16.
            - **param_init_type** (dtype.Number): The parameter initialization type of the module. Default mstype.
                float32. Should be mstype.float32 or mstype.float16.
            - **qkv_has_bias** (bool): Whether Q/K/V in attention has bias or not.
            - **use_past** (bool): Use the past state to compute, used for incremental prediction.
                For example, if we have two words and want to generate the ten more words.
                We just need to compute the two words" state only once, and generate the next word one by one.
                When use_past is True, there are two steps to run the prediction.
                In the first step, set the is_first_iteration to be True by
                `model.add_flags_recursive(is_first_iteration=True)`, and pass the full inputs. Then, set the
                is_first_iteration to be False by `model.add_flags_recursive(is_first_iteration=False)`. At this moment,
                pass the single step"s input tensor, and loop it. Default False.

    Inputs:
            - **x** (Tensor) - The input tokens with shape (batch_size, src_seq_length, hidden_size) or
                (batch_size * src_seq_length, hidden_size), if the use_past is False or is_first_iteration=True.
                Otherwise, must be (batch_size, 1, hidden_size)
            - **freqs_cis** (Tuple) - The precompute freqs and mask for rotary position embedding used in attention.
            - **attention_mask** (Tensor) - If the use_past is False or is_first_iteration=True, the attention mask
                matrix should ba (batch_size, src_seq_length, tgt_seq_length), or None. None means there will be no mask
                in softmax computation. Otherwise, the mask must be (batch_size, 1, tgt_seq_length)
            - **key_past** (Tensor) - Float16 tensor with shape (batch_size, num_heads, head_dim, tgt_seq_length).
                The past calculated key vector. Used for incremental prediction when the use_past is True.
                Default None.
            - **value_past** (Tensor) - Float16 tensor with shape (batch_size, num_heads, tgt_seq_length,
                head_dim).
                The past calculated value vector. Used for incremental prediction when the use_past is True.
                Default None.
            - **batch_valid_length** (Tensor) - Int32 tensor with shape (batch_size,) the past calculated the index.
                Used for incremental prediction when the use_past is True. Default None.

    Outputs:
            Tuple, a tuple contains(`output`, `layer_present`)

            - **output** (Tensor) - Tensor, the float tensor of the output of the layer with
                shape (batch_size, src_seq_length, hidden_size) or (batch_size * src_seq_length, hidden_size),
                if the use_past is False or is_first_iteration=True. Otherwise, it will be (batch_size, 1, hidden_size).

            - **layer_present** (Tuple) - A tuple of the Tensor of the projected key and value vector with
                ((batch_size, num_heads, head_dim, tgt_seq_length),
                (batch_size, num_heads, tgt_seq_length, head_dim)).
    """

    def __init__(self, config: InternLM2Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads # num_query_head
        self.head_dim = config.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # num_query_head per key_value
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.is_first_iteration = True

        self.wqkv = nn.Dense(self.hidden_size,
                             (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
                             has_bias=config.bias)
        self.wo = nn.Dense(self.num_heads * self.head_dim, self.hidden_size, has_bias=config.bias)

        if False and self.use_flash_attention:
            self.flash_attention = FlashAttention(self.head_dim, n_heads, next_block_num=0, high_precision=True)

        self.inv_norm_factor = Tensor(1.0 / self.head_dim**0.5)

        self.batch_matmul_q_k = ops.BatchMatMul(transpose_b=True)
        self.batch_matmul = ops.BatchMatMul()
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    def construct(
            self,
            hidden_states: mindspore.Tensor,
            attention_mask: Optional[mindspore.Tensor] = None,  # (bs, 1, seq_len, seq_len)
            position_ids: Optional[mindspore.Tensor] = None,  # (bs, seq_len/1)
            past_key_value: Optional[Tuple[ms.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor], Optional[Tuple[ms.Tensor]]]:
        """Forward process of the MultiHeadAttention"""
        # [bs, seq/1, hidden_dim]
        bsz, q_len, _ = ops.shape(hidden_states)
        # [bs * seq/1, hidden_dim]
        hidden_states = ops.reshape(hidden_states, (-1, hidden_states.shape[-1]))
        bs_seq = hidden_states.shape[0]
        qkv = self.wqkv(hidden_states)

        # split qkv
        qkv = ops.reshape(qkv, (bsz, q_len, -1, self.num_key_value_groups + 2, self.head_dim))
        query_states = qkv[:, :, :, :self.num_key_value_groups, :].reshape(bsz, q_len, self.num_heads, self.head_dim)
        key_states = qkv[:, :, :, self.num_key_value_groups, :]  # (bs, seq, num_key_head, head_dim)
        value = qkv[:, :, :, self.num_key_value_groups + 1, :]  # (bs, seq, num_key_head, head_dim)

        # [bs, n_query_head/n_key_head, seq/1, head_dim]
        query_states = ops.transpose(query_states, (0, 2, 1, 3))
        key_states = ops.transpose(key_states, (0, 2, 1, 3))
        value_states = ops.transpose(value, (0, 2, 1, 3))

        kv_seq_len = key_states.shape[-2]  # seq/1
        if past_key_value is not None:
            kv_seq_len = past_key_value[0].shape[-2]  # seq
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)  # seq 1024

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # kv cache: [bs, n_kv_head, 1, head_dim] -> [bs, n_kv_head, seq, head_dim]
        if past_key_value is not None:
            # current_valid_pos: one hot vector at position_id
            seq_range = ops.arange(0, kv_seq_len, dtype=ms.int32)
            full_shape = past_key_value[0].shape
            current_valid_pos = ops.equal(seq_range.reshape(1, 1, -1, 1), position_ids)
            key_states = ops.where(current_valid_pos.broadcast_to(full_shape), key_states.broadcast_to(full_shape), past_key_value[0])  # (bs, n_kv_head, seq, head_dim)
            value_states = ops.where(current_valid_pos.broadcast_to(full_shape), value_states.broadcast_to(full_shape), past_key_value[1])

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = ops.matmul(query_states, key_states.swapaxes(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.shape != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.shape}"
            )

        if attention_mask is not None:
            if attention_mask.shape != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = ops.softmax(attn_weights, axis=-1, dtype=mindspore.float32).to(query_states.dtype)
        # attn_weights = ops.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = ops.matmul(attn_weights, value_states)

        if attn_output.shape != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.swapaxes(1, 2)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        attn_output = self.wo(attn_output)  # dp, mp -> dp, 1 / dp * mp, 1

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value



class InternLM2MLP(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.w1 = nn.Dense(self.hidden_size, self.intermediate_size, has_bias=False)
        self.w3 = nn.Dense(self.hidden_size, self.intermediate_size, has_bias=False)
        self.w2 = nn.Dense(self.intermediate_size, self.hidden_size, has_bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def construct(self, x):
        down_proj = self.w2(self.act_fn(self.w1(x)) * self.w3(x))
        return down_proj


# actually Qwen Decoder layer
class InternLM2DecoderLayer(nn.Cell):
    def __init__(self, config: InternLM2Config):
        super().__init__()
        self.attention_norm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = InternLM2Attention(config)
        self.feed_forward = InternLM2MLP(config)

    def construct(self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_value: Optional[Tuple[ms.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs):
        """Forward of transformer block."""
        # [bs, seq/1, hidden_dim]
        residual = hidden_states

        hidden_states = self.attention_norm(hidden_states)
        # [bs, seq/1, hidden_dim]
        hidden_states, self_attn_weights, present_key_value = self.attention(
                           hidden_states=hidden_states,
                           attention_mask=attention_mask,
                           position_ids=position_ids,
                           past_key_value=past_key_value,
                           output_attentions=output_attentions,
                           use_cache=use_cache,
                           **kwargs,)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

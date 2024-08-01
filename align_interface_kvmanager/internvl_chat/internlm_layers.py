from enum import Enum
from typing import Optional, Tuple

import mindspore
import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
from mindnlp.transformers.activations import ACT2FN
from mindnlp.transformers.models.llama.modeling_llama import LlamaMLP
from mindspore import Parameter, Tensor, nn, ops
from mindspore._c_expression import MSContext
from mindspore.common.initializer import initializer

from .configuration_internlm2 import InternLM2Config
from utils.kvcache_mgr import KVCacheMgr
from utils.layers import Linear


def is_910a():
    device = MSContext.get_instance().get_ascend_soc_version()
    return device in ["910a", "ascend910"]


class SeqExtendMethod(Enum):
    """Stores the acceptable string identifiers for seq length extend method"""

    PI = "PI"
    NTK = "NTK"
    NONE = "None"



class FreqsMgr(nn.Cell):
    r"""freqs_cis manager."""

    def __init__(
        self,
        head_dim,
        max_position_embedding=4096,
        rotary_dtype=mstype.float16,
        rope_theta=10000.0,
        rope_scaling=1.0,
        extend_method=SeqExtendMethod.NONE.value,
    ):
        super().__init__()
        if extend_method == SeqExtendMethod.NTK.value:
            rope_theta *= rope_scaling
        freqs_base = np.arange(0, head_dim, 2)[: (head_dim // 2)].astype(np.float32)  # (head_dim // 2, )
        freqs = 1.0 / (rope_theta ** (freqs_base / head_dim))  # (head_dim // 2, )
        if extend_method == SeqExtendMethod.PI.value:
            t = np.arange(0, max_position_embedding / rope_scaling, 1 / rope_scaling).astype(np.float32)
        else:
            t = np.arange(0, max_position_embedding, 1).astype(np.float32)
        freqs = np.outer(t, freqs)  # (max_position_embedding, head_dim // 2)
        emb = np.concatenate((freqs, freqs), axis=-1)
        freqs_cos = np.cos(emb)  # (seq_len, head_dim)
        freqs_sin = np.sin(emb)  # (seq_len, head_dim)
        swap_mask = FreqsMgr.get_swap_mask(head_dim)

        self.head_dim = head_dim
        self.freqs_cos = Tensor(freqs_cos, dtype=rotary_dtype)
        self.freqs_sin = Tensor(freqs_sin, dtype=rotary_dtype)
        self.swap_mask = Tensor(swap_mask, dtype=rotary_dtype)

    def construct(self, seq_len):
        freqs_cos, freqs_sin = self.freqs_cos, self.freqs_sin
        freqs_cos = ops.strided_slice(freqs_cos, (0, 0), (seq_len, self.head_dim), (1, 1))
        freqs_sin = ops.strided_slice(freqs_sin, (0, 0), (seq_len, self.head_dim), (1, 1))
        return freqs_cos, freqs_sin, self.swap_mask

    def increment(self, batch_valid_length, batch_size):
        freqs_cos = ops.reshape(ops.gather(self.freqs_cos, batch_valid_length, 0), (batch_size, 1, 1, self.head_dim))
        freqs_sin = ops.reshape(ops.gather(self.freqs_sin, batch_valid_length, 0), (batch_size, 1, 1, self.head_dim))
        return freqs_cos, freqs_sin, self.swap_mask

    @staticmethod
    def get_swap_mask(head_dim):
        """Swap matrix that swap two halves of the query """
        zero_block = np.zeros((head_dim // 2, head_dim // 2), dtype=np.float32)
        id_block = np.identity(head_dim // 2, dtype=np.float32)
        return np.block([[zero_block, id_block], [-id_block, zero_block]])


class LlamaRotaryEmbedding(nn.Cell):
    r"""
    Rotary Position Embedding.

    Args:
            - **head_dim** (int): The dim of multi head attention.
            - **compute_dtype** (mstype): The compute type, default mstype.float16.
    Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(batch, seq\_length, hidden\_size)`.

    Outputs:
            Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """

    def __init__(self, head_dim=128, use_rope_slice=False):
        super().__init__(auto_prefix=False)
        self.half_head_dim = head_dim // 2
        self.head_dim = head_dim
        self.use_rope_slice = use_rope_slice
        self.is_first_iteration = True
        self.is_ascend = ms.get_context("device_target") == "Ascend"

        self.bmm = ops.BatchMatMul()

    def rotate_half(self, x, swap_mask):
        # [bs, n_head/n_kv_head, seq/1, head_dim], [head_dim, head_dim]
        if self.is_ascend:
            x = self.bmm(x, swap_mask)
        else:
            x = ops.matmul(x, swap_mask)
        return x

    def slice_half(self, x):
        bs, n_head, seq, _ = ops.shape(x)
        x1 = ops.strided_slice(x, (0, 0, 0, 0), (bs, n_head, seq, self.half_head_dim), (1, 1, 1, 1))
        x2 = ops.strided_slice(x, (0, 0, 0, self.half_head_dim), (bs, n_head, seq, self.head_dim), (1, 1, 1, 1))
        x = ops.concat((ops.neg(x2), x1), -1)
        return x

    def construct(self, xq: Tensor, xk: Tensor, freqs_cis):
        """Forward of rotary position embedding."""
        # xq, xk: [bs, n_head/n_kv_head, seq/1, head_dim]
        freqs_cos, freqs_sin, swap_mask = freqs_cis
        if self.use_rope_slice:
            xq_out = ops.add(ops.mul(xq, freqs_cos), ops.mul(self.slice_half(xq), freqs_sin))
            xk_out = ops.add(ops.mul(xk, freqs_cos), ops.mul(self.slice_half(xk), freqs_sin))
        else:
            xq_out = ops.add(ops.mul(xq, freqs_cos), ops.mul(self.rotate_half(xq, swap_mask), freqs_sin))
            xk_out = ops.add(ops.mul(xk, freqs_cos), ops.mul(self.rotate_half(xk, swap_mask), freqs_sin))

        return xq_out, xk_out


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

        if ms.get_context("device_target") == "Ascend" and not is_910a():
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


class CausalMaskForInternLM2(nn.Cell):
    r"""Get the Lower triangular matrix from the input_ids.
    [[[1. 0. 0. 0. 0]
      [1. 1. 0. 0. 0]
      [1. 1. 1. 0. 0]
      [1. 1. 1. 1. 0]
      [1. 1. 1. 1. 0]]]"""

    def __init__(
        self, seq_length, pad_token_id=0, use_flash_attention=False
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.use_flash_attention = use_flash_attention
        self.multiply_data = Tensor([-10000.0])
        self.one = Tensor([1.0])
        self.lower_triangle_mask = Tensor(np.tril(np.ones(shape=(seq_length, seq_length))), mstype.float32)

    def construct(self, attention_mask):
        """Forward process of the CausalMask"""
        bs, seq_len = attention_mask.shape
        shape_right = (bs, 1, seq_len)
        # Mask the padded inputs
        mask_right = ops.reshape(attention_mask, shape_right)
        lower_triangle = ops.expand_dims(self.lower_triangle_mask, 0)
        # the returned shape is [bs, seq_length, seq_length]
        attention_mask = ops.mul(mask_right, lower_triangle.astype(attention_mask.dtype))
        return attention_mask

    def increment(self, seq_range, batch_valid_length, zactivate_len=None):
        if zactivate_len is not None:
            seq_range = ops.strided_slice(seq_range, (0, 0, 0), (1, 1, ops.shape(zactivate_len)[0]), (1, 1, 1))
        mask = ops.less_equal(ops.reshape(seq_range, (1, 1, -1)), ops.reshape(batch_valid_length, (-1, 1, 1)))
        return mask

    def post_process(self, mask, compute_dtype=ms.float32):
        mask = ops.sub(1.0, ops.cast(mask, compute_dtype))
        if not self.use_flash_attention:
            mask = ops.expand_dims(mask, 1)
            mask = ops.mul(mask, self.multiply_data.astype(compute_dtype))
        return mask


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
        self.use_flash_attention = config.use_flash_attention
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.is_first_iteration = True

        self.apply_rotary_emb = LlamaRotaryEmbedding(self.head_dim)
        self.wqkv = nn.Dense(self.hidden_size,
                             (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
                             has_bias=config.bias)
        self.wo = nn.Dense(self.num_heads * self.head_dim, self.hidden_size, has_bias=config.bias)

        if False and self.use_flash_attention:
            self.flash_attention = FlashAttention(self.head_dim, n_heads, next_block_num=0, high_precision=True)

        if config.use_cache:
            self.kvcache_mgr = KVCacheMgr(
                self.num_key_value_heads,
                self.head_dim,
                max_batch_size=config.batch_size,
                max_seq_length=config.seq_len,
                use_kvcache_op=config.use_kvcache_op,
            )

        self.inv_norm_factor = Tensor(1.0 / self.head_dim**0.5)

        self.batch_matmul_q_k = ops.BatchMatMul(transpose_b=True)
        self.batch_matmul = ops.BatchMatMul()

    def construct(
            self,
            hidden_states: mindspore.Tensor,
            attention_mask: Optional[mindspore.Tensor] = None,  # (bs, 1, seq_len, seq_len)
            position_ids: Optional[mindspore.Tensor] = None,
            freqs_cis: Tuple[Tensor, Tensor] = None, # precomputed rope cos and sin value
            kvcache_inputs: Optional[Tuple]=None,  # kv cache preprocess data
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
    ) -> mindspore.Tensor:
        """Forward process of the MultiHeadAttention"""
        # [bs, seq/1, hidden_dim]
        bs, seq_len, _ = ops.shape(hidden_states)
        # [bs * seq/1, hidden_dim]
        hidden_states = ops.reshape(hidden_states, (-1, hidden_states.shape[-1]))
        bs_seq = hidden_states.shape[0]
        qkv = self.wqkv(hidden_states)

        # split qkv
        qkv = ops.reshape(qkv, (bs, seq_len, -1, self.num_key_value_groups + 2, self.head_dim))
        query = qkv[:, :, :, :self.num_key_value_groups, :].reshape(bs, seq_len, self.num_heads, self.head_dim)
        key = qkv[:, :, :, self.num_key_value_groups, :]  # (bs, seq, num_key_head, head_dim)
        value = qkv[:, :, :, self.num_key_value_groups + 1, :]  # (bs, seq, num_key_head, head_dim)

        # [bs, n_query_head/n_key_head, seq/1, head_dim]
        query = ops.transpose(query, (0, 2, 1, 3))
        key = ops.transpose(key, (0, 2, 1, 3))
        value = ops.transpose(value, (0, 2, 1, 3))

        query, key = self.apply_rotary_emb(query, key, freqs_cis)
        # kv cache: [bs, n_kv_head, 1, head_dim] -> [bs, n_kv_head, seq, head_dim]
        if use_cache:
            key, value = self.kvcache_mgr(key, value, kvcache_inputs)
        # kv share: [bs, n_kv_head, seq, head_dim] -> [bs, n_head, seq, head_dim]
        key = self._repeat_kv(key, self.num_key_value_groups)
        value = self._repeat_kv(value, self.num_key_value_groups)
        # q, k, v: [bs, n_head, seq/1, head_dim], [bs, n_head, seq, head_dim], [bs, n_head, seq, head_dim]
        if self.use_flash_attention:
            attention = self.flash_attention(query, key, value, attention_mask)
            attention = self._merge_heads(attention)
        else:
            attention = self._attn(query, key, value, attention_mask)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        output = self.wo(attention)  # dp, mp -> dp, 1 / dp * mp, 1

        return output

    def _repeat_kv(self, x, rep):
        if rep == 1:
            return x
        bs, n_kv_head, seqlen, head_dim = ops.shape(x)
        x = ops.reshape(x, (bs, n_kv_head, 1, seqlen * head_dim))
        x = ops.tile(x, (1, 1, rep, 1))
        x = ops.reshape(x, (bs, n_kv_head * rep, seqlen, head_dim))
        return x

    def _merge_heads(self, x):
        """
        convert a 4d input to a 2d or 3d output

        Inputs:
            x: input tensor

        Output:
            x_merge: the 2d output
        """
        # [bs, n_head, seq/1, head_dim]
        x = ops.transpose(x, (0, 2, 1, 3))  # dp,mp,1,1 -> dp,1,mp,1
        # [bs, seq/1, n_head, head_dim]
        bs, seq_len, n_head, head_dim = ops.shape(x)
        # [bs, seq/1, hidden_dim]
        new_shape = (bs, seq_len, n_head * head_dim)
        x_merge = ops.reshape(x, new_shape)
        return x_merge

    def _attn(self, query, key, value, mask):
        """
        Get the weighted score along the seq_length

        Inputs:
            query: the query matrix
            key: the key matrix
            value: the value matrix
            mask: the attention mask adder matrix with shape (batch_size,
            1, seq_length, seq_length)
        Outputs:
            weighted_values: Tensor, the weighted sum scores
        """
        # q, k: [bs, n_head, seq/1, head_dim], [bs, n_head, seq, head_dim]
        score = self.batch_matmul_q_k(query, key)
        # score: [bs, n_head, seq/1, seq]
        score = ops.mul(score, self.inv_norm_factor.astype(query.dtype))
        score = ops.add(mask, score)

        attention_probs = ops.softmax(score)
        # score, v: [bs, n_head, seq/1, seq], [bs, n_head, seq, head_dim]
        weighted_values = self.batch_matmul(attention_probs, value)
        # [bs, n_head, seq/1, head_dim]
        attention_merge = self._merge_heads(weighted_values)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        return attention_merge



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

    def construct(self, hidden_states, attention_mask, freqs_cis, kvcache_inputs, use_cache):
        """Forward of transformer block."""
        # [bs, seq/1, hidden_dim]
        input_x = self.attention_norm(hidden_states)
        # [bs, seq/1, hidden_dim]
        h = self.attention(input_x,
                           attention_mask=attention_mask,
                           freqs_cis=freqs_cis,
                           kvcache_inputs=kvcache_inputs,
                           use_cache=use_cache)
        h = ops.add(hidden_states, h)
        ffn_norm = self.ffn_norm(h)
        # [bs, seq/1, hidden_dim]
        ffn_out = self.feed_forward(ffn_norm)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        out = ops.add(h, ffn_out)
        return out

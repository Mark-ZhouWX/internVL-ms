from typing import Optional

import mindspore
import numpy as np
from mindnlp.transformers.modeling_attn_mask_utils import AttentionMaskConverter
from mindspore import ops


def _expand_mask(mask: mindspore.Tensor, dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].broadcast_to((bsz, 1, tgt_len, src_len)).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(mindspore.bool_),
                                     mindspore.tensor(np.finfo(mindspore.dtype_to_nptype(dtype)).min, dtype=dtype))


def _make_causal_mask(
        input_ids_shape,
        dtype,
        past_key_values_length: int = 0,
        sliding_window: Optional[int] = None,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = ops.full((tgt_len, tgt_len), mindspore.tensor(np.finfo(mindspore.dtype_to_nptype(dtype)).min))
    mask_cond = ops.arange(mask.shape[-1])
    mask = mask.masked_fill(mask_cond < (mask_cond + 1).view(mask.shape[-1], 1), mindspore.Tensor(0, dtype=mask.dtype))

    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = ops.cat([ops.zeros((tgt_len, past_key_values_length), dtype=dtype), mask], axis=-1)

    # add lower triangular sliding window mask if necessary
    if sliding_window is not None:
        diagonal = past_key_values_length - sliding_window + 1

        context_mask = 1 - ops.triu(ops.ones_like(mask, dtype=mindspore.int32), diagonal=diagonal)
        mask = mask.masked_fill(context_mask.bool(), mindspore.tensor(np.finfo(mindspore.dtype_to_nptype(dtype)).min))

    return mask[None, None, :, :].broadcast_to((bsz, 1, tgt_len, tgt_len + past_key_values_length))


def to_4d(
    self,
    attention_mask_2d: mindspore.Tensor,
    query_length: int,
    key_value_length: Optional[int] = None,
    dtype = mindspore.float32,
) -> mindspore.Tensor:
    """
    Converts 2D attention mask to 4D attention mask by expanding mask to (bsz, head_dim=1, query_length,
    key_value_length) shape and by adding a large negative bias to not-attended positions. If attention_mask is
    causal, a causal mask will be added.
    """
    input_shape = (attention_mask_2d.shape[0], query_length)

    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    causal_4d_mask = None
    if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:
        if key_value_length is None:
            raise ValueError(
                "This attention mask converter is causal. Make sure to pass `key_value_length` to correctly create a causal mask."
            )

        past_key_values_length = key_value_length - query_length
        causal_4d_mask = self._make_causal_mask(
            input_shape,
            dtype,
            past_key_values_length=past_key_values_length,
            sliding_window=self.sliding_window,
        )
    elif self.sliding_window is not None:
        raise NotImplementedError("Sliding window is currently only implemented for causal masking")

    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    expanded_attn_mask = self._expand_mask(attention_mask_2d, dtype, tgt_len=input_shape[-1])

    if causal_4d_mask is not None:
        expanded_attn_mask = causal_4d_mask.masked_fill(expanded_attn_mask.bool(), mindspore.Tensor(np.finfo(mindspore.dtype_to_nptype(dtype)).min, dtype=dtype))

    # expanded_attn_mask + causal_4d_mask can cause some overflow
    expanded_4d_mask = expanded_attn_mask

    return expanded_4d_mask
def patch_attn_mask():
    if mindspore.get_context('mode') == 0:
        AttentionMaskConverter._make_causal_mask = _make_causal_mask
        AttentionMaskConverter._expand_mask = _expand_mask
        AttentionMaskConverter.to_4d = to_4d
        print('Monkey patch AttentionMaskConverter._make_causal_mask')
        print('Monkey patch AttentionMaskConverter.to_4d')
        print('Monkey patch AttentionMaskConverter._expand_mask')

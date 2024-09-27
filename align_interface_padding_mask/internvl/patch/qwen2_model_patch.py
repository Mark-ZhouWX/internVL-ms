import math
from typing import Optional, List, Tuple, Union

import mindspore
from mindnlp.transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Config
from mindnlp.transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from mindnlp.transformers.modeling_outputs import BaseModelOutputWithPast
from mindnlp.transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, Qwen2Attention, Qwen2Model
from mindnlp.utils import logging
from mindspore import ops
# expand method in modeling_qwen2.repeat_kv only works for pynative mode, use the one from llama instead
# from mindnlp.transformers.models.qwen2.modeling_qwen2 import repeat_kv
from mindnlp.transformers.models.llama.modeling_llama import repeat_kv

logger = logging.get_logger(__name__)


def Qwen2Config__init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        **kwargs,
):
    self.vocab_size = vocab_size
    self.max_position_embeddings = max_position_embeddings
    self.hidden_size = hidden_size
    self.intermediate_size = intermediate_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.use_sliding_window = use_sliding_window
    self.sliding_window = sliding_window if use_sliding_window else None
    self.max_window_layers = max_window_layers

    # for backward compatibility
    if num_key_value_heads is None:
        num_key_value_heads = num_attention_heads

    self.num_key_value_heads = num_key_value_heads
    self.hidden_act = hidden_act
    self.initializer_range = initializer_range
    self.rms_norm_eps = rms_norm_eps
    self.use_cache = use_cache
    self.rope_theta = rope_theta
    self.attention_dropout = attention_dropout

    super(Qwen2Config, self).__init__(
        tie_word_embeddings=tie_word_embeddings,
        **kwargs,
    )


def Qwen2Model_construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
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

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    if input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    past_key_values_length = 0
    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]

    if position_ids is None:
        position_ids = ops.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=mindspore.int64
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # 4d mask is passed through the layers
    attention_mask = _prepare_4d_causal_attention_mask(
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
        sliding_window=self.config.sliding_window,
    )

    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def Qwen2Attention_construct(
    self,
    hidden_states: mindspore.Tensor,
    attention_mask: Optional[mindspore.Tensor] = None,
    position_ids: Optional[mindspore.Tensor] = None,
    past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = True,
    **kwargs,
) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
    bsz, q_len, _ = hidden_states.shape

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapaxes(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapaxes(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len = past_key_value[0].shape[-2]  # seq

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # kv cache: [bs, n_kv_head, 1, head_dim] -> [bs, n_kv_head, seq, head_dim]
    if past_key_value is not None:
        # current_valid_pos: one hot vector at position_id
        seq_range = ops.arange(0, kv_seq_len, dtype=mindspore.int32)
        full_shape = past_key_value[0].shape
        current_valid_pos = ops.equal(seq_range.reshape(1, 1, -1, 1), position_ids)
        key_states = ops.where(current_valid_pos.broadcast_to(full_shape), key_states.broadcast_to(full_shape),
                               past_key_value[0])  # (bs, n_kv_head, seq, head_dim)
        value_states = ops.where(current_valid_pos.broadcast_to(full_shape), value_states.broadcast_to(full_shape),
                                 past_key_value[1])

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
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
    # use nn.Dropout instead ops.dropout in pynative mode due to speed
    # attn_weights = ops.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_weights = self.attention_dropout(attn_weights)
    attn_output = ops.matmul(attn_weights, value_states)

    if attn_output.shape != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.shape}"
        )

    attn_output = attn_output.swapaxes(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def Qwen2ForCausalLM_prepare_inputs_for_generation(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    # Omit tokens covered by past_key_values
    if past_key_values is not None:
        cache_length = past_length = past_key_values[0][0].shape[2]
        max_cache_length = None

        # Some generation methods already pass only the last input ID
        if input_ids.shape[1] > past_length:
            remove_prefix_length = past_length
        else:
            # Default to old behavior: keep only final ID
            remove_prefix_length = input_ids.shape[1] - 1

        input_ids = input_ids[:, remove_prefix_length:]

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids_cum = attention_mask.cumsum(-1) - 1
        position_ids = position_ids_cum.masked_fill(attention_mask == 0, 1)
        if past_key_values:
            # attention_mask is the same for each batch
            position_ids = position_ids_cum[:, -input_ids.shape[1]:]

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


def _set_cos_sin_cache(self, seq_len, dtype):
    print(f'seq_len {seq_len}, dtype {dtype}, type: {type(seq_len)}, {type(dtype)}')
    self.max_seq_len_cached = seq_len
    t = ops.arange(self.max_seq_len_cached, dtype=mindspore.int64).type_as(self.inv_freq)

    freqs = ops.outer(t, self.inv_freq)
    # Different from paper, but it uses a different permutation in order to obtain the same calculation
    emb = ops.cat((freqs, freqs), axis=-1)
    self.cos_cached = emb.cos().to(dtype)
    self.sin_cached = emb.sin().to(dtype)


def patch_qwen2_model():
    Qwen2ForCausalLM.prepare_inputs_for_generation = Qwen2ForCausalLM_prepare_inputs_for_generation
    Qwen2Attention.construct = Qwen2Attention_construct
    Qwen2Model.construct = Qwen2Model_construct
    Qwen2Config.__init__ = Qwen2Config__init__
    # Qwen2RotaryEmbedding._set_cos_sin_cache = _set_cos_sin_cache
    print(f'Monkey patch for Qwen2ForCausalLM.prepare_inputs_for_generation')
    print(f'Monkey patch for Qwen2Attention.construct')
    print(f'Monkey patch for Qwen2Model.construct')
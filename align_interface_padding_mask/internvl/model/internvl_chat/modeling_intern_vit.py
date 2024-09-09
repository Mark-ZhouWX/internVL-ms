import numpy as np
from typing import Optional, Tuple

import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore.numpy import ones

from mindnlp.transformers.modeling_utils import PreTrainedModel

from .configuration_intern_vit import InternVisionConfig


has_flash_attn = False

NORM2FN = {
    'layer_norm': nn.LayerNorm,
}


class DropPath(nn.Cell):
    """DropPath (Stochastic Depth) regularization layers"""

    def __init__(
        self,
        drop_prob: float = 0.0,
        scale_by_keep: bool = True,
    ) -> None:
        super().__init__()
        self.keep_prob = 1.0 - drop_prob
        self.scale_by_keep = scale_by_keep
        self.dropout = nn.Dropout(p=drop_prob)

    def construct(self, x: Tensor) -> Tensor:
        if self.keep_prob == 1.0 or not self.training:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = self.dropout(ops.ones(shape, dtype=x.dtype))
        if not self.scale_by_keep:
            random_tensor = ops.mul(random_tensor, self.keep_prob)
        return x * random_tensor


class InternVisionEmbeddings(nn.Cell):
    def __init__(self, config: InternVisionConfig):
        super().__init__(config)
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = ms.Parameter(
            ops.randn((1, 1, self.embed_dim)),
        )

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size,
            has_bias=True
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = ms.Parameter(ops.randn((1, self.num_positions, self.embed_dim)))

    def _get_pos_embed(self, pos_embed, H, W):
        target_dtype = pos_embed.dtype
        pos_embed = pos_embed.float().reshape(
            1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1).permute(0, 3, 1, 2)
        pos_embed = ops.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False). \
            reshape(1, -1, H * W).permute(0, 2, 1).to(target_dtype)
        return pos_embed

    def construct(self, pixel_values: ms.Tensor) -> ms.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, channel, width, height]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(start_dim=2).swapaxes(1, 2)
        class_embeds = self.class_embedding.broadcast_to((batch_size, 1, -1)).to(target_dtype)
        embeddings = ops.cat([class_embeds, patch_embeds], axis=1)
        embeddings = embeddings + self.position_embedding.to(target_dtype)
        return embeddings


class InternAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: InternVisionConfig):
        super().__init__(config)
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.use_flash_attn = config.use_flash_attn

        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).'
            )

        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Dense(self.embed_dim, 3 * self.embed_dim, has_bias=config.qkv_bias)
        self.attn_drop = nn.Dropout(p=config.attention_dropout)
        self.proj_drop = nn.Dropout(p=config.dropout)

        self.qk_normalization = config.qk_normalization

        self.proj = nn.Dense(self.embed_dim, self.embed_dim)

    def _naive_attn(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.swapaxes(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).swapaxes(1, 2)
            k = self.k_norm(k.swapaxes(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).swapaxes(1, 2)

        attn = ((q * self.scale) @ k.swapaxes(-2, -1))
        attn = ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).swapaxes(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _flash_attn(self, x, key_padding_mask=None, need_weights=False):
        qkv = self.qkv(x)
        # (7, 1025, 3072) --> (7, 1025, 3, 16, 64)
        h = self.num_heads
        qkv = ops.reshape(qkv, (qkv.shape[0], qkv.shape[1], 3, h, qkv.shape[2]//(3 * h)))

        if self.qk_normalization:
            q, k, v = qkv.unbind(2)
            q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
            k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
            qkv = ops.stack([q, k, v], axis=2)

        context, _ = self.inner_attn(
            qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=False
        )

        b, s, h, d = context.shape
        outs = self.proj(ops.reshape(context, (b, s, int(h * d))))
        outs = self.proj_drop(outs)
        return outs

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        x = self._naive_attn(hidden_states) if not has_flash_attn else self._flash_attn(hidden_states)
        return x


class InternMLP(nn.Cell):
    def __init__(self, config: InternVisionConfig):
        super().__init__(config)
        self.config = config
        self.fc1 = nn.Dense(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Dense(config.intermediate_size, config.hidden_size)

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = ops.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class InternVisionEncoderLayer(nn.Cell):
    def __init__(self, config: InternVisionConfig, drop_path_rate: float):
        super().__init__(config)
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.norm_type = config.norm_type

        self.attn = InternAttention(config)
        self.mlp = InternMLP(config)
        self.norm1 = NORM2FN[self.norm_type]([self.embed_dim], epsilon=config.layer_norm_eps)
        self.norm2 = NORM2FN[self.norm_type]([self.embed_dim], epsilon=config.layer_norm_eps)

        self.ls1 = ms.Parameter(config.initializer_factor * ops.ones(self.embed_dim))
        self.ls2 = ms.Parameter(config.initializer_factor * ops.ones(self.embed_dim))
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def construct(
            self,
            hidden_states: ms.Tensor,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor], Optional[Tuple[ms.Tensor]]]:
        """
        Args:
            hidden_states (`Tuple[ms.Tensor, Optional[ms.Tensor]]`): input to the layer of shape `(batch, seq_len, embed_dim)`
        """
        hidden_states = hidden_states + self.drop_path1(self.attn(self.norm1(hidden_states)) * self.ls1)

        hidden_states = hidden_states + self.drop_path2(self.mlp(self.norm2(hidden_states)) * self.ls2)

        return hidden_states


class InternVisionEncoder(nn.Cell):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`InternEncoderLayer`].
    Args:
        config (`InternConfig`):
            The corresponding vision configuration for the `InternEncoder`.
    """

    def __init__(self, config: InternVisionConfig):
        super().__init__(config)
        self.config = config
        # stochastic depth decay rule
        dpr = [x.item() for x in np.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        self.layers = nn.CellList([
            InternVisionEncoderLayer(config, dpr[idx]) for idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = True

    def construct(
            self,
            inputs_embeds,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """

        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
            )
            hidden_states = layer_outputs

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return tuple(v for v in [hidden_states, encoder_states] if v is not None)


class InternVisionModel(PreTrainedModel):
    def __init__(self, config: InternVisionConfig):

        super().__init__(config)
        self.config = config

        self.embeddings = InternVisionEmbeddings(config)
        self.encoder = InternVisionEncoder(config)

    def get_input_embeddings(self):
        return self.embeddings

    def construct(
            self,
            pixel_values: Optional[ms.Tensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_embeds: Optional[ms.Tensor] = None,
    ):

        if pixel_values is None and pixel_embeds is None:
            raise ValueError('You have to specify pixel_values or pixel_embeds')

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        else:
            if len(pixel_values.shape) == 4:
                hidden_states = self.embeddings(pixel_values)
            else:
                raise ValueError(f'wrong pixel_values size: {pixel_values.shape}')
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]

        return (last_hidden_state, pooled_output) + encoder_outputs[1:]


class DummyVitModel(nn.Cell):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.data = ms.Tensor(np.load('./vit_embeds_fp16.npy'))

    def construct(self,
                  pixel_values: Optional[ms.Tensor] = None,
                  output_hidden_states: Optional[bool] = None,
                  return_dict: Optional[bool] = None,
                  pixel_embeds: Optional[ms.Tensor] = None,):
        # ret = Tensor(np.random.randn(7, 1025, 1024), dtype=self.compute_dtype)
        ret = self.data.astype(pixel_values.dtype)
        return tuple([ret])
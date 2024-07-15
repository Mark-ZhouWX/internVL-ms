from internvl.llm.configs import convert_mstype


class InternVisionConfig():
    def __init__(self,
                hidden_size: int = 1024,
                image_size: int = 448,
                patch_size: int = 14,
                compute_dtype: str = "float16",
                attention_dropout: float = 0.0,
                drop_path_rate: float = 0.1,
                dropout: float = 0.0,
                initializer_factor: float = 1.0,
                initializer_range: float = 0.02,
                intermediate_size: int = 4096,
                layer_norm_eps: float = 1e-6,
                model_type: str = "intern_vit_6b",
                norm_type: str = "layer_norm",
                num_attention_heads: int = 16,
                num_channels: int = 3,
                num_hidden_layers: int = 24,
                output_attentions: bool = False,
                output_hidden_states: bool = False,
                qk_normalization: bool = False,
                qkv_bias: bool = True,
                return_dict: bool = True,
                use_flash_attn: bool = False,
                 **kwargs):
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.compute_dtype = convert_mstype(compute_dtype)
        self.attention_dropout = attention_dropout
        self.drop_path_rate = drop_path_rate
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.initializer_factor = initializer_factor
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.model_type = model_type
        self.norm_type = norm_type
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.num_hidden_layers = num_hidden_layers
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.patch_size = patch_size
        self.qk_normalization = qk_normalization
        self.qkv_bias = qkv_bias
        self.return_dict = return_dict
        self.use_flash_attn = use_flash_attn

from typing import Optional

import mindspore as ms
from mindspore import nn

from internvl.intern_vit import InternVisionModel
from internvl.internlm import InternLM2CausalLM
from internvl.internlm_config import InternVLChatConfig
from internvl.llm.base_llm_model import BaseLLMModel
from internvl.utils.layers import Linear


class InternVLChat(BaseLLMModel):
    def __init__(self, config: InternVLChatConfig):
        super().__init__(config)

        self.vision_model = InternVisionModel(config.vision_config)

        self.language_model = InternLM2CausalLM(config.llm_config)

        self.template = config.template

        image_size = config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size
        self.downsample_ratio = config.downsample_ratio



        self.mlp1 = nn.SequentialCell(
            nn.LayerNorm([vit_hidden_size * int(1 / self.downsample_ratio) ** 2], epsilon=1e-5),
            Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            Linear(llm_hidden_size, llm_hidden_size)
        )

    def chat(self, tokenizer, pixel_values, question,
             IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'):

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        from .conversation import get_conv_template

        template = get_conv_template(self.template)
        image_bs = pixel_values.shape[1] # (bs, bs_patch, c, h, w)
        print(f'dynamic ViT batch size: {image_bs}')

        # pad image token as IMG_CONTEXT_TOKEN
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * image_bs + IMG_END_TOKEN
        question = image_tokens + '\n' + question

        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()  # str

        input_ids = tokenizer([query])['input_ids'] # return numpy

        vit_embeds = self.extract_feature(pixel_values) # (1*7, 256, 2048)

        generation_output = self.language_model.generate(input_ids=input_ids, image_features=vit_embeds)

        answer = [generation_output[i][len(input_ids[i]):] for i in range(len(input_ids))]

        decode_output = tokenizer.decode(answer, skip_special_tokens=False)

        response = decode_output[0]

        return response

    def extract_feature(self, pixel_values) -> ms.float_:
        bs, num_patch, channel, h, w = pixel_values.shape  # (1, 7, 3, 448, 448)
        pixel_values = pixel_values.reshape(bs*num_patch, channel, h, w)

        vit_embeds = self.vision_model(pixel_values=pixel_values)[0] # (7, 1025, 1024)
        vit_embeds = vit_embeds[:, 1:, :] # (7, 1024, 1024)
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1) # (7, 32, 32, 1024)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio) # (7, 16, 16, 4096)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1]) # (7, 256, 4096)
        vit_embeds = self.mlp1(vit_embeds) # (7, 256, 2048)

        llm_hidden_dim = vit_embeds.shape[-1]
        vit_embeds = vit_embeds.reshape(bs, -1, llm_hidden_dim)  # (1, 1792, 2048)

        return vit_embeds

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.shape
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3)
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))

        x = x.permute(0, 2, 1, 3)
        return x

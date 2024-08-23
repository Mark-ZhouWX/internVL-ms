from typing import Any, List, Optional, Tuple, Union

import mindspore as ms
import numpy as np
from mindnlp.transformers.models.qwen2 import Qwen2ForCausalLM
from mindspore import nn, ops

from mindnlp.utils import logging
from mindnlp.transformers.modeling_utils import PreTrainedModel
from mindnlp.transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_internvl_chat import InternVLChatConfig
from internvl.conversation import get_conv_template
from .modeling_intern_vit import InternVisionModel

from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM

logger = logging.get_logger(__name__)


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None):
        super().__init__(config)

        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                raise NotImplementedError
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size



        self.mlp1 = nn.SequentialCell(
            nn.LayerNorm([vit_hidden_size * int(1 / self.downsample_ratio) ** 2], epsilon=1e-5),
            nn.Dense(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Dense(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        if hasattr(config, 'system_message'):
            self.system_message = config.system_message
        else:
            self.system_message = self.conv_template.system_message
        self.num_samples = 0

        # TODO add code about lora
        pass

    def construct(
            self,
            pixel_values: ms.Tensor,
            input_ids: ms.Tensor = None,
            img_context_token_index: Tuple[Tuple[int]] = None,
            attention_mask: Optional[ms.Tensor] = None,
            position_ids: Optional[ms.Tensor] = None,
            image_flags: Optional[ms.Tensor] = None,
            past_key_values: Optional[List[ms.Tensor]] = None,
            labels: Optional[ms.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # print(self.base_model.dtype)
        pixel_values = pixel_values.astype(self.vision_model.dtype)

        # image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).copy()  # (bs, seq_len, 2048)

        vit_embeds = self.extract_feature(pixel_values)  # (sum_dyn_patch_along_bs, 256, 2048)
        # vit_embeds = vit_embeds[image_flags == 1]
        # vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        # if ms.communication.get_rank() == 0:
        #     print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        # input_ids = input_ids.reshape(B * N)
        # selected = (input_ids == self.img_context_token_id)
        # try:
        # input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        # image_bs = pixel_values.shape[0]
        # image_token_total_num = self.num_image_token * image_bs
        # img_context_token_start = np.equal(input_ids, self.img_context_token_id).argmax(axis=1)  # (bs,)
        # img_context_token_end = img_context_token_start + image_token_total_num
        # img_context_token_start = img_context_token_start.tolist()
        # img_context_token_end = img_context_token_end.tolist()
        input_embeds[img_context_token_index] = vit_embeds.reshape(-1, C)
        input_embeds = input_embeds.reshape(B, N, C)
        # input_embeds_list = []
        # for i in range(input_ids.shape[0]):
        #     input_embeds_list.append(ops.concat([input_embeds[i, :img_context_token_index[i][0]],
        #                                          vit_embeds[i], input_embeds[i, img_context_token_index[i][1]:]], axis=0))
        # input_embeds = ops.stack(input_embeds_list)
        ignore_flag = False
        # except Exception as e:
        #     vit_embeds = vit_embeds.reshape(-1, C)
        #     print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
        #           f'vit_embeds.shape={vit_embeds.shape}')
        #     n_token = selected.sum()
        #     input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
        #     ignore_flag = True

        # input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs[1]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            # shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if ignore_flag:
                loss = loss * 0.0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return loss, logits
            # return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def chat(self, tokenizer, pixel_values, question, generation_config=None,
             IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'):
        # 1. get input string
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        image_bs = pixel_values.shape[0] # (bs_patch, c, h, w)
        print(f'dynamic ViT batch size: {image_bs}')

        # pad image token as IMG_CONTEXT_TOKEN
        image_token_total_num = self.num_image_token * image_bs
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * image_bs + IMG_END_TOKEN
        question = image_tokens + '\n' + question

        # 2. get text embed
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()  # str

        model_inputs = tokenizer([query], max_length=min(8192, self.config.llm_config.max_position_embeddings),
                                 padding='max_length', return_tensors='np')
        input_ids = model_inputs['input_ids'] # numpy (bs, N)
        attention_mask = model_inputs['attention_mask'] # numpy (bs, N) 1 for valid

        # 3. get img embed
        img_context_token_start = np.equal(input_ids, img_context_token_id).argmax(axis=1) # (bs,)
        img_context_token_end = img_context_token_start + image_token_total_num
        img_context_token_start = img_context_token_start.tolist()
        img_context_token_end = img_context_token_end.tolist()

        input_embeds = self.language_model.get_input_embeddings()(ms.Tensor(input_ids))
        vit_embeds = self.extract_feature(pixel_values)
        B, N, C = input_embeds.shape
        llm_hidden_dim = vit_embeds.shape[-1]
        vit_embeds = vit_embeds.reshape(B, -1, llm_hidden_dim)


        input_embeds_list = []
        for i in range(input_ids.shape[0]):
            input_embeds_list.append(ops.concat([input_embeds[i, :img_context_token_start[i]],
                                          vit_embeds[i], input_embeds[i, img_context_token_end[i]:]], axis=0))
        input_embeds = ops.stack(input_embeds_list)

        # 4. generate
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        generation_config['eos_token_id'] = eos_token_id

        generation_output = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=ms.Tensor(attention_mask, ms.int32),
            output_hidden_states=False,
            return_dict=False,  # return_dict=False is activated due to lack of custom class in graph mode
            use_cache=True,
            **generation_config,
        )

        # 5. decode
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=False)[0]

        return response

    def extract_feature(self, pixel_values) -> ms.float_:
        vit_embeds = self.vision_model(pixel_values=pixel_values)[0] # (7, 1025, 1024)
        vit_embeds = vit_embeds[:, 1:, :] # (7, 1024, 1024)
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1) # (7, 32, 32, 1024)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio) # (7, 16, 16, 4096)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1]) # (7, 256, 4096)
        vit_embeds = self.mlp1(vit_embeds) # (7, 256, 2048)

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

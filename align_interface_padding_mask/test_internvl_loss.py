# from patch.generator_mixin_patch import patch_generator_mixin
# from patch.modeling_attn_mask_utils_patch import patch_attn_mask
import time

import mindspore as ms
import numpy as np
from mindspore import nn
from mindnlp.transformers import Qwen2Tokenizer

from internvl.model.internlm2.tokenization_internlm2 import InternLM2Tokenizer
from internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatModel
from internvl.patch.qwen2_model_patch import patch_qwen2_model

patch_qwen2_model()

import torch

mode = 0
ms.set_context(mode=mode, device_target='Ascend')
print(f'mode: {mode}')

#
model_name = "InternVL2-1B"
ms_dtype = ms.float16
print(f'model name : {model_name}, dtype: {ms_dtype}')

path_dict = {
    "InternVL2-1B": "./pretrained/InternVL2-1B",
    "InternVL2-2B": "./pretrained/InternVL2-2B/",
    "Mini-InternVL-Chat-2B-V1-5": "./pretrained/Mini-InternVL-Chat-2B-V1-5/"
}

tokenizer_dict = {
    "InternVL2-1B": Qwen2Tokenizer,
    "InternVL2-2B": InternLM2Tokenizer,
    "Mini-InternVL-Chat-2B-V1-5": InternLM2Tokenizer
}
# patch_attn_mask()
# patch_generator_mixin()


# path = "/home/hukang/models/internVL/InternVL2-2B/"
# path = "./pretrained/Mini-InternVL-Chat-2B-V1-5/"
path = path_dict[model_name]
tokenzier_cls = tokenizer_dict[model_name]
config = InternVLChatModel.config_class.from_pretrained(path)
config.llm_config.num_hidden_layers = 2  # to reduce memory
print(f"set llm_config.num_hidden_layers to {config.llm_config.num_hidden_layers} to reduce memory")
model = InternVLChatModel.from_pretrained(path, ms_dtype=ms_dtype, config=config)
tokenizer = tokenzier_cls.from_pretrained(path)

model.img_context_token_id = 92546

data = torch.load("./mul_args.txt", map_location=torch.device('cpu'))
pixel_values, input_ids, attention_mask, position_ids, image_flags, past_key_values, labels, use_cache, output_attentions, output_hidden_states, return_dict = data

pixel_values = ms.Tensor(pixel_values.float().cpu().numpy(), dtype=ms_dtype)
print('pixel_values: ' ,pixel_values.shape)
input_ids_np = input_ids
input_ids = ms.Tensor(input_ids.cpu().numpy())
print('input_ids: ', input_ids.shape)
attention_mask = ms.Tensor(attention_mask.cpu().numpy())
print('attention_mask: ', attention_mask.shape)
image_flags = ms.Tensor(image_flags.cpu().numpy())
print('image_flags: ', image_flags.shape)
labels = ms.Tensor(labels.cpu().numpy(), dtype=ms.int32)
print(labels.shape)
for d in data[1:]:
    print(d)


weight = model.trainable_params()
optimizer = nn.SGD(weight, learning_rate=0.0001)

def forward(pixel_values=None, input_ids=None, img_context_token_index=None, attention_mask=None,
                       image_flags=None, labels=None, return_dict=None):
    loss, logits = model(pixel_values=pixel_values, input_ids=input_ids, img_context_token_index=img_context_token_index,
                         attention_mask=attention_mask, image_flags=image_flags, labels=labels, return_dict=False)
    return loss, logits

# grad_fn = ms.ops.GradOperation(get_by_list=True)(model, ms.ParameterTuple(weight))
grad_fn = ms.ops.value_and_grad(forward, grad_position=None, weights=weight, has_aux=True)
# 41 1321

image_bs = pixel_values.shape[0]  # (bs_patch, c, h, w)
print(f'dynamic ViT batch size: {image_bs}')
image_token_total_num = model.num_image_token * image_bs
img_context_token_start = np.equal(input_ids_np.cpu().numpy(), model.img_context_token_id).argmax(axis=1)  # (bs,)
img_context_token_end = img_context_token_start + image_token_total_num
img_context_token_start = img_context_token_start.tolist()
img_context_token_end = img_context_token_end.tolist()
# print(img_context_token_start, img_context_token_end)
img_context_token_index = np.stack([img_context_token_start, img_context_token_end], axis=1).tolist()
img_context_token_index = tuple(tuple(l) for l in img_context_token_index)
# img_context_token_index = ((41, 1321),)
print(img_context_token_index)
# img_context_token_index = ms.mutable(img_context_token_index)

for k in range(20):
    t = time.time()
    (loss, logits), gradients = grad_fn(pixel_values, input_ids, img_context_token_index, attention_mask,
                           image_flags, labels, False)
    # print(gradients)
    optimizer(gradients)
    t = time.time() - t
    print(f"loss of the {k} step: {loss}, step time: {t:.2f}s")
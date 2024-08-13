from patch.generator_mixin_patch import patch_generator_mixin
from patch.modeling_attn_mask_utils_patch import patch_attn_mask

import mindspore as ms
from mindspore import nn

from internvl_chat.modeling_internvl_chat import InternVLChatModel
from internvl_chat.tokenization_internlm2 import InternLM2Tokenizer

import torch

mode = 0
ms.set_context(mode=mode, device_target='Ascend')
print(f'mode: {mode}')

patch_attn_mask()
patch_generator_mixin()

path = "/home/hukang/models/internVL/InternVL2-2B"

model = InternVLChatModel.from_pretrained(path)
tokenizer = InternLM2Tokenizer.from_pretrained(path)

model.img_context_token_id = 92546

data = torch.load("/home/hukang/data/mul_args.txt", map_location=torch.device('cpu'))
pixel_values, input_ids, attention_mask, position_ids, image_flags, past_key_values, labels, use_cache, output_attentions, output_hidden_states, return_dict = data

pixel_values = ms.Tensor(pixel_values.float().cpu().numpy(), dtype=ms.float16)
print(pixel_values.shape)
input_ids = ms.Tensor(input_ids.cpu().numpy())
print(input_ids.shape)
attention_mask = ms.Tensor(attention_mask.cpu().numpy())
print(attention_mask.shape)
image_flags = ms.Tensor(image_flags.cpu().numpy())
print(image_flags.shape)
labels = ms.Tensor(labels.cpu().numpy(), dtype=ms.int32)
print(labels.shape)
for d in data[1:]:
    print(d)


weight = model.trainable_params()
optimizer = nn.SGD(weight, learning_rate=0.001)

def forward(pixel_values=None, input_ids=None, img_context_token_index=None, attention_mask=None,
                       image_flags=None, labels=None, return_dict=None):
    loss, logits = model(pixel_values=pixel_values, input_ids=input_ids, img_context_token_index=img_context_token_index,
                         attention_mask=attention_mask, image_flags=image_flags, labels=labels, return_dict=False)
    return loss, logits

# grad_fn = ms.ops.GradOperation(get_by_list=True)(model, ms.ParameterTuple(weight))
grad_fn = ms.ops.value_and_grad(forward, grad_position=None, weights=weight, has_aux=True)
# 41 1321
img_context_token_index = ((41, 1321),)
for k in range(10):
    (loss, logits), gradients = grad_fn(pixel_values, input_ids, img_context_token_index, attention_mask,
                           image_flags, labels, False)
    print(f"loss of the {k} step ", loss)
    # print(gradients)
    optimizer(gradients)
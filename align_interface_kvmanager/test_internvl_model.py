from patch.generator_mixin_patch import patch_generator_mixin

patch_generator_mixin()

import mindspore as ms

from internvl.data_process import load_image
from internvl_chat.modeling_internvl_chat import InternVLChatModel
from internvl_chat.tokenization_internlm2 import InternLM2Tokenizer
mode = 0
ms.set_context(mode=mode, device_target='Ascend')
print(f'mode: {mode}')

path = "./Mini-InternVL-Chat-2B-V1-5-raw"

model = InternVLChatModel.from_pretrained(path)
tokenizer = InternLM2Tokenizer.from_pretrained(path)

# set the max number of tiles in `max_num`
pixel_value = load_image('./examples/image1.jpg', max_num=6)
pixel_values = ms.Tensor(pixel_value, dtype=ms.float16)

generation_config = dict(
    num_beams=1,
    max_new_tokens=512,
    do_sample=False,
)

# for name, value in model.named_parameters():
#     print(name, value.shape)
# exit()
# single-round single-image conversation
question = "请详细描述图片" # Please describe the picture in detail
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(question, response)

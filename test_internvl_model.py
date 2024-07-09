import numpy as np
import yaml

import mindspore as ms
from mindspore import Tensor

from internvl.internlm_config import InternVLChatConfig
from internvl.internlm_tokenizer import InternLM2Tokenizer
from internvl.internvl import InternVLChat

if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    ms.set_context(mode=1, device_target="Ascend")
    config_path = './configs/mini_internvl_chat_2b_v1_5.yaml'

    with open(config_path) as stream:
        data = yaml.safe_load(stream)
    intern_config = InternVLChatConfig(**data['model'])
    model = InternVLChat(intern_config)
    model.load_checkpoint(intern_config)

    # for k, v in model.parameters_and_names():
    #     print(k, v.shape)
    # exit()

    tokenizer = InternLM2Tokenizer.from_pretrained("MiniInternLM2Chat2B")

    question = "请详细描述图片"

    pixel_values = Tensor(np.random.randn(1, 7, 3, 448, 448)) # TODO to change to real image

    outputs = model.chat(tokenizer, pixel_values, question)


    print(question, outputs)
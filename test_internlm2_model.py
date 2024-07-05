import logging
from logging import Logger

import yaml

import mindspore as ms
import numpy as np
import numpy as np

from internvl.internlm import InternLM2CausalLM
from internvl.internlm_config import InternLM2Config
from internvl.internlm_tokenizer import InternLM2Tokenizer

if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    ms.set_context(mode=1, device_target="Ascend")
    config_path = './configs/mini_internvl_chat_2b_v1_5.yaml'

    with open(config_path) as stream:
        data = yaml.safe_load(stream)
    intern_config = InternLM2Config(**data['llm_model']) # model_ckpt in data
    model = InternLM2CausalLM(intern_config)
    model.load_checkpoint(intern_config)
    # tokenizer = InternLM2Tokenizer(**data['processor']['tokenizer'])
    tokenizer = InternLM2Tokenizer.from_pretrained("MiniInternLM2Chat2B")
    words = ["translate the English to the Chinese: UN Chief Says There Is No Military Solution in Syria"]
    words = [f"""<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n""" for query in words]
    # input_ids = tokenizer(words, max_length=intern_config.seq_length, padding='max_length')['input_ids']
    input_ids = tokenizer(words)['input_ids']

    outputs = model.generate(input_ids=input_ids)

    response = tokenizer.decode(outputs, skip_special_tokens=True)

    print(response)
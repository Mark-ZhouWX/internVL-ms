# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Convert internlm weight.
Support huggingface format format.
"""

import os
import json
import argparse

import mindspore as ms


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def name_replace_vit(name: str):
    name = name.replace('norm1.weight', 'norm1.gamma')
    name = name.replace('norm1.bias', 'norm1.beta')
    name = name.replace('norm2.weight', 'norm2.gamma')
    name = name.replace('norm2.bias', 'norm2.beta')
    return name

def name_replace_llm(name: str):
    """replace hf param name to ms."""
    name = name.replace('language_model.model.tok_embeddings.weight', 'language_model.transformer.wte.embedding_weight')
    name = name.replace('language_model.model', 'language_model.transformer')
    name = name.replace('language_model.output', 'language_model.lm_head')

    return name

def name_replace_other(name: str):
    name = name.replace('mlp1.0.weight', 'mlp1.0.gamma')
    name = name.replace('mlp1.0.bias', 'mlp1.0.beta')
    return name

def convert_hf_ckpt(ckpt_dir, output_path, dtype=ms.float16):
    """convert hf weight to ms."""
    print(f"Trying to convert huggingface checkpoint in '{ckpt_dir}'.", flush=True)
    try:
        from transformers import AutoModel
        model_hf = AutoModel.from_pretrained(
            ckpt_dir,
            low_cpu_mem_usage=True,
            trust_remote_code=True)
    # pylint: disable=W0703
    except Exception as e:
        print(f"Do not find huggingface checkpoint in '{ckpt_dir}', Error {e}.", flush=True)
        return False

    ckpt_list = []
    for name, value in model_hf.named_parameters():
        if name.startswith("vision_model"):
            name = name_replace_vit(name)
        elif name.startswith("language_model"):
            name = name_replace_llm(name)
        else: # mlp
            name = name_replace_other(name)

        value = value.detach().numpy()
        print(f'\rprocessing parameter: {name} {value.shape}     ', end='', flush=True)
        ckpt_list.append({'name': name, 'data': ms.Tensor(value, dtype=dtype)})

    ms.save_checkpoint(ckpt_list, output_path)
    print(f"\rConvert huggingface checkpoint finished, the mindspore checkpoint is saved in '{output_path}'.", flush=True)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_ckpt_dir', default='/home/zhouwuxing/huggingface_download/models--OpenGVLab--Mini-InternVL-Chat-2B-V1-5/')
    parser.add_argument('--mindspore_ckpt_path', default='./MiniInternLM2Chat2B/mini-internvl-chat-2b.ckpt')
    args = parser.parse_args()
    convert_hf_ckpt(ckpt_dir=args.torch_ckpt_dir, output_path=args.mindspore_ckpt_path)
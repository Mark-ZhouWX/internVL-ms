import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from collections import OrderedDict


# If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
path = '/data1/hukang/models/InternVL2-2B'
#path = '/data1/hukang/models/Mini-InternVL-Chat-2B-V1-5'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

model.img_context_token_id = 92546

data = torch.load("mul_args.txt")
pixel_values, input_ids, attention_mask, position_ids, image_flags, past_key_values, labels, use_cache, output_attentions, output_hidden_states, return_dict = data

pixel_values = pixel_values.to(torch.float16)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for k in range(10):
    optimizer.zero_grad()
    loss = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask,
                       image_flags=image_flags, labels=labels, return_dict=True).loss
    loss.backward()
    print(f'loss of the {k} step ', loss)

    #grads = OrderedDict()
    #for n, p in model.named_parameters():
    #    grads[n] = p.grad
    #print(grads)
    optimizer.step()

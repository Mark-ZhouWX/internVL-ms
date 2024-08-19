import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from collections import OrderedDict


#
from internvl.model import InternVLChatModel

model_name = "InternVL2-1B"

torch_dtype = torch.float16

num_transformer_layer = 2

print(f'model_name: {model_name}, dtype: {torch_dtype}')

path_dict = {
    "InternVL2-1B": "./pretrained/InternVL2-1B",
    "InternVL2-2B": "/home/hukang/models/internVL/InternVL2-2B/",
    "Mini-InternVL-Chat-2B-V1-5": "./pretrained/Mini-InternVL-Chat-2B-V1-5/"
}

# tokenizer_dict = {
#     "InternVL2-1B": Qwen2Tokenizer,
#     "InternVL2-2B": InternLM2Tokenizer,
#     "Mini-InternVL-Chat-2B-V1-5": InternLM2Tokenizer
# }

# If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
path = path_dict[model_name]
#path = '/data1/hukang/models/Mini-InternVL-Chat-2B-V1-5'
config = InternVLChatModel.config_class.from_pretrained(path)
config.llm_config.num_hidden_layers = num_transformer_layer  # to reduce memory
print(f"set llm_config.num_hidden_layers to {config.llm_config.num_hidden_layers} to reduce memory")
model = InternVLChatModel.from_pretrained(
    path,
    torch_dtype=torch_dtype,
    config=config,
    low_cpu_mem_usage=True,
    trust_remote_code=True).cuda()


tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

model.img_context_token_id = 92546

data = torch.load("mul_args.txt")
pixel_values, input_ids, attention_mask, position_ids, image_flags, past_key_values, labels, use_cache, output_attentions, output_hidden_states, return_dict = data

pixel_values = pixel_values.to(torch_dtype)

optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

image_bs = pixel_values.shape[0]  # (bs_patch, c, h, w)
print(f'dynamic ViT batch size: {image_bs}')

# pad image token as IMG_CONTEXT_TOKEN
image_token_total_num = model.num_image_token * image_bs
img_context_token_start = np.equal(input_ids.cpu().numpy(), model.img_context_token_id).argmax(axis=1)  # (bs,)
img_context_token_end = img_context_token_start + image_token_total_num
img_context_token_start = img_context_token_start.tolist()
img_context_token_end = img_context_token_end.tolist()
print(img_context_token_start, img_context_token_end)

for k in range(20):
    optimizer.zero_grad()
    loss = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask,
                       image_flags=image_flags, labels=labels, return_dict=True).loss
    loss.backward()
    print(f'loss of the {k} step ', loss.item())

    #grads = OrderedDict()
    #for n, p in model.named_parameters():
    #    grads[n] = p.grad
    #print(grads)
    optimizer.step()

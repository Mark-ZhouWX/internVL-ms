import numpy as np
from mindspore import nn, Tensor

from internvl.intern_vit_config import InternVisionConfig


class InternVisionModel(nn.Cell):
    def __init__(self, config: InternVisionConfig):

        super().__init__()
        self.compute_dtype = config.compute_dtype
        pass

    def construct(self, pixel_values):
        # ret = Tensor(np.random.randn(7, 1025, 1024), dtype=self.compute_dtype)
        ret = Tensor(np.load('./vit_embeds_fp16.npy'), dtype=self.compute_dtype)
        return ret


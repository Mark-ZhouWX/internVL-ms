from internvl.llm.configs import convert_mstype


class InternVisionConfig():
    def __init__(self,
                 hidden_size,
                 image_size=448,
                 patch_size=14,
                 compute_dtype: str = "float16",
                 **kwargs):
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.compute_dtype = convert_mstype(compute_dtype)
        pass

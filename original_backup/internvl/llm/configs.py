import copy
import os

import yaml

import mindspore.common.dtype as mstype


def convert_mstype(ms_type: str = "float16"):
    """Convert the string type to MindSpore type."""
    if isinstance(ms_type, mstype.Float):
        return ms_type
    if ms_type == "float16":
        return mstype.float16
    if ms_type == "bfloat16":
        return mstype.bfloat16
    if ms_type == "float32":
        return mstype.float32
    raise KeyError(f"Supported data type keywords include: [float16, float32, bfloat16], but get {ms_type}")


class LLMConfig(dict):
    def __init__(self, *args, **kwargs):
        super(LLMConfig, self).__init__()
        cfg_dict = {}

        # load from file
        for arg in args:
            if isinstance(arg, str):
                if arg.endswith("yaml") or arg.endswith("yml"):
                    raw_dict = LLMConfig._file2dict(arg)
                    cfg_dict.update(raw_dict)

        # load dictionary configs
        if kwargs is not None:
            cfg_dict.update(kwargs)

        LLMConfig._dict2config(self, cfg_dict)

    def __getattr__(self, key):
        """Get a object attr by its `key`

        Args:
            key (str) : the name of object attr.

        Returns:
            attr of object that name is `key`
        """
        if key not in self:
            return None

        return self[key]

    def __setattr__(self, key, value):
        """Set a object value `key` with `value`

        Args:
            key (str) : The name of object attr.
            value : the `value` need to set to the target object attr.
        """
        self[key] = value

    def __delattr__(self, key):
        """Delete a object attr by its `key`.

        Args:
            key (str) : The name of object attr.
        """
        del self[key]

    def __deepcopy__(self):
        """Deep copy operation on arbitrary LLMConfig objects.

        Returns:
            LLMConfig : The deep copy of the given LLMConfig object.
        """
        config = LLMConfig()
        for key in self.keys():
            config.__setattr__(copy.deepcopy(key), copy.deepcopy(self.__getattr__(key)))
        return config

    @staticmethod
    def _file2dict(filename=None):
        """Convert config file to dictionary.

        Args:
            filename (str) : config file.
        """
        if filename is None:
            raise NameError("This {} cannot be empty.".format(filename))

        filepath = os.path.realpath(filename)
        with open(filepath, encoding="utf-8") as fp:
            cfg_dict = yaml.load(fp, yaml.Loader)

        return cfg_dict

    @staticmethod
    def _dict2config(config, dic):
        """Convert dictionary to config.

        Args:
            config : Config object
            dic (dict) : dictionary
        Returns:

        Exceptions:

        """
        if isinstance(dic, dict):
            for key, value in dic.items():
                if isinstance(value, dict):
                    sub_config = LLMConfig()
                    dict.__setitem__(config, key, sub_config)
                    LLMConfig._dict2config(sub_config, value)
                else:
                    config[key] = dic[key]


class BaseConfig(dict):
    def __init__(self, **kwargs):
        super(BaseConfig, self).__init__()
        self.update(kwargs)

    def __getattr__(self, key):
        if key not in self:
            return None
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]

    @classmethod
    def from_pretrained(cls, yaml_name_or_path, **kwargs):
        """
        From pretrain method, which instantiates a config by yaml name or path.

        Args:
            yaml_name_or_path (str): A supported model path to model config (.yaml).

        Returns:
            A model config, which inherited from BaseConfig.
        """
        pretrained_model_name_or_path = kwargs.pop("pretrained_model_name_or_path", None)
        if pretrained_model_name_or_path is not None:
            yaml_name_or_path = pretrained_model_name_or_path

        if not isinstance(yaml_name_or_path, str):
            raise TypeError(f"yaml_name_or_path should be a str, but got {type(yaml_name_or_path)}.")

        if os.path.exists(yaml_name_or_path):
            if not yaml_name_or_path.endswith(".yaml"):
                raise ValueError(f"{yaml_name_or_path} should be a .yaml file for model config.")

            config_args = LLMConfig(yaml_name_or_path)
        else:
            raise ValueError(f"{yaml_name_or_path} is not a supported model type or a valid path to model config.")
        config_args.model.update(**kwargs)
        config = config_args.model
        return config



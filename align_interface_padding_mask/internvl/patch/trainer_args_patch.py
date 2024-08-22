from dataclasses import asdict

import mindspore
from mindnlp.engine import TrainingArguments


def TrainingArguments__str__(self):
    self_as_dict = asdict(self)

    # Remove deprecated arguments. That code should be removed once
    # those deprecated arguments are removed from TrainingArguments. (TODO: v5)
    # del self_as_dict["per_gpu_train_batch_size"]
    # del self_as_dict["per_gpu_eval_batch_size"]

    self_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in self_as_dict.items()}

    attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
    return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"


def patch_trainer_args_str():
    TrainingArguments.__str__ = TrainingArguments__str__
    # Trainer.compute_loss = compute_loss
    # Trainer._prepare_inputs = _prepare_inputs


from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.mint.optim import AdamW

_optim_adamw_opt = C.MultitypeFuncGraph("optim_adamw_opt")
hyper_map = C.HyperMap()

@_optim_adamw_opt.register("Function", "Float", "Float", "Float", "Float", "Float", "Tensor", "Bool", "Bool", "Tensor",
                           "Tensor", "Tensor", "Tensor", "Tensor")
def _run_optim_adamw_opt(opt, beta1, beta2, lr, eps, weight_decay, step, amsgrad, maximize, parameters, grads, exp_avg,
                         exp_avg_sq, max_exp_avg_sq):
    """Apply adamw optimizer to the weight parameter."""
    success = True
    opt(parameters, exp_avg, exp_avg_sq, max_exp_avg_sq, P.Cast()(grads, F.dtype(parameters)), step, lr, beta1, beta2,
        weight_decay, eps, amsgrad, maximize)
    return success

def construct(self, gradients):
    self.assignadd(self.state_step, self.increase_tensor)
    for group_id, group in enumerate(self.param_groups):
        beta1, beta2 = group['betas']
        maximize = group.get("maximize")
        start_id = self.group_start_id[group_id]
        end_id = self.group_start_id[group_id + 1]
        lr = group.get("lr")
        grads = tuple(gradients[start_id: end_id])
        print("----ignore---", lr.value())

        self.hyper_map(F.partial(_optim_adamw_opt, self.adamw_opt, beta1, beta2, float(lr),
                                 group.get("eps"), group.get("weight_decay"), self.state_step,
                                 group.get("amsgrad"), maximize),
                       self.parameters[start_id: end_id], grads, self.exp_avg[start_id: end_id],
                       self.exp_avg_sq[start_id: end_id], self.max_exp_avg_sq[start_id: end_id])
    return True



def patch_adamw():
    AdamW.construct = construct
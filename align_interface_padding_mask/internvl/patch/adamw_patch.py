from mindspore.ops import functional as F, composite as C, operations as P
from mindspore import ops
from mindspore.experimental.optim import AdamW


def prepare_func(lr, weight_decay, state_step, beta1, beta2):
    weight_decay_new = 1 - lr * weight_decay
    bias_correction1 = 1 - op_pow(beta1, state_step)
    bias_correction2 = 1 - op_pow(beta2, state_step)
    step_size = lr / bias_correction1
    bias_correction2_sqrt = op_sqrt(bias_correction2)
    return weight_decay_new, step_size, bias_correction2_sqrt


op_mul = P.Mul()
op_pow = P.Pow()
op_sqrt = P.Sqrt()
op_maximum = P.Maximum()
hyper_map = C.HyperMap()
_adamw_opt = C.MultitypeFuncGraph("adamw_opt")
@_adamw_opt.register("Tensor", "Tensor", "Bool", "Float", "Tensor", "Float", "Float", "Tensor", "Tensor",
                     "Tensor", "Tensor", "Tensor")
def _run_adamw_opt(weight_decay_new, step_size, amsgrad, eps, bias_correction2_sqrt, beta1, beta2, param, grad,
                   exp_avg, exp_avg_sq, max_exp_avg_sq):
    """Apply adamw optimizer to the weight parameter."""
    success = True
    next_param = op_mul(param, weight_decay_new)
    F.assign(exp_avg, op_mul(exp_avg, beta1) + op_mul(grad, 1 - beta1))
    F.assign(exp_avg_sq, ops.addcmul(op_mul(exp_avg_sq, beta2), grad, grad, 1 - beta2))
    if amsgrad:
        next_max_exp_avg = op_maximum(max_exp_avg_sq, exp_avg_sq)
        denom = op_sqrt(next_max_exp_avg) / bias_correction2_sqrt + eps
        F.assign(max_exp_avg_sq, next_max_exp_avg)
    else:
        denom = op_sqrt(exp_avg_sq) / bias_correction2_sqrt + eps
    return_param = next_param - op_mul(exp_avg / denom, step_size)
    F.assign(param, return_param.astype(param.dtype))   # 无法自动转换，需要显示指定dtype
    return success

def implementation(self, lr, weight_decay, beta1, beta2, amsgrad, eps, grads, start_id, end_id):
    """Extract the common computing part for acceleration"""
    weight_decay_new, step_size, bias_correction2_sqrt = prepare_func(lr, weight_decay,
                                                                      self.state_step, beta1, beta2)
    self.hyper_map(F.partial(_adamw_opt, weight_decay_new, step_size, amsgrad,
                             eps, bias_correction2_sqrt, beta1, beta2),
                   self.parameters[start_id: end_id], grads, self.exp_avg[start_id: end_id],
                   self.exp_avg_sq[start_id: end_id], self.max_exp_avg_sq[start_id: end_id])
    return True



def patch_adamw():
    AdamW.implementation = implementation
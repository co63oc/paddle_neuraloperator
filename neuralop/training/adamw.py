# copy dependencies from transformers/optimization.py
import math
import warnings  # noqa
from typing import Callable
from typing import Iterable
from typing import List  # noqa
from typing import Tuple
from typing import Union  # noqa

import paddle

from neuralop import paddle_aux


class AdamW(paddle.optimizer.Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters
    ----------
    params : Iterable[nn.parameter.Parameter]
        Iterable of parameters to optimize or dictionaries defining parameter groups.
    lr : float, *optional*, defaults to 0.001
        The learning rate to use.
    betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
        Adam's betas parameters (b1, b2).
    eps (`float`, *optional*, defaults to 1e-06):
        Adam's epsilon for numerical stability.
    weight_decay (`float`, *optional*, defaults to 0.0):
        Decoupled weight decay to apply.
    correct_bias (`bool`, *optional*, defaults to `True`):
        Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
    """

    def __init__(
        self,
        params: Iterable[paddle.base.framework.EagerParamBase],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-06,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        self.param_groups = [defaults]
        super().__init__(parameters=params, learning_rate=lr, weight_decay=weight_decay)
        self.state = {}
        for p in params:
            self.state[p] = {}

    @paddle.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # param_group save some default parameter, _param_groups save input parameter
        group = self.param_groups[0]
        for p in self._param_groups:
            if not hasattr(p, "grad"):
                continue
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse():
                raise RuntimeError(
                    "Adam does not support sparse gradients, please consider SparseAdam instead"
                )
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
            if "dim" not in group:
                group["dim"] = 2
            # State initialization
            if "exp_avg" not in state:
                # Exponential moving average of gradient values
                state["exp_avg"] = paddle.zeros_like(x=grad)
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = paddle.zeros_like(x=grad)
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            beta1, beta2 = group["betas"]
            state["step"] += 1
            # Decay the first and second moment running average coefficient
            # In-place operations to update the averages at the same time
            exp_avg.multiply_(y=paddle.to_tensor(beta1)).add_(y=(1.0 - beta1) * grad)
            if paddle.is_complex(x=grad):
                exp_avg_sq.multiply_(y=paddle.to_tensor(beta2)).add_(
                    (1.0 - beta2) * grad * grad.conj()
                )
            else:
                exp_avg_sq.multiply_(y=paddle.to_tensor(beta2)).add_((1.0 - beta2) * grad * grad)
            denom = paddle_aux.sqrt(exp_avg_sq).add_(y=paddle.to_tensor(group["eps"]))
            step_size = group["lr"]
            if group["correct_bias"]:  # No bias correction for Bert
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
            # compute norm gradient
            norm_grad = exp_avg / denom
            p.add_(y=(-step_size * norm_grad))
            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want to decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            # Add weight decay at the end (fixed version)
            if group["weight_decay"] > 0.0:
                p.add_(y=(-group["lr"] * group["weight_decay"] * p))
        return loss

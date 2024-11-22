import paddle
import math
import warnings
from typing import Callable, Iterable, Tuple, Union, List


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

    def __init__(self, params: Iterable[paddle.base.framework.
        EagerParamBase.from_tensor], lr: float=0.001, betas: Tuple[float,
        float]=(0.9, 0.999), eps: float=1e-06, weight_decay: float=0.0,
        correct_bias: bool=True):
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr} - should be >= 0.0')
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f'Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)'
                )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f'Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)'
                )
        if not 0.0 <= eps:
            raise ValueError(f'Invalid epsilon value: {eps} - should be >= 0.0'
                )
        defaults = {'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay':
            weight_decay, 'correct_bias': correct_bias}
        super().__init__(params, defaults)

    @paddle.no_grad()
    def step(self, closure: Callable=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse():
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead'
                        )
                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                if 'dim' not in group:
                    group['dim'] = 2
                if 'exp_avg' not in state:
                    state['exp_avg'] = paddle.zeros_like(x=grad)
                    state['exp_avg_sq'] = paddle.zeros_like(x=grad)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.multiply_(y=paddle.to_tensor(beta1)).add_(y=paddle.
                    to_tensor((1.0 - beta1) * grad))
                if paddle.is_complex(x=grad):
                    exp_avg_sq.multiply_(y=paddle.to_tensor(beta2)).add_((
                        1.0 - beta2) * grad * grad.conj())
                else:
                    exp_avg_sq.multiply_(y=paddle.to_tensor(beta2)).add_((
                        1.0 - beta2) * grad * grad)
                denom = exp_avg_sq.sqrt().add_(y=paddle.to_tensor(group['eps'])
                    )
                step_size = group['lr']
                if group['correct_bias']:
                    bias_correction1 = 1.0 - beta1 ** state['step']
                    bias_correction2 = 1.0 - beta2 ** state['step']
                    step_size = step_size * math.sqrt(bias_correction2
                        ) / bias_correction1
                norm_grad = exp_avg / denom
                p.add_(y=paddle.to_tensor(-step_size * norm_grad))
                if group['weight_decay'] > 0.0:
                    p.add_(y=paddle.to_tensor(-group['lr'] * group[
                        'weight_decay'] * p))
        return loss

import paddle
import pytest

from ..adamw import AdamW


@pytest.mark.parametrize("adam_optimizer_cls", [AdamW])
def test_correct_complex_adam_momentum(adam_optimizer_cls):
    # param = x * 2j
    x = paddle.randn(shape=(3, 3), dtype="float64")
    param = paddle.base.framework.EagerParamBase.from_tensor(
        tensor=((0.0 + 1.0j) * x).to("complex64")
    )
    optimizer = adam_optimizer_cls(params=[param], betas=(0.5, 0.5))
    loss = paddle.as_real(x=param * param.conj()).sum()
    # grad x^2 = 2x, grads are all 0 + 2j * x
    loss.backward()
    optimizer.step()

    # momentum value should be elemwise (2jx * -2jx * (1 - 0.5)) = 4x**2 * 0.5 = 2x**2
    # exp_avg_sq should be empty, meaning it is just momentum * (1-beta2)
    momentum = optimizer.state[param]["exp_avg_sq"]
    assert paddle.allclose(
        x=momentum.as_real(), y=(2 * x**2).to("complex64").as_real()
    ).item(), ""

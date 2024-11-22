import paddle
import pytest
from ..adamw import AdamW


@pytest.mark.parametrize('adam_optimizer_cls', [AdamW])
def test_correct_complex_adam_momentum(adam_optimizer_cls):
    x = paddle.randn(shape=(3, 3), dtype='float64')
    param = paddle.base.framework.EagerParamBase.from_tensor(tensor=((0.0 +
        1.0j) * x).to('complex64'))
    optimizer = adam_optimizer_cls(params=[param], betas=(0.5, 0.5))
    loss = paddle.as_real(x=param * param.conj()).sum()
    loss.backward()
    optimizer.step()
    momentum = optimizer.state[param]['exp_avg_sq']
    assert paddle.allclose(x=momentum, y=(2 * x ** 2).to('complex64')).item(
        ), ''

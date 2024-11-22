import paddle
import pytest
from tensorly import tenalg
tenalg.set_backend('einsum')
from ..fnogno import FNOGNO


@pytest.mark.parametrize('gno_transform_type', ['linear',
    'nonlinear_kernelonly', 'nonlinear'])
@pytest.mark.parametrize('fno_n_modes', [(8,), (8, 8), (8, 8, 8)])
@pytest.mark.parametrize('gno_batched', [False, True])
@pytest.mark.parametrize('gno_coord_embed_dim', [None, 32])
@pytest.mark.parametrize('fno_norm', [None, 'ada_in'])
def test_fnogno(gno_transform_type, fno_n_modes, gno_batched,
    gno_coord_embed_dim, fno_norm):
>>>>>>    if torch.has_cuda:
        device = paddle.CUDAPlace(int('cuda:0'.replace('cuda:', '')))
    else:
        device = paddle.CPUPlace()
    in_channels = 3
    out_channels = 2
    batch_size = 4
    n_dim = len(fno_n_modes)
    model = FNOGNO(in_channels=in_channels, out_channels=out_channels,
        gno_radius=0.2, gno_coord_dim=n_dim, gno_coord_embed_dim=
        gno_coord_embed_dim, gno_transform_type=gno_transform_type,
        gno_batched=gno_batched, fno_n_modes=fno_n_modes, fno_norm='ada_in',
        fno_ada_in_features=4).to(device)
    in_p_shape = [32] * n_dim
    in_p_shape.append(n_dim)
    in_p = paddle.randn(shape=in_p_shape)
    out_p = paddle.randn(shape=[100, n_dim])
    f_shape = [32] * n_dim
    f_shape.append(in_channels)
    if gno_batched:
        f_shape = [batch_size] + f_shape
    f = paddle.randn(shape=f_shape)
    out_1 = f
    out_1.stop_gradient = not True
    out_1
    ada_in = paddle.randn(shape=[1])
    out = model(in_p, out_p, f, ada_in)
    if gno_batched:
        assert list(tuple(out.shape)) == [batch_size, 100, out_channels]
    else:
        assert list(tuple(out.shape)) == [100, out_channels]
    if gno_batched:
        loss = out[0].sum()
    else:
        loss = out.sum()
    loss.backward()
    n_unused_params = 0
    for param in model.parameters():
        if param.grad is None:
            n_unused_params += 1
    assert n_unused_params == 0, f'{n_unused_params} parameters were unused!'
    if gno_batched:
        assert not f.grad[1:].nonzero().astype('bool').any()

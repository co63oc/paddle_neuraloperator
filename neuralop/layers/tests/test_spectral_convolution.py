import paddle
import pytest
from tltorch import FactorizedTensor
from ..spectral_convolution import SpectralConv3d, SpectralConv2d, SpectralConv1d, SpectralConv


@pytest.mark.parametrize('factorization', ['Dense', 'CP', 'Tucker', 'TT'])
@pytest.mark.parametrize('implementation', ['factorized', 'reconstructed'])
@pytest.mark.parametrize('separable', [False, True])
@pytest.mark.parametrize('dim', [1, 2, 3, 4])
@pytest.mark.parametrize('complex_data', [True, False])
def test_SpectralConv(factorization, implementation, separable, dim,
    complex_data):
    """Test for SpectralConv of any order
    
    Compares Factorized and Dense convolution output
    Verifies that a dense conv and factorized conv with the same weight produce the same output

    Checks the output size

    Verifies that dynamically changing the number of Fourier modes doesn't break the conv
    """
    modes = 10, 8, 6, 6
    incremental_modes = 6, 6, 4, 4
    conv = SpectralConv(3, 3, modes[:dim], n_layers=1, bias=False,
        implementation=implementation, factorization=factorization,
        separable=separable, complex_data=complex_data)
    conv_dense = SpectralConv(3, 3, modes[:dim], n_layers=1, bias=False,
        implementation='reconstructed', factorization=None, complex_data=
        complex_data)
    if complex_data:
        x = paddle.randn(shape=[2, 3, *((12,) * dim)], dtype='complex64')
    else:
        x = paddle.randn(shape=[2, 3, *((12,) * dim)])
    assert paddle.is_complex(x=conv._get_weight(0))
    assert paddle.is_complex(x=conv_dense._get_weight(0))
    if not separable:
        conv_dense.weight[0] = FactorizedTensor.from_tensor(conv.weight[0].
            to_tensor(), rank=None, factorization='ComplexDense')
    res_dense = conv_dense(x)
    res = conv(x)
    res_shape = tuple(res.shape)
    if not separable:
        assert paddle.allclose(x=res_dense, y=res).item(), ''
    conv.n_modes = incremental_modes[:dim]
    res = conv(x)
    assert res_shape == tuple(res.shape)
    block = SpectralConv(3, 4, modes[:dim], n_layers=1,
        output_scaling_factor=0.5)
    x = paddle.randn(shape=[2, 3, *((12,) * dim)])
    res = block(x)
    assert list(tuple(res.shape)[2:]) == [12 // 2] * dim
    block = SpectralConv(3, 4, modes[:dim], n_layers=1, output_scaling_factor=2
        )
    x = paddle.randn(shape=[2, 3, *((12,) * dim)])
    res = block(x)
    assert tuple(res.shape)[1] == 4
    assert list(tuple(res.shape)[2:]) == [12 * 2] * dim


def test_SpectralConv_output_scaling_factor():
    """Test SpectralConv with upsampled or downsampled outputs
    """
    modes = 4, 4, 4, 4
    size = [6] * 4
    for dim in [1, 2, 3, 4]:
        conv = SpectralConv(3, 3, modes[:dim], n_layers=1,
            output_scaling_factor=0.5)
        x = paddle.randn(shape=[2, 3, *size[:dim]])
        res = conv(x)
        assert list(tuple(res.shape)[2:]) == [(m // 2) for m in size[:dim]]
        conv = SpectralConv(3, 3, modes[:dim], n_layers=1,
            output_scaling_factor=2)
        x = paddle.randn(shape=[2, 3, *size[:dim]])
        res = conv(x)
        assert list(tuple(res.shape)[2:]) == [(m * 2) for m in size[:dim]]


@pytest.mark.parametrize('factorization', ['ComplexCP', 'ComplexTucker'])
@pytest.mark.parametrize('implementation', ['factorized', 'reconstructed'])
def test_SpectralConv3D(factorization, implementation):
    """Compare generic SpectralConv with hand written SpectralConv2D
    
    Verifies that a dense conv and factorized conv with the same weight produce the same output
    Note that this implies the order in which the conv is done in the manual implementation matches the automatic one, 
    take with a grain of salt
    """
    conv = SpectralConv(3, 6, (4, 4, 3), n_layers=1, bias=False,
        implementation=implementation, factorization=factorization)
    conv_dense = SpectralConv3d(3, 6, (4, 4, 3), n_layers=1, bias=False,
        implementation='reconstructed', factorization=None)
    for i, w in enumerate(conv.weight):
        rec = w.to_tensor()
        dtype = rec.dtype
        assert dtype == 'complex64'
        conv_dense.weight[i] = FactorizedTensor.from_tensor(rec, rank=None,
            factorization='ComplexDense')
    x = paddle.randn(shape=[2, 3, 12, 12, 12])
    res_dense = conv_dense(x)
    res = conv(x)
    assert paddle.allclose(x=res_dense, y=res).item(), ''


@pytest.mark.parametrize('factorization', ['ComplexCP', 'ComplexTucker',
    'ComplexDense'])
@pytest.mark.parametrize('implementation', ['factorized', 'reconstructed'])
def test_SpectralConv2D(factorization, implementation):
    """Compare generic SpectralConv with hand written SpectralConv2D
    
    Verifies that a dense conv and factorized conv with the same weight produce the same output
    Note that this implies the order in which the conv is done in the manual implementation matches the automatic one, 
    take with a grain of salt
    """
    conv = SpectralConv(10, 11, (4, 5), n_layers=1, bias=False,
        implementation=implementation, factorization=factorization)
    conv_dense = SpectralConv2d(10, 11, (4, 5), n_layers=1, bias=False,
        implementation='reconstructed', factorization=None)
    for i, w in enumerate(conv.weight):
        rec = w.to_tensor()
        dtype = rec.dtype
        assert dtype == 'complex64'
        conv_dense.weight[i] = FactorizedTensor.from_tensor(rec, rank=None,
            factorization='ComplexDense')
    x = paddle.randn(shape=[2, 10, 12, 12])
    res_dense = conv_dense(x)
    res = conv(x)
    assert paddle.allclose(x=res_dense, y=res).item(), ''


@pytest.mark.parametrize('factorization', ['ComplexCP', 'ComplexTucker'])
@pytest.mark.parametrize('implementation', ['factorized', 'reconstructed'])
def test_SpectralConv1D(factorization, implementation):
    """Test for SpectralConv1D
    
    Verifies that a dense conv and factorized conv with the same weight produce the same output
    """
    conv = SpectralConv(10, 11, (5,), n_layers=1, bias=False,
        implementation=implementation, factorization=factorization)
    conv_dense = SpectralConv1d(10, 11, (5,), n_layers=1, bias=False,
        implementation='reconstructed', factorization=None)
    for i, w in enumerate(conv.weight):
        rec = w.to_tensor()
        dtype = rec.dtype
        assert dtype == 'complex64'
        conv_dense.weight[i] = FactorizedTensor.from_tensor(rec, rank=None,
            factorization='ComplexDense')
    x = paddle.randn(shape=[2, 10, 12])
    res_dense = conv_dense(x)
    res = conv(x)
    assert paddle.allclose(x=res_dense, y=res).item(), ''
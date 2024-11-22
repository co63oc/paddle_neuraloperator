import paddle
from ..segment_csr import segment_csr
import pytest


@pytest.mark.parametrize('batch_size', [1, 4])
def test_native_segcsr_shapes(batch_size):
    n_pts = 25
    n_channels = 5
    max_nbrhd_size = 7
    src = paddle.randn(shape=(batch_size, n_pts, n_channels))
    nbrhd_sizes = [paddle.to_tensor(data=[0])]
    while sum(nbrhd_sizes) < n_pts:
        nbrhd_sizes.append(paddle.randint(low=0, high=max_nbrhd_size + 1,
            shape=(1,)))
        max_nbrhd_size = min(max_nbrhd_size, n_pts - sum(nbrhd_sizes))
    indptr = paddle.cumsum(x=paddle.to_tensor(data=nbrhd_sizes, dtype=
        'int64'), axis=0)
    if batch_size > 1:
        indptr = indptr.tile(repeat_times=[batch_size] + [1] * indptr.ndim)
    else:
        src = src.squeeze(axis=0)
    out = segment_csr(src=src, indptr=indptr, reduce='sum', use_scatter=False)
    if batch_size == 1:
        assert tuple(out.shape) == (len(indptr) - 1, n_channels)
    else:
        assert tuple(out.shape) == (batch_size, tuple(indptr.shape)[1] - 1,
            n_channels)


def test_native_segcsr_reductions():
    src = paddle.ones(shape=[10, 3])
    indptr = paddle.to_tensor(data=[0, 3, 8, 10], dtype='int64')
    out_sum = segment_csr(src, indptr, reduce='sum', use_scatter=False)
    assert tuple(out_sum.shape) == (3, 3)
    diff = out_sum - paddle.to_tensor(data=[[3, 5, 2]]).T * paddle.ones(shape
        =[3, 3])
    assert not diff.nonzero().astype('bool').any()
    out_mean = segment_csr(src, indptr, reduce='mean', use_scatter=False)
    assert tuple(out_mean.shape) == (3, 3)
    diff = out_mean - paddle.ones(shape=[3, 3])
    assert not diff.nonzero().astype('bool').any()

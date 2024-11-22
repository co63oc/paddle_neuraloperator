import paddle
import math
from ..data_losses import LpLoss, H1Loss, MSELoss
from ..finite_diff import central_diff_1d, central_diff_2d, central_diff_3d
from neuralop.layers.embeddings import regular_grid_nd


def test_lploss():
    l2_2d_mean = LpLoss(d=2, p=2, reductions='mean')
    l2_2d_sum = LpLoss(d=2, p=2, reductions='sum')
    x = paddle.randn(shape=[10, 4, 4])
    abs_0 = l2_2d_mean.abs(x, x)
    assert abs_0.item() == 0.0
    zeros = paddle.zeros_like(x=x)
    ones = paddle.ones_like(x=x)
    mean_abs_l2_err = l2_2d_mean.abs(zeros, ones, h=1.0)
    assert mean_abs_l2_err.item() == 4.0
    sum_abs_l2_err = l2_2d_sum.abs(zeros, ones, h=1.0)
    assert sum_abs_l2_err.item() == 40.0
    eps = 1e-07
    mean_abs_l2_err = l2_2d_mean.abs(zeros, ones)
    assert mean_abs_l2_err.item() - 4.0 * math.pi / 2 <= eps
    sum_abs_l2_err = l2_2d_sum.abs(zeros, ones)
    assert sum_abs_l2_err.item() - 40.0 * math.pi / 2 <= eps


def test_h1loss():
    h1 = H1Loss(d=2, reductions='mean')
    x = paddle.randn(shape=[10, 4, 4])
    abs_0 = h1.abs(x, x)
    assert abs_0.item() == 0.0
    zeros = paddle.zeros_like(x=x)
    ones = paddle.ones_like(x=x)
    mean_abs_h1 = h1.abs(zeros, ones, h=1.0)
    assert mean_abs_h1.item() == 4.0


def test_mseloss():
    mse_2d = MSELoss(reductions='sum')
    x = paddle.randn(shape=[10, 4, 4])
    abs_0 = mse_2d(x, x)
    assert abs_0.item() == 0.0
    zeros = paddle.zeros_like(x=x)
    ones = paddle.ones_like(x=x)
    mean_abs_mse = mse_2d(zeros, ones)
    assert mean_abs_mse.item() == 10.0


def test_central_diff1d():
    x = paddle.arange(end=10)
    dx = central_diff_1d(x, h=1.0, fix_x_bnd=True)
    assert paddle.allclose(x=dx, y=paddle.ones_like(x=dx)).item(), ''


def test_central_diff2d():
    grid = regular_grid_nd(resolutions=[10, 10], grid_boundaries=[[0, 10]] * 2)
    x = paddle.stack(x=grid, axis=0)
    dx, dy = central_diff_2d(x, h=1.0, fix_x_bnd=True, fix_y_bnd=True)
    assert paddle.allclose(x=dx[0], y=paddle.ones_like(x=dx[0])).item(), ''
    assert paddle.allclose(x=dx[1], y=paddle.zeros_like(x=dx[1])).item(), ''
    assert paddle.allclose(x=dy[0], y=paddle.zeros_like(x=dy[0])).item(), ''
    assert paddle.allclose(x=dy[1], y=paddle.ones_like(x=dy[1])).item(), ''


def test_central_diff3d():
    grid = regular_grid_nd(resolutions=[10, 10, 10], grid_boundaries=[[0, 
        10]] * 3)
    x = paddle.stack(x=grid, axis=0)
    dx, dy, dz = central_diff_3d(x, h=1.0, fix_x_bnd=True, fix_y_bnd=True,
        fix_z_bnd=True)
    assert paddle.allclose(x=dx[0], y=paddle.ones_like(x=dx[0])).item(), ''
    assert paddle.allclose(x=dx[1], y=paddle.zeros_like(x=dx[1])).item(), ''
    assert paddle.allclose(x=dx[2], y=paddle.zeros_like(x=dx[1])).item(), ''
    assert paddle.allclose(x=dy[0], y=paddle.zeros_like(x=dy[0])).item(), ''
    assert paddle.allclose(x=dy[1], y=paddle.ones_like(x=dy[1])).item(), ''
    assert paddle.allclose(x=dy[2], y=paddle.zeros_like(x=dy[2])).item(), ''
    assert paddle.allclose(x=dz[0], y=paddle.zeros_like(x=dz[0])).item(), ''
    assert paddle.allclose(x=dz[1], y=paddle.zeros_like(x=dz[1])).item(), ''
    assert paddle.allclose(x=dz[2], y=paddle.ones_like(x=dz[2])).item(), ''

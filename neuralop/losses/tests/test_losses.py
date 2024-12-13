import math

import paddle

from neuralop.layers.embeddings import regular_grid_nd

from ..data_losses import H1Loss
from ..data_losses import LpLoss
from ..data_losses import MSELoss
from ..finite_diff import central_diff_1d
from ..finite_diff import central_diff_2d
from ..finite_diff import central_diff_3d


def test_lploss():
    l2_2d_mean = LpLoss(d=2, p=2, reductions="mean")
    l2_2d_sum = LpLoss(d=2, p=2, reductions="sum")
    x = paddle.randn(shape=[10, 4, 4])
    abs_0 = l2_2d_mean.abs(x, x)
    assert abs_0.item() == 0.0
    zeros = paddle.zeros_like(x=x)
    ones = paddle.ones_like(x=x)
    # L2 w/out normalizing constant
    # sum of items in each element in ones is 16
    # norm is 4
    mean_abs_l2_err = l2_2d_mean.abs(zeros, ones, h=1.0)
    assert mean_abs_l2_err.item() == 4.0
    sum_abs_l2_err = l2_2d_sum.abs(zeros, ones, h=1.0)
    assert sum_abs_l2_err.item() == 40.0
    eps = 5e-06
    # L2 with default 1d normalizing constant
    # result should be scaled by 2pi/(geometric mean of input dims= 4)
    mean_abs_l2_err = l2_2d_mean.abs(zeros, ones)
    assert mean_abs_l2_err.item() - 4.0 * math.pi / 2 <= eps
    sum_abs_l2_err = l2_2d_sum.abs(zeros, ones)
    assert sum_abs_l2_err.item() - 40.0 * math.pi / 2 <= eps


def test_h1loss():
    h1 = H1Loss(d=2, reductions="mean")
    x = paddle.randn(shape=[10, 4, 4])
    abs_0 = h1.abs(x, x)
    assert abs_0.item() == 0.0
    zeros = paddle.zeros_like(x=x)
    ones = paddle.ones_like(x=x)
    # H1 w/out normalizing constant,
    # finite-difference derivatives of both sides are zero
    # sum of items in each element in ones is 16
    # norm is 4
    mean_abs_h1 = h1.abs(zeros, ones, h=1.0)
    assert mean_abs_h1.item() == 4.0


def test_mseloss():
    mse_2d = MSELoss(reductions="sum")
    x = paddle.randn(shape=[10, 4, 4])
    abs_0 = mse_2d(x, x)
    assert abs_0.item() == 0.0
    zeros = paddle.zeros_like(x=x)
    ones = paddle.ones_like(x=x)

    # all elem-wise differences are 1., squared and averaged = 1.
    # reduced by sum across batch = 10 * 1. = 10.
    mean_abs_mse = mse_2d(zeros, ones)
    assert mean_abs_mse.item() == 10.0


def test_central_diff1d():
    # assert f(x) = x
    # has derivative 1 everywhere when boundaries are fixed
    x = paddle.arange(end=10)
    dx = central_diff_1d(x, h=1.0, fix_x_bnd=True)
    assert paddle.allclose(x=dx, y=paddle.ones_like(x=dx)).item(), ""


def test_central_diff2d():
    grid = regular_grid_nd(resolutions=[10, 10], grid_boundaries=[[0, 10]] * 2)
    x = paddle.stack(x=grid, axis=0)
    dx, dy = central_diff_2d(x, h=1.0, fix_x_bnd=True, fix_y_bnd=True)
    # pos encoding A[:,i,j] = [xi, yj]

    # dx[:,i,j] = f(x_i, y_j) vector valued <fx, fy>
    # dfx(coords) == 1s

    assert paddle.allclose(x=dx[0], y=paddle.ones_like(x=dx[0])).item(), ""
    assert paddle.allclose(x=dx[1], y=paddle.zeros_like(x=dx[1])).item(), ""
    assert paddle.allclose(x=dy[0], y=paddle.zeros_like(x=dy[0])).item(), ""
    assert paddle.allclose(x=dy[1], y=paddle.ones_like(x=dy[1])).item(), ""


def test_central_diff3d():
    grid = regular_grid_nd(resolutions=[10, 10, 10], grid_boundaries=[[0, 10]] * 3)
    x = paddle.stack(x=grid, axis=0)
    # pos encoding A[:,i,j,k] = [xi, yj, zk]
    dx, dy, dz = central_diff_3d(x, h=1.0, fix_x_bnd=True, fix_y_bnd=True, fix_z_bnd=True)
    # dx[:,i,j,k] = f(x_i, y_j, z_k) vector valued <fx, fy, fz>
    # dfx(coords) == 1s

    assert paddle.allclose(x=dx[0], y=paddle.ones_like(x=dx[0])).item(), ""
    assert paddle.allclose(x=dx[1], y=paddle.zeros_like(x=dx[1])).item(), ""
    assert paddle.allclose(x=dx[2], y=paddle.zeros_like(x=dx[1])).item(), ""
    assert paddle.allclose(x=dy[0], y=paddle.zeros_like(x=dy[0])).item(), ""
    assert paddle.allclose(x=dy[1], y=paddle.ones_like(x=dy[1])).item(), ""
    assert paddle.allclose(x=dy[2], y=paddle.zeros_like(x=dy[2])).item(), ""
    assert paddle.allclose(x=dz[0], y=paddle.zeros_like(x=dz[0])).item(), ""
    assert paddle.allclose(x=dz[1], y=paddle.zeros_like(x=dz[1])).item(), ""
    assert paddle.allclose(x=dz[2], y=paddle.ones_like(x=dz[2])).item(), ""

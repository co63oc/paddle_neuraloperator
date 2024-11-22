import sys
sys.path.append('/nfs/github/paddle/paddle_neuraloperator/utils')
import paddle_aux
import paddle
import pytest
import numpy as np
import math
from ..differential_conv import FiniteDifferenceConvolution


def get_grid(S, batchsize, device):
    gridx = paddle.to_tensor(data=np.linspace(0, 1, S), dtype='float32')
    gridx = gridx.reshape(1, 1, S, 1).tile(repeat_times=[batchsize, 1, 1, S])
    gridy = paddle.to_tensor(data=np.linspace(0, 1, S), dtype='float32')
    gridy = gridy.reshape(1, 1, 1, S).tile(repeat_times=[batchsize, 1, S, 1])
    return paddle.concat(x=(gridx, gridy), axis=1).to(device)


@pytest.mark.parametrize('resolution', [500, 700, 1000])
def test_convergence_FiniteDifferenceConvolution_subtract_middle(resolution):
    paddle.seed(seed=0)
    device = paddle.CPUPlace()
    num_channels = 10
    kernel_size = 3
    coeff = paddle.rand(shape=(num_channels,))
    differential_block = FiniteDifferenceConvolution(in_channels=
        num_channels, out_channels=1, num_dim=2, kernel_size=kernel_size,
        groups=1, implementation='subtract_middle').to(device)
    with paddle.no_grad():
        weight = differential_block.conv.weight[0]
    diff_block_output_list = []
    grid_width = 1 / resolution
    grid = get_grid(resolution, 1, device)
    channels = [paddle.sum(x=coeff[i] * paddle.square(x=grid), axis=1) for
        i in range(num_channels)]
    parabola = paddle.stack(x=channels, axis=1).to(device)
    diff_block_output = differential_block(parabola, grid_width)
    diff_block_output_list.append(diff_block_output)
    theoretical_value = 0
    for k in range(num_channels):
        direction_k = 0
        for i in range(kernel_size):
            for j in range(kernel_size):
                direction_k += weight[k, i, j] * paddle.to_tensor(data=[[[[
                    i - kernel_size // 2, j - kernel_size // 2]]]]).to(device)
        direction_k = direction_k.moveaxis(source=-1, destination=1).tile(
            repeat_times=[1, 1, resolution, resolution])
        theoretical_value += 2 * coeff[k] * paddle.sum(x=grid * direction_k,
            axis=1)
    error = 1 / (resolution - 2) * paddle.linalg.norm(x=diff_block_output.
        squeeze()[1:-1, 1:-1] - theoretical_value.squeeze()[1:-1, 1:-1]).item()
    assert math.isclose(0, error, abs_tol=0.1)


@pytest.mark.parametrize('resolution', [500, 700, 1000])
def test_convergence_FiniteDifferenceConvolution_subtract_all(resolution):

    def get_grid(S, batchsize, device):
        gridx = paddle.to_tensor(data=np.linspace(0, 1, S), dtype='float32')
        gridx = gridx.reshape(1, 1, S, 1).tile(repeat_times=[batchsize, 1, 
            1, S])
        gridy = paddle.to_tensor(data=np.linspace(0, 1, S), dtype='float32')
        gridy = gridy.reshape(1, 1, 1, S).tile(repeat_times=[batchsize, 1,
            S, 1])
        return paddle.concat(x=(gridx, gridy), axis=1).to(device)
    paddle.seed(seed=0)
    device = paddle.CPUPlace()
    num_channels = 10
    kernel_size = 3
    coeff = paddle.rand(shape=(num_channels,))
    differential_block = FiniteDifferenceConvolution(in_channels=
        num_channels, out_channels=1, num_dim=2, kernel_size=kernel_size,
        groups=1, implementation='subtract_all').to(device)
    with paddle.no_grad():
        weight = differential_block.conv_kernel[0].detach()
        weight -= paddle.mean(x=weight, axis=(-2, -1), keepdim=True).tile(
            repeat_times=[1, kernel_size, kernel_size])
    diff_block_output_list = []
    grid_width = 1 / resolution
    grid = get_grid(resolution, 1, device)
    channels = [paddle.sum(x=coeff[i] * paddle.square(x=grid), axis=1) for
        i in range(num_channels)]
    parabola = paddle.stack(x=channels, axis=1).to(device)
    diff_block_output = differential_block(parabola, grid_width)
    diff_block_output_list.append(diff_block_output)
    theoretical_value = 0
    for k in range(num_channels):
        direction_k = 0
        for i in range(kernel_size):
            for j in range(kernel_size):
                direction_k += weight[k, i, j] * paddle.to_tensor(data=[[[[
                    i - kernel_size // 2, j - kernel_size // 2]]]]).to(device)
        direction_k = direction_k.moveaxis(source=-1, destination=1).tile(
            repeat_times=[1, 1, resolution, resolution])
        theoretical_value += 2 * coeff[k] * paddle.sum(x=grid * direction_k,
            axis=1)
    error = 1 / (resolution - 2) * paddle.linalg.norm(x=diff_block_output.
        squeeze()[1:-1, 1:-1] - theoretical_value.squeeze()[1:-1, 1:-1]).item()
    assert math.isclose(0, error, abs_tol=0.1)

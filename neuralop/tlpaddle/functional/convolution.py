import paddle
import tensorly as tl
from tensorly import tenalg

import neuralop.paddle_aux  # noqa

from ..factorized_tensors import CPTensor
from ..factorized_tensors import DenseTensor
from ..factorized_tensors import TTTensor
from ..factorized_tensors import TuckerTensor

tl.set_backend("paddle")

_CONVOLUTION = {
    (1): paddle.nn.functional.conv1d,
    (2): paddle.nn.functional.conv2d,
    (3): paddle.nn.functional.conv3d,
}


def convolve(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Convolution of any specified order, wrapper on paddle's F.convNd

    Parameters
    ----------
    x : paddle.Tensor or FactorizedTensor
        input tensor
    weight : paddle.Tensor
        convolutional weights
    bias : bool, optional
        by default None
    stride : int, optional
        by default 1
    padding : int, optional
        by default 0
    dilation : int, optional
        by default 1
    groups : int, optional
        by default 1

    Returns
    -------
    paddle.Tensor
        `x` convolved with `weight`
    """
    try:
        if paddle.is_tensor(x=weight):
            return _CONVOLUTION[weight.ndim - 2](
                x,
                weight,
                bias=bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
        else:
            if isinstance(weight, TTTensor):
                weight = tl.moveaxis(weight.to_tensor(), -1, 0)
            else:
                weight = weight.to_tensor()
            return _CONVOLUTION[weight.ndim - 2](
                x,
                weight,
                bias=bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
    except KeyError:
        raise ValueError(
            f"Got tensor of order={weight.ndim} but paddle only supports up to 3rd order (3D) Convs."
        )


def general_conv1d_(
    x, kernel, mode, bias=None, stride=1, padding=0, groups=1, dilation=1, verbose=False
):
    """General 1D convolution along the mode-th dimension

    Parameters
    ----------
    x : batch-dize, in_channels, K1, ..., KN
    kernel : out_channels, in_channels/groups, K{mode}
    mode : int
        weight along which to perform the decomposition
    stride : int
    padding : int
    groups : 1
        typically would be equal to thhe number of input-channels
        at least for CP convolutions

    Returns
    -------
    x convolved with the given kernel, along dimension `mode`
    """
    if verbose:
        print(
            f"Convolving {tuple(x.shape)} with {tuple(kernel.shape)} along mode {mode}, stride={stride}, padding={padding}, groups={groups}"
        )
    in_channels = tl.shape(x)[1]
    n_dim = tl.ndim(x)
    permutation = list(range(n_dim))
    spatial_dim = permutation.pop(mode)
    channels_dim = permutation.pop(1)
    permutation += [channels_dim, spatial_dim]
    x = tl.transpose(x, permutation)
    x_shape = list(tuple(x.shape))
    x = tl.reshape(x, (-1, in_channels, x_shape[-1]))
    x = paddle.nn.functional.conv1d(
        x=x.contiguous(),
        weight=kernel,
        bias=bias,
        stride=stride,
        dilation=dilation,
        padding=padding,
        groups=groups,
    )
    x_shape[-2:] = tuple(x.shape)[-2:]
    x = tl.reshape(x, x_shape)
    permutation = list(range(n_dim))[:-2]
    permutation.insert(1, n_dim - 2)
    permutation.insert(mode, n_dim - 1)
    x = tl.transpose(x, permutation)
    return x


def general_conv1d(
    x, kernel, mode, bias=None, stride=1, padding=0, groups=1, dilation=1, verbose=False
):
    """General 1D convolution along the mode-th dimension

    Uses an ND convolution under the hood

    Parameters
    ----------
    x : batch-dize, in_channels, K1, ..., KN
    kernel : out_channels, in_channels/groups, K{mode}
    mode : int
        weight along which to perform the decomposition
    stride : int
    padding : int
    groups : 1
        typically would be equal to the number of input-channels
        at least for CP convolutions

    Returns
    -------
    x convolved with the given kernel, along dimension `mode`
    """
    if verbose:
        print(
            f"Convolving {tuple(x.shape)} with {tuple(kernel.shape)} along mode {mode}, stride={stride}, padding={padding}, groups={groups}"
        )

    def _pad_value(value, mode, order, padding=1):
        return tuple([(value if i == mode - 2 else padding) for i in range(order)])

    ndim = tl.ndim(x)
    order = ndim - 2
    for i in range(2, ndim):
        if i != mode:
            kernel = kernel.unsqueeze(axis=i)
    return _CONVOLUTION[order](
        x,
        kernel,
        bias=bias,
        stride=_pad_value(stride, mode, order),
        padding=_pad_value(padding, mode, order, padding=0),
        dilation=_pad_value(dilation, mode, order),
        groups=groups,
    )


def tucker_conv(x, tucker_tensor, bias=None, stride=1, padding=0, dilation=1):
    rank = tucker_tensor.rank
    batch_size = tuple(x.shape)[0]
    n_dim = tl.ndim(x)
    x_shape = list(tuple(x.shape))
    x = x.reshape((batch_size, x_shape[1], -1)).contiguous()
    x = paddle.nn.functional.conv1d(x=x, weight=tl.transpose(tucker_tensor.factors[1]).unsqueeze(2))
    x_shape[1] = rank[1]
    x = x.reshape(x_shape)
    modes = list(range(2, n_dim + 1))
    weight = tl.tenalg.multi_mode_dot(tucker_tensor.core, tucker_tensor.factors[2:], modes=modes)
    x = convolve(x, weight, bias=None, stride=stride, padding=padding, dilation=dilation)
    x_shape = list(tuple(x.shape))
    x = x.reshape((batch_size, x_shape[1], -1))
    x = paddle.nn.functional.conv1d(
        x=x, weight=tucker_tensor.factors[0].unsqueeze(axis=2), bias=bias
    )
    x_shape[1] = tuple(x.shape)[1]
    x = x.reshape(x_shape)
    return x


def tt_conv(x, tt_tensor, bias=None, stride=1, padding=0, dilation=1):
    """Perform a factorized tt convolution

    Parameters
    ----------
    x : paddle.Tensor
        tensor of shape (batch_size, C, I_2, I_3, ..., I_N)

    Returns
    -------
    NDConv(x) with an tt kernel
    """
    shape = tuple(tt_tensor.shape)
    # rank = tt_tensor.rank
    batch_size = tuple(x.shape)[0]
    order = len(shape) - 2
    if isinstance(padding, int):
        padding = (padding,) * order
    if isinstance(stride, int):
        stride = (stride,) * order
    if isinstance(dilation, int):
        dilation = (dilation,) * order
    x_shape = list(tuple(x.shape))
    x = x.reshape((batch_size, x_shape[1], -1)).contiguous()
    x = paddle.nn.functional.conv1d(x=x, weight=tl.transpose(tt_tensor.factors[0], [2, 1, 0]))
    x_shape[1] = tuple(x.shape)[1]
    x = x.reshape(x_shape)
    for i in range(order):
        kernel = tl.transpose(tt_tensor.factors[i + 1], [2, 0, 1])
        x = general_conv1d(
            x.contiguous(),
            kernel,
            i + 2,
            stride=stride[i],
            padding=padding[i],
            dilation=dilation[i],
        )
    x_shape = list(tuple(x.shape))
    x = x.reshape((batch_size, x_shape[1], -1))
    x = paddle.nn.functional.conv1d(
        x=x, weight=tl.transpose(tt_tensor.factors[-1], [1, 0, 2]), bias=bias
    )
    x_shape[1] = tuple(x.shape)[1]
    x = x.reshape(x_shape)
    return x


def cp_conv(x, cp_tensor, bias=None, stride=1, padding=0, dilation=1):
    """Perform a factorized CP convolution

    Parameters
    ----------
    x : paddle.Tensor
        tensor of shape (batch_size, C, I_2, I_3, ..., I_N)

    Returns
    -------
    NDConv(x) with an CP kernel
    """
    shape = tuple(cp_tensor.shape)
    rank = cp_tensor.rank
    batch_size = tuple(x.shape)[0]
    order = len(shape) - 2
    if isinstance(padding, int):
        padding = (padding,) * order
    if isinstance(stride, int):
        stride = (stride,) * order
    if isinstance(dilation, int):
        dilation = (dilation,) * order
    x_shape = list(tuple(x.shape))
    x = x.reshape((batch_size, x_shape[1], -1)).contiguous()
    x = paddle.nn.functional.conv1d(x=x, weight=tl.transpose(cp_tensor.factors[1]).unsqueeze(2))
    x_shape[1] = rank
    x = x.reshape(x_shape)
    for i in range(order):
        kernel = tl.transpose(cp_tensor.factors[i + 2]).unsqueeze(1)
        x = general_conv1d(
            x.contiguous(),
            kernel,
            i + 2,
            stride=stride[i],
            padding=padding[i],
            dilation=dilation[i],
            groups=rank,
        )
    x_shape = list(tuple(x.shape))
    x = x.reshape((batch_size, x_shape[1], -1))
    x = paddle.nn.functional.conv1d(
        x=x * cp_tensor.weights.unsqueeze(1).unsqueeze(axis=0),
        weight=cp_tensor.factors[0].unsqueeze(axis=2),
        bias=bias,
    )
    x_shape[1] = tuple(x.shape)[1]
    x = x.reshape(x_shape)
    return x


def cp_conv_mobilenet(x, cp_tensor, bias=None, stride=1, padding=0, dilation=1):
    """Perform a factorized CP convolution

    Parameters
    ----------
    x : paddle.Tensor
        tensor of shape (batch_size, C, I_2, I_3, ..., I_N)

    Returns
    -------
    NDConv(x) with an CP kernel
    """
    factors = cp_tensor.factors
    shape = tuple(cp_tensor.shape)
    rank = cp_tensor.rank
    batch_size = tuple(x.shape)[0]
    order = len(shape) - 2
    x_shape = list(tuple(x.shape))
    x = x.reshape((batch_size, x_shape[1], -1)).contiguous()
    x = paddle.nn.functional.conv1d(x=x, weight=tl.transpose(factors[1]).unsqueeze(2))
    x_shape[1] = rank
    x = x.reshape(x_shape)
    if order == 1:
        weight = tl.transpose(factors[2]).unsqueeze(1)
        x = paddle.nn.functional.conv1d(
            x=x.contiguous(),
            weight=weight,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=rank,
        )
    elif order == 2:
        weight = tenalg.tensordot(
            tl.transpose(factors[2]),
            tl.transpose(factors[3]),
            modes=(),
            batched_modes=0,
        ).unsqueeze(1)
        x = paddle.nn.functional.conv2d(
            x=x.contiguous(),
            weight=weight,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=rank,
        )
    elif order == 3:
        weight = tenalg.tensordot(
            tl.transpose(factors[2]),
            tenalg.tensordot(
                tl.transpose(factors[3]),
                tl.transpose(factors[4]),
                modes=(),
                batched_modes=0,
            ),
            modes=(),
            batched_modes=0,
        ).unsqueeze(1)
        x = paddle.nn.functional.conv3d(
            x=x.contiguous(),
            weight=weight,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=rank,
        )
    x_shape = list(tuple(x.shape))
    x = x.reshape((batch_size, x_shape[1], -1))
    x = paddle.nn.functional.conv1d(
        x=x * cp_tensor.weights.unsqueeze(1).unsqueeze(axis=0),
        weight=factors[0].unsqueeze(axis=2),
        bias=bias,
    )
    x_shape[1] = tuple(x.shape)[1]
    x = x.reshape(x_shape)
    return x


def _get_factorized_conv(factorization, implementation="factorized"):
    if implementation == "reconstructed" or factorization == "Dense":
        return convolve
    if isinstance(factorization, CPTensor):
        if implementation == "factorized":
            return cp_conv
        elif implementation == "mobilenet":
            return cp_conv_mobilenet
    elif isinstance(factorization, TuckerTensor):
        return tucker_conv
    elif isinstance(factorization, TTTensor):
        return tt_conv
    raise ValueError(f"Got unknown type {factorization}")


def convNd(x, weight, bias=None, stride=1, padding=0, dilation=1, implementation="factorized"):
    if implementation == "reconstructed":
        weight = weight.to_tensor()
    if isinstance(weight, DenseTensor):
        return convolve(
            x,
            weight.tensor,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
    if paddle.is_tensor(x=weight):
        return convolve(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation)
    if isinstance(weight, CPTensor):
        if implementation == "factorized":
            return cp_conv(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation)
        elif implementation == "mobilenet":
            return cp_conv_mobilenet(
                x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation
            )
    elif isinstance(weight, TuckerTensor):
        return tucker_conv(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation)
    elif isinstance(weight, TTTensor):
        return tt_conv(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation)

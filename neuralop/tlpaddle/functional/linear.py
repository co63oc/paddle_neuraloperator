import numpy as np
import paddle
import tensorly as tl

import neuralop.paddle_aux  # noqa

from ..factorized_tensors import TensorizedTensor
from ..factorized_tensors.tensorized_matrices import BlockTT
from ..factorized_tensors.tensorized_matrices import CPTensorized
from ..factorized_tensors.tensorized_matrices import TuckerTensorized
from .factorized_linear import linear_blocktt
from .factorized_linear import linear_cp
from .factorized_linear import linear_tucker

tl.set_backend("numpy")


def factorized_linear(x, weight, bias=None, in_features=None, implementation="factorized"):
    """Linear layer with a dense input x and factorized weight"""
    assert implementation in {
        "factorized",
        "reconstructed",
    }, f"Expect implementation from [factorized, reconstructed], but got {implementation}"
    if in_features is None:
        in_features = np.prod(tuple(x.shape)[-1])
    if not paddle.is_tensor(x=weight):
        if isinstance(weight, TensorizedTensor):
            if implementation == "factorized":
                x_shape = tuple(x.shape)[:-1] + weight.tensorized_shape[1]
                out_shape = tuple(x.shape)[:-1] + (-1,)
                if isinstance(weight, CPTensorized):
                    x = linear_cp(x.reshape(x_shape), weight).reshape(out_shape)
                    if bias is not None:
                        x = x + bias
                    return x
                elif isinstance(weight, TuckerTensorized):
                    x = linear_tucker(x.reshape(x_shape), weight).reshape(out_shape)
                    if bias is not None:
                        x = x + bias
                    return x
                elif isinstance(weight, BlockTT):
                    x = linear_blocktt(x.reshape(x_shape), weight).reshape(out_shape)
                    if bias is not None:
                        x = x + bias
                    return x
            weight = weight.to_matrix()
        else:
            weight = weight.to_tensor()
    return paddle.nn.functional.linear(
        x=x, weight=paddle.reshape(x=weight, shape=(-1, in_features)).T, bias=bias
    )

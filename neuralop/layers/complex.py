import paddle
"""
Functionality for handling complex-valued spatial data
"""
from copy import deepcopy


def CGELU(x: paddle.Tensor):
    """Complex GELU activation function
    Follows the formulation of CReLU from Deep Complex Networks (https://openreview.net/pdf?id=H1T2hmZAb)
    apply GELU is real and imag part of the input separately, then combine as complex number
    Args:
        x: complex tensor
    """
    return paddle.nn.functional.gelu(x=x.real()).astype('complex64'
        ) + 1.0j * paddle.nn.functional.gelu(x=x.imag()).astype('complex64')


def ctanh(x: paddle.Tensor):
    """Complex-valued tanh stabilizer
    Apply ctanh is real and imag part of the input separately, then combine as complex number
    Args:
        x: complex tensor
    """
    return paddle.nn.functional.tanh(x=x.real()).astype('complex64'
        ) + 1.0j * paddle.nn.functional.tanh(x=x.imag()).astype('complex64')


def apply_complex(real_func, imag_func, x, dtype='complex64'):
    """
    fr: a function (e.g., conv) to be applied on real part of x
    fi: a function (e.g., conv) to be applied on imag part of x
    x: complex input.
    """
    return (real_func(x.real()) - imag_func(x.imag())).astype(dtype) + 1.0j * (
        real_func(x.imag()) + imag_func(x.real())).astype(dtype)


class ComplexValued(paddle.nn.Layer):
    """
    Wrapper class that converts a standard nn.Module that operates on real data
    into a module that operates on complex-valued spatial data.
    """

    def __init__(self, module):
        super(ComplexValued, self).__init__()
        self.fr = deepcopy(module)
        self.fi = deepcopy(module)

    def forward(self, x):
        return apply_complex(self.fr, self.fi, x)

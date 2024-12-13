import paddle
import tensorly as tl

import neuralop
from neuralop import paddle_aux

from .factorized_tensors import CPTensor
from .factorized_tensors import DenseTensor
from .factorized_tensors import TTTensor
from .factorized_tensors import TuckerTensor

tl.set_backend("numpy")


class ComplexHandler:
    def __setattr__(self, key, value):
        if isinstance(value, neuralop.tlpaddle.utils.parameter_list.FactorList):
            value = neuralop.tlpaddle.utils.parameter_list.ComplexFactorList(value)
            super().__setattr__(key, value)
        elif isinstance(value, paddle.base.framework.EagerParamBase):
            self.add_parameter(name=key, parameter=value)
        elif paddle.is_tensor(x=value):
            self.register_buffer(name=key, tensor=value)
        else:
            super().__setattr__(key, value)

    def __getattr__(self, key):
        value = super().__getattr__(key)
        if paddle.is_tensor(x=value):
            value = paddle_aux.as_complex(x=value)
        return value

    def register_parameter(self, key, value):
        value = paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.as_real(x=value))
        super().add_parameter(name=key, parameter=value)

    def register_buffer(self, key, value):
        value = paddle.as_real(x=value)
        super().register_buffer(name=key, tensor=value)


class ComplexDenseTensor(ComplexHandler, DenseTensor, name="ComplexDense"):
    """Complex Dense Factorization"""

    @classmethod
    def new(cls, shape, rank=None, device=None, dtype="complex64", **kwargs):
        return super().new(shape, rank, device=device, dtype=dtype, **kwargs)


class ComplexTuckerTensor(ComplexHandler, TuckerTensor, name="ComplexTucker"):
    """Complex Tucker Factorization"""

    @classmethod
    def new(
        cls, shape, rank="same", fixed_rank_modes=None, device=None, dtype="complex64", **kwargs
    ):
        return super().new(
            shape, rank, fixed_rank_modes=fixed_rank_modes, device=device, dtype=dtype, **kwargs
        )


class ComplexTTTensor(ComplexHandler, TTTensor, name="ComplexTT"):
    """Complex TT Factorization"""

    @classmethod
    def new(
        cls, shape, rank="same", fixed_rank_modes=None, device=None, dtype="complex64", **kwargs
    ):
        return super().new(shape, rank, device=device, dtype=dtype, **kwargs)


class ComplexCPTensor(ComplexHandler, CPTensor, name="ComplexCP"):
    """Complex CP Factorization"""

    @classmethod
    def new(
        cls, shape, rank="same", fixed_rank_modes=None, device=None, dtype="complex64", **kwargs
    ):
        return super().new(shape, rank, device=device, dtype=dtype, **kwargs)

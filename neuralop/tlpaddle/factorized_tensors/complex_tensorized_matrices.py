import tensorly as tl

from .complex_factorized_tensors import ComplexHandler
from .tensorized_matrices import BlockTT
from .tensorized_matrices import CPTensorized
from .tensorized_matrices import DenseTensorized
from .tensorized_matrices import TuckerTensorized

tl.set_backend("numpy")


class ComplexDenseTensorized(ComplexHandler, DenseTensorized, name="ComplexDense"):
    """Complex DenseTensorized Factorization"""

    _complex_params = ["tensor"]

    @classmethod
    def new(cls, tensorized_shape, rank=None, device=None, dtype="complex64", **kwargs):
        return super().new(tensorized_shape, rank, device=device, dtype=dtype, **kwargs)


class ComplexTuckerTensorized(ComplexHandler, TuckerTensorized, name="ComplexTucker"):
    """Complex TuckerTensorized Factorization"""

    _complex_params = ["core", "factors"]

    @classmethod
    def new(cls, tensorized_shape, rank=None, device=None, dtype="complex64", **kwargs):
        return super().new(tensorized_shape, rank, device=device, dtype=dtype, **kwargs)


class ComplexBlockTT(ComplexHandler, BlockTT, name="ComplexTT"):
    """Complex BlockTT Factorization"""

    _complex_params = ["factors"]

    @classmethod
    def new(cls, tensorized_shape, rank=None, device=None, dtype="complex64", **kwargs):
        return super().new(tensorized_shape, rank, device=device, dtype=dtype, **kwargs)


class ComplexCPTensorized(ComplexHandler, CPTensorized, name="ComplexCP"):
    """Complex Tensorized CP Factorization"""

    _complex_params = ["weights", "factors"]

    @classmethod
    def new(cls, tensorized_shape, rank=None, device=None, dtype="complex64", **kwargs):
        return super().new(tensorized_shape, rank, device=device, dtype=dtype, **kwargs)

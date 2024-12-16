import math

import numpy as np
import paddle
import tensorly as tl
from tensorly import tenalg
from tensorly.decomposition import parafac
from tensorly.decomposition import tensor_train
from tensorly.decomposition import tucker

from neuralop import paddle_aux

from ..utils import FactorList
from .core import FactorizedTensor

tl.set_backend("paddle")


class DenseTensor(FactorizedTensor, name="Dense"):
    """Dense tensor"""

    def __init__(self, tensor, shape=None, rank=None):
        super().__init__()
        if shape is not None and rank is not None:
            self.shape, self.rank = shape, rank
        else:
            self.shape = tuple(tensor.shape)
            self.rank = None
        self.order = len(self.shape)
        if isinstance(tensor, paddle.base.framework.EagerParamBase):
            self.add_parameter(name="tensor", parameter=tensor)
        else:
            self.register_buffer("tensor", tensor)

    @classmethod
    def new(cls, shape, rank=None, device=None, dtype=None, **kwargs):
        tensor = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.empty(shape=shape, dtype=dtype)
        )
        return cls(tensor)

    @classmethod
    def from_tensor(cls, tensor, rank="same", **kwargs):
        return cls(paddle.base.framework.EagerParamBase.from_tensor(tensor=tl.copy(tensor)))

    def init_from_tensor(self, tensor, l2_reg=1e-05, **kwargs):
        with paddle.no_grad():
            self.tensor = paddle.base.framework.EagerParamBase.from_tensor(tensor=tl.copy(tensor))
        return self

    @property
    def decomposition(self):
        return self.tensor

    def to_tensor(self):
        return self.tensor

    def normal_(self, mean=0, std=1):
        with paddle.no_grad():
            self.tensor.data.normal_(mean=mean, std=std)
        return self

    def __getitem__(self, indices):
        return self.__class__(self.tensor[indices])


def _validate_cp_tensor(cp_tensor):
    """Validates a cp_tensor in the form (weights, factors)

        Returns the rank and shape of the validated tensor

    Parameters
    ----------
    cp_tensor : CPTensor or (weights, factors)

    Returns
    -------
    (shape, rank) : (int tuple, int)
        size of the full tensor and rank of the CP tensor
    """
    if isinstance(cp_tensor, CPTensor):
        # it's already been validated at creation
        return cp_tensor.shape, cp_tensor.rank
    elif isinstance(cp_tensor, (float, int)):  # 0-order tensor
        return 0, 0

    weights, factors = cp_tensor

    if paddle_aux.ndim(factors[0]) == 2:
        rank = int(paddle.shape(factors[0])[1])
    elif paddle_aux.ndim(factors[0]) == 1:
        rank = 1
    else:
        raise ValueError(
            "Got a factor with 3 dimensions but CP factors should be at most 2D, of shape (size, rank)."
        )

    shape = []
    for i, factor in enumerate(factors):
        s = paddle.shape(factor)
        if len(s) == 2:
            current_mode_size, current_rank = s
        else:  # The shape is just (size, ) if rank 1
            current_mode_size, current_rank = *s, 1

        if current_rank != rank:
            raise ValueError(
                "All the factors of a CP tensor should have the same number of column."
                f"However, factors[0].shape[1]={rank} but factors[{i}].shape[1]={paddle.shape(factor)[1]}."
            )
        shape.append(current_mode_size)

    if weights is not None and tuple(paddle.shape(weights)) != (rank,):
        raise ValueError(
            f"Given factors for a rank-{rank} CP tensor but len(weights)={paddle.shape(weights)}."
        )

    return tuple(shape), rank


class CPTensor(FactorizedTensor, name="CP"):
    """CP Factorization

    Parameters
    ----------
    weights
    factors
    shape
    rank
    """

    def __init__(self, weights, factors, shape=None, rank=None):
        super().__init__()
        if shape is not None and rank is not None:
            self.shape, self.rank = shape, rank
        else:
            self.shape, self.rank = _validate_cp_tensor((weights, factors))
        self.order = len(self.shape)
        if isinstance(weights, paddle.base.framework.EagerParamBase):
            self.add_parameter(name="weights", parameter=weights)
        else:
            self.register_buffer("weights", weights)
        self.factors = FactorList(factors)

    @classmethod
    def new(cls, shape, rank, device=None, dtype=None, **kwargs):
        rank = tl.cp_tensor.validate_cp_rank(shape, rank)
        weights = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.empty(shape=[rank], dtype=dtype)
        )
        factors = [
            paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.empty(shape=(s, rank), dtype=dtype)
            )
            for s in shape
        ]
        return cls(weights, factors)

    @classmethod
    def from_tensor(cls, tensor, rank="same", **kwargs):
        shape = tuple(tensor.shape)
        rank = tl.cp_tensor.validate_cp_rank(shape, rank)
        dtype = tensor.dtype
        with paddle.no_grad():
            weights, factors = parafac(tensor.to("float64"), rank, **kwargs)
        return cls(
            paddle.base.framework.EagerParamBase.from_tensor(tensor=weights.to(dtype).contiguous()),
            [
                paddle.base.framework.EagerParamBase.from_tensor(tensor=f.to(dtype).contiguous())
                for f in factors
            ],
        )

    def init_from_tensor(self, tensor, l2_reg=1e-05, **kwargs):
        with paddle.no_grad():
            weights, factors = parafac(tensor, self.rank, l2_reg=l2_reg, **kwargs)
        self.weights = paddle.base.framework.EagerParamBase.from_tensor(tensor=weights.contiguous())
        self.factors = FactorList(
            [
                paddle.base.framework.EagerParamBase.from_tensor(tensor=f.contiguous())
                for f in factors
            ]
        )
        return self

    @property
    def decomposition(self):
        return self.weights, self.factors

    def to_tensor(self):
        return tl.cp_to_tensor(self.decomposition)

    def normal_(self, mean=0, std=1):
        super().normal_(mean=mean, std=std)
        std_factors = (std / math.sqrt(self.rank)) ** (1 / self.order)
        with paddle.no_grad():
            self.weights.fill_(value=1)
            for factor in self.factors:
                factor.data.normal_(0, std_factors)
        return self

    def __getitem__(self, indices):
        if isinstance(indices, int):
            mixing_factor, *factors = self.factors
            weights = self.weights * mixing_factor[indices, :]
            return self.__class__(weights, factors)
        elif isinstance(indices, slice):
            mixing_factor, *factors = self.factors
            factors = [mixing_factor[indices, :], *factors]
            weights = self.weights
            return self.__class__(weights, factors)
        else:
            factors = self.factors
            index_factors = []
            weights = self.weights
            for index in indices:
                if index is Ellipsis:
                    raise ValueError(
                        f"Ellipsis is not yet supported, yet got indices={indices} which contains one."
                    )
                mixing_factor, *factors = factors
                if isinstance(index, (np.integer, int)):
                    if factors or index_factors:
                        weights = weights * mixing_factor[index, :]
                    else:
                        return tl.sum(weights * mixing_factor[index, :])
                else:
                    index_factors.append(mixing_factor[index, :])
            return self.__class__(weights, index_factors + factors)

    def transduct(self, new_dim, mode=0, new_factor=None):
        """Transduction adds a new dimension to the existing factorization

        Parameters
        ----------
        new_dim : int
            dimension of the new mode to add
        mode : where to insert the new dimension, after the channels, default is 0
            by default, insert the new dimensions before the existing ones
            (e.g. add time before height and width)

        Returns
        -------
        self
        """
        factors = self.factors
        self.order += 1
        self.shape = self.shape[:mode] + (new_dim,) + self.shape[mode:]
        if new_factor is None:
            new_factor = paddle.ones(shape=[new_dim, self.rank])
        factors.insert(
            mode,
            paddle.base.framework.EagerParamBase.from_tensor(
                tensor=new_factor.to(factors[0].place).contiguous()
            ),
        )
        self.factors = FactorList(factors)
        return self


class TuckerTensor(FactorizedTensor, name="Tucker"):
    """Tucker Factorization

    Parameters
    ----------
    core
    factors
    shape
    rank
    """

    def __init__(self, core, factors, shape=None, rank=None):
        super().__init__()
        if shape is not None and rank is not None:
            self.shape, self.rank = shape, rank
        else:
            self.shape, self.rank = tl.tucker_tensor._validate_tucker_tensor((core, factors))
        self.order = len(self.shape)
        if isinstance(core, paddle.base.framework.EagerParamBase):
            self.add_parameter(name="core", parameter=core)
        else:
            self.register_buffer("core", core)
        self.factors = FactorList(factors)

    @classmethod
    def new(cls, shape, rank, fixed_rank_modes=None, device=None, dtype=None, **kwargs):
        rank = tl.tucker_tensor.validate_tucker_rank(shape, rank, fixed_modes=fixed_rank_modes)
        core = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.empty(shape=rank, dtype=dtype)
        )
        factors = [
            paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.empty(shape=(s, r), dtype=dtype)
            )
            for s, r in zip(shape, rank)
        ]
        return cls(core, factors)

    @classmethod
    def from_tensor(cls, tensor, rank="same", fixed_rank_modes=None, **kwargs):
        shape = tuple(tensor.shape)
        rank = tl.tucker_tensor.validate_tucker_rank(shape, rank, fixed_modes=fixed_rank_modes)
        with paddle.no_grad():
            core, factors = tucker(tensor, rank, **kwargs)
        return cls(
            paddle.base.framework.EagerParamBase.from_tensor(tensor=core.contiguous()),
            [
                paddle.base.framework.EagerParamBase.from_tensor(tensor=f.contiguous())
                for f in factors
            ],
        )

    def init_from_tensor(self, tensor, unsqueezed_modes=None, unsqueezed_init="average", **kwargs):
        """Initialize the tensor factorization from a tensor

        Parameters
        ----------
        tensor : paddle.Tensor
            full tensor to decompose
        unsqueezed_modes : int list
            list of modes for which the rank is 1 that don't correspond to a mode in the full tensor
            essentially we are adding a new dimension for which the core has dim 1,
            and that is not initialized through decomposition.
            Instead first `tensor` is decomposed into the other factors.
            The `unsqueezed factors` are then added and  initialized e.g. with 1/dim[i]
        unsqueezed_init : 'average' or float
            if unsqueezed_modes, this is how the added "unsqueezed" factors will be initialized
            if 'average', then unsqueezed_factor[i] will have value 1/tensor.shape[i]
        """
        if unsqueezed_modes is not None:
            unsqueezed_modes = sorted(unsqueezed_modes)
            for mode in unsqueezed_modes[::-1]:
                if self.rank[mode] != 1:
                    msg = (
                        "It is only possible to initialize by averagig over mode for which rank=1."
                    )
                    msg += f"However, got unsqueezed_modes={unsqueezed_modes} but rank[{mode}]={self.rank[mode]} != 1."
                    raise ValueError(msg)
            rank = tuple(r for i, r in enumerate(self.rank) if i not in unsqueezed_modes)
        else:
            rank = self.rank
        with paddle.no_grad():
            core, factors = tucker(tensor, rank, **kwargs)
            if unsqueezed_modes is not None:
                for mode in unsqueezed_modes:
                    size = self.shape[mode]
                    factor = paddle.ones(shape=[size, 1])
                    if unsqueezed_init == "average":
                        factor /= size
                    else:
                        factor *= unsqueezed_init
                    factors.insert(mode, factor)
                    core = core.unsqueeze(axis=mode)
        self.core = paddle.base.framework.EagerParamBase.from_tensor(tensor=core.contiguous())
        self.factors = FactorList(
            [
                paddle.base.framework.EagerParamBase.from_tensor(tensor=f.contiguous())
                for f in factors
            ]
        )
        return self

    @property
    def decomposition(self):
        return self.core, self.factors

    def to_tensor(self):
        return tl.tucker_to_tensor(self.decomposition)

    def normal_(self, mean=0, std=1):
        if mean != 0:
            raise ValueError(f"Currently only mean=0 is supported, but got mean={mean}")
        r = np.prod([math.sqrt(r) for r in self.rank])
        std_factors = (std / r) ** (1 / (self.order + 1))
        with paddle.no_grad():
            self.core.data.normal_(mean=0, std=std_factors)
            for factor in self.factors:
                factor.data.normal_(0, std_factors)
        return self

    def __getitem__(self, indices):
        if isinstance(indices, int):
            mixing_factor, *factors = self.factors
            core = tenalg.mode_dot(self.core, mixing_factor[indices, :], 0)
            return self.__class__(core, factors)
        elif isinstance(indices, slice):
            mixing_factor, *factors = self.factors
            factors = [mixing_factor[indices, :], *factors]
            return self.__class__(self.core, factors)
        else:
            modes = []
            factors = []
            factors_contract = []
            for i, (index, factor) in enumerate(zip(indices, self.factors)):
                if index is Ellipsis:
                    raise ValueError(
                        f"Ellipsis is not yet supported, yet got indices={indices}, indices[{i}]={index}."
                    )
                if isinstance(index, int):
                    modes.append(i)
                    factors_contract.append(factor[index, :])
                else:
                    factors.append(factor[index, :])
            if modes:
                core = tenalg.multi_mode_dot(self.core, factors_contract, modes=modes)
            else:
                core = self.core
            factors = factors + self.factors[i + 1 :]
            if factors:
                return self.__class__(core, factors)
            return core


class TTTensor(FactorizedTensor, name="TT"):
    """Tensor-Train (Matrix-Product-State) Factorization

    Parameters
    ----------
    factors
    shape
    rank
    """

    def __init__(self, factors, shape=None, rank=None):
        super().__init__()
        if shape is None or rank is None:
            self.shape, self.rank = tl.tt_tensor._validate_tt_tensor(factors)
        else:
            self.shape, self.rank = shape, rank
        self.order = len(self.shape)
        self.factors = FactorList(factors)

    @classmethod
    def new(cls, shape, rank, device=None, dtype=None, **kwargs):
        rank = tl.tt_tensor.validate_tt_rank(shape, rank)
        factors = [
            paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.empty(shape=(rank[i], s, rank[i + 1]), dtype=dtype)
            )
            for i, s in enumerate(shape)
        ]
        return cls(factors)

    @classmethod
    def from_tensor(cls, tensor, rank="same", **kwargs):
        shape = tuple(tensor.shape)
        rank = tl.tt_tensor.validate_tt_rank(shape, rank)
        with paddle.no_grad():
            factors = tensor_train(tensor, rank)
        return cls(
            [
                paddle.base.framework.EagerParamBase.from_tensor(tensor=f.contiguous())
                for f in factors
            ]
        )

    def init_from_tensor(self, tensor, **kwargs):
        with paddle.no_grad():
            factors = tensor_train(tensor, self.rank)
        self.factors = FactorList(
            [
                paddle.base.framework.EagerParamBase.from_tensor(tensor=f.contiguous())
                for f in factors
            ]
        )
        self.rank = tuple([tuple(f.shape)[0] for f in factors] + [1])
        return self

    @property
    def decomposition(self):
        return self.factors

    def to_tensor(self):
        return tl.tt_to_tensor(self.decomposition)

    def normal_(self, mean=0, std=1):
        if mean != 0:
            raise ValueError(f"Currently only mean=0 is supported, but got mean={mean}")
        r = np.prod(self.rank)
        std_factors = (std / r) ** (1 / self.order)
        with paddle.no_grad():
            for factor in self.factors:
                factor.data.normal_(0, std_factors)
        return self

    def __getitem__(self, indices):
        if isinstance(indices, int):
            factor, next_factor, *factors = self.factors
            next_factor = tenalg.mode_dot(next_factor, factor[:, indices, :].squeeze(axis=1), 0)
            return self.__class__([next_factor, *factors])
        elif isinstance(indices, slice):
            mixing_factor, *factors = self.factors
            factors = [mixing_factor[:, indices], *factors]
            return self.__class__(factors)
        else:
            factors = []
            all_contracted = True
            for i, index in enumerate(indices):
                if index is Ellipsis:
                    raise ValueError(
                        f"Ellipsis is not yet supported, yet got indices={indices}, indices[{i}]={index}."
                    )
                if isinstance(index, int):
                    if i:
                        factor = tenalg.mode_dot(factor, self.factors[i][:, index, :].T, -1)
                    else:
                        factor = self.factors[i][:, index, :]
                else:
                    if i:
                        if all_contracted:
                            factor = tenalg.mode_dot(self.factors[i][:, index, :], factor, 0)
                        else:
                            factors.append(factor)
                            factor = self.factors[i][:, index, :]
                    else:
                        factor = self.factors[i][:, index, :]
                    all_contracted = False
            if factor.ndim == 2:
                if self.order == i + 1:
                    return factor.squeeze()
                else:
                    next_factor, *factors = self.factors[i + 1 :]
                    factor = tenalg.mode_dot(next_factor, factor, 0)
                    return self.__class__([factor, *factors])
            else:
                return self.__class__([*factors, factor, *self.factors[i + 1 :]])

    def transduct(self, new_dim, mode=0, new_factor=None):
        """Transduction adds a new dimension to the existing factorization

        Parameters
        ----------
        new_dim : int
            dimension of the new mode to add
        mode : where to insert the new dimension, after the channels, default is 0
            by default, insert the new dimensions before the existing ones
            (e.g. add time before height and width)

        Returns
        -------
        self
        """
        factors = self.factors
        self.order += 1
        new_rank = self.rank[mode]
        self.rank = self.rank[:mode] + (new_rank,) + self.rank[mode:]
        self.shape = self.shape[:mode] + (new_dim,) + self.shape[mode:]
        if new_factor is None:
            new_factor = paddle.zeros(shape=[new_rank, new_dim, new_rank])
            for i in range(new_dim):
                new_factor[:, i, :] = paddle.eye(num_rows=new_rank)
        factors.insert(
            mode,
            paddle.base.framework.EagerParamBase.from_tensor(
                tensor=new_factor.to(factors[0].place).contiguous()
            ),
        )
        self.factors = FactorList(factors)
        return self

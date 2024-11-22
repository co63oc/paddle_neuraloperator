import paddle
import types
from typing import Any
from .comm import get_model_parallel_group
from .helpers import split_tensor_along_dim
from .helpers import _reduce
from .helpers import _split
from .helpers import _gather


class _CopyToModelParallelRegion(paddle.autograd.PyLayer):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output, group=get_model_parallel_group())


class _ReduceFromModelParallelRegion(paddle.autograd.PyLayer):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_, group=get_model_parallel_group())

    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_, group=get_model_parallel_group())

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToModelParallelRegion(paddle.autograd.PyLayer):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_, dim_):
        return _split(input_, dim_, group=get_model_parallel_group())

    @staticmethod
    def forward(ctx, input_, dim_):
        ctx.dim = dim_
        return _split(input_, dim_, group=get_model_parallel_group())

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output, ctx.dim, group=get_model_parallel_group()
            ), None


class _GatherFromModelParallelRegion(paddle.autograd.PyLayer):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_, dim_):
        return _gather(input_, dim_, group=get_model_parallel_group())

    @staticmethod
    def forward(ctx, input_, dim_):
        ctx.dim = dim_
        return _gather(input_, dim_, group=get_model_parallel_group())

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output, ctx.dim, group=get_model_parallel_group()
            ), None


def copy_to_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_model_parallel_region(input_, dim):
    return _ScatterToModelParallelRegion.apply(input_, dim)


def gather_from_model_parallel_region(input_, dim):
    return _GatherFromModelParallelRegion.apply(input_, dim)

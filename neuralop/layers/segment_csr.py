import importlib
from typing import Literal

import paddle


def segment_csr(
    src: paddle.Tensor,
    indptr: paddle.Tensor,
    reduce: Literal["mean", "sum"],
    use_scatter=True,
):
    """segment_csr reduces all entries of a CSR-formatted
    matrix by summing or averaging over neighbors.

    Used to reduce features over neighborhoods
    in neuralop.layers.IntegralTransform

    If use_scatter is set to False or paddle_scatter is not
    properly built, segment_csr falls back to a naive Paddle implementation

    Note: the native version is mainly intended for running tests on
    CPU-only GitHub CI runners to get around a versioning issue.
    paddle_scatter should be installed and built if possible.

    Parameters
    ----------
    src : paddle.Tensor
        tensor of features for each point
    indptr : paddle.Tensor
        splits representing start and end indices
        of each neighborhood in src
    reduce : Literal['mean', 'sum']
        how to reduce a neighborhood. if mean,
        reduce by taking the average of all neighbors.
        Otherwise take the sum.
    """
    if reduce not in ["mean", "sum"]:
        raise ValueError("reduce must be one of 'mean', 'sum'")
    if importlib.util.find_spec("paddle_scatter") is not None and use_scatter:
        """only import paddle_scatter when cuda is available"""
        import paddle_scatter.segment_csr as scatter_segment_csr

        return scatter_segment_csr(src, indptr, reduce=reduce)
    else:
        if use_scatter:
            print(
                "Warning: use_scatter is True but paddle_scatter is not properly built.                   Defaulting to naive Paddle implementation"
            )
        # if batched, shape [b, n_reps, channels]
        # otherwise shape [n_reps, channels]
        if src.ndim == 3:
            batched = True
            point_dim = 1
        else:
            batched = False
            point_dim = 0
        # if batched, shape [b, n_out, channels]
        # otherwise shape [n_out, channels]
        output_shape = list(tuple(src.shape))
        n_out = tuple(indptr.shape)[point_dim] - 1
        output_shape[point_dim] = n_out
        out = paddle.zeros(shape=output_shape)
        for i in range(n_out):
            # reduce all indices pointed to in indptr from src into out
            if batched:
                from_idx = slice(None), slice(indptr[0, i], indptr[0, i + 1])
                ein_str = "bio->bo"
                start = indptr[0, i]
                n_nbrs = indptr[0, i + 1] - start
                to_idx = slice(None), i
            else:
                from_idx = slice(indptr[i], indptr[i + 1])
                ein_str = "io->o"
                start = indptr[i]
                n_nbrs = indptr[i + 1] - start
                to_idx = i
            src_from = src[from_idx]
            if n_nbrs > 0:
                to_reduce = paddle.einsum(ein_str, src_from)
                if reduce == "mean":
                    to_reduce /= n_nbrs
                out[to_idx] += to_reduce
        return out

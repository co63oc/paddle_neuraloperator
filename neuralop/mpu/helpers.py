# coding=utf-8
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle

from neuralop import paddle_aux  # noqa


def pad_helper(tensor, dim, new_size, mode="zero"):
    ndim = tensor.ndim
    dim = (dim + ndim) % ndim
    ndim_pad = ndim - dim
    output_shape = [(0) for _ in range(2 * ndim_pad)]
    orig_size = tuple(tensor.shape)[dim]
    output_shape[1] = new_size - orig_size
    tensor_pad = paddle.nn.functional.pad(
        x=tensor, pad=output_shape, mode="constant", value=0.0, pad_from_left_axis=False
    )
    if mode == "conj":
        lhs_slice = [
            (slice(0, x) if idx != dim else slice(orig_size, new_size))
            for idx, x in enumerate(tuple(tensor.shape))
        ]
        rhs_slice = [
            (slice(0, x) if idx != dim else slice(1, output_shape[1] + 1))
            for idx, x in enumerate(tuple(tensor.shape))
        ]
        tensor_pad[lhs_slice] = paddle.flip(x=paddle.conj(x=tensor_pad[rhs_slice]), axis=[dim])
    return tensor_pad


def truncate_helper(tensor, dim, new_size):
    ndim = tensor.ndim
    dim = (dim + ndim) % ndim
    output_slice = [
        (slice(0, x) if idx != dim else slice(0, new_size))
        for idx, x in enumerate(tuple(tensor.shape))
    ]
    tensor_trunc = tensor[output_slice].contiguous()
    return tensor_trunc


def split_tensor_along_dim(tensor, dim, num_chunks):
    assert (
        dim < tensor.dim()
    ), f"Error, tensor dimension is {tensor.dim()} which cannot be split along {dim}"
    assert (
        tuple(tensor.shape)[dim] % num_chunks == 0
    ), f"Error, cannot split dim {dim} evenly. Dim size is \
        {tuple(tensor.shape)[dim]} and requested numnber of splits is {num_chunks}"
    chunk_size = tuple(tensor.shape)[dim] // num_chunks
    tensor_list = paddle_aux.split(x=tensor, num_or_sections=chunk_size, axis=dim)
    return tensor_list


# distributed primitives
def _transpose(tensor, dim0, dim1, group=None, async_op=False):
    # get comm params
    comm_size = paddle.distributed.get_world_size(group=group)
    # split and local transposition
    split_size = tuple(tensor.shape)[dim0] // comm_size
    x_send = [
        y.contiguous() for y in paddle_aux.split(x=tensor, num_or_sections=split_size, axis=dim0)
    ]
    x_recv = [paddle.empty_like(x=x_send[0]) for _ in range(comm_size)]
    # global transposition
    req = paddle.distributed.alltoall(out_tensor_list=x_recv, in_tensor_list=x_send, group=group)
    return x_recv, req


def _reduce(input_, use_fp32=True, group=None):
    """All-reduce the input tensor across model parallel group."""
    # Bypass the function if we are using only 1 GPU.
    if paddle.distributed.get_world_size(group=group) == 1:
        return input_

    # All-reduce.
    if use_fp32:
        dtype = input_.dtype
        inputf_ = input_.astype(dtype="float32")
        paddle.distributed.all_reduce(tensor=inputf_, group=group)
        input_ = inputf_.to(dtype)
    else:
        paddle.distributed.all_reduce(tensor=input_, group=group)
    return input_


def _split(input_, dim_, group=None):
    """Split the tensor along its last dimension and keep the corresponding slice."""
    # Bypass the function if we are using only 1 GPU.
    comm_size = paddle.distributed.get_world_size(group=group)
    if comm_size == 1:
        return input_
    # Split along last dimension.
    input_list = split_tensor_along_dim(input_, dim_, comm_size)
    # Note: paddle.split does not create contiguous tensors by default.
    rank = paddle.distributed.get_rank(group=group)
    output = input_list[rank].contiguous()
    return output


def _gather(input_, dim_, group=None):
    """Gather tensors and concatinate along the last dimension."""
    comm_size = paddle.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if comm_size == 1:
        return input_
    # sanity checks
    assert (
        dim_ < input_.dim()
    ), f"Error, cannot gather along {dim_} for tensor with {input_.dim()} dimensions."
    # Size and dimension.
    comm_rank = paddle.distributed.get_rank(group=group)
    tensor_list = [paddle.empty_like(x=input_) for _ in range(comm_size)]
    tensor_list[comm_rank] = input_.contiguous()
    paddle.distributed.all_gather(tensor_list=tensor_list, tensor=input_, group=group)
    # Note: paddle.concat already creates a contiguous tensor.
    output = paddle.concat(x=tensor_list, axis=dim_).contiguous()
    return output

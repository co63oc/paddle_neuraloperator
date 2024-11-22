import paddle
"""
Using `torchtnt` to count FLOPS
================================

In this example, we demonstrate how to use torchtnt to estimate the number of floating-point
operations per second (FLOPS) required for a model's forward and backward pass. 

We will use the FLOP computation to compare the resources used by a base FNO.
"""
from copy import deepcopy
from torchtnt.utils.flops import FlopTensorDispatchMode
from neuralop.models import FNO
device = 'cpu'
fno = FNO(n_modes=(64, 64), in_channels=3, out_channels=1, hidden_channels=
    64, projection_channels=64)
batch_size = 4
model_input = paddle.randn(shape=[batch_size, 3, 128, 128])
with FlopTensorDispatchMode(fno) as ftdm:
    res = fno(model_input).mean()
    fno_forward_flops = deepcopy(ftdm.flop_counts)
    ftdm.reset()
    res.backward()
    fno_backward_flops = deepcopy(ftdm.flop_counts)
print(fno_forward_flops)
from collections import defaultdict


def get_max_flops(flop_count_dict, max_value=0):
    for _, value in flop_count_dict.items():
        if isinstance(value, int):
            max_value = max(max_value, value)
        elif isinstance(value, defaultdict):
            new_val = get_max_flops(value, max_value)
            max_value = max(max_value, new_val)
    return max_value


print(f'Max FLOPS required for FNO.forward: {get_max_flops(fno_forward_flops)}'
    )
print(
    f'Max FLOPS required for FNO.backward: {get_max_flops(fno_backward_flops)}'
    )

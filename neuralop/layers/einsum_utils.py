import paddle
import opt_einsum
import tensorly as tl
from tensorly.plugins import use_opt_einsum
tl.set_backend('pytorch')
use_opt_einsum('optimal')


def einsum_complexhalf_two_input(eq, a, b):
    """
    Compute (two-input) einsum for complexhalf tensors.
    Because torch.einsum currently does not support complex32 (complexhalf) types.
    The inputs and outputs are the same as in torch.einsum
    """
    assert len(eq.split(',')) == 2, 'Equation must have two inputs.'
    a = paddle.as_real(x=a)
    b = paddle.as_real(x=b)
    a = a.astype(dtype='float16')
    b = b.astype(dtype='float16')
    input_output = eq.split('->')
    new_output = 'xy' + input_output[1]
    input_terms = input_output[0].split(',')
    new_inputs = [input_terms[0] + 'x', input_terms[1] + 'y']
    new_eqn = new_inputs[0] + ',' + new_inputs[1] + '->' + new_output
    tmp = tl.einsum(new_eqn, a, b)
    res = paddle.stack(x=[tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] +
        tmp[0, 1, ...]], axis=-1)
    return paddle.as_complex(x=res)


def einsum_complexhalf(eq, *args):
    """
    Compute einsum for complexhalf tensors.
    Because torch.einsum currently does not support complex32 (complexhalf) types.
    The inputs and outputs are the same as in torch.einsum
    """
    if len(args) == 2:
        return einsum_complexhalf_two_input(eq, *args)
    _, path_info = opt_einsum.contract_path(eq, *args)
    partial_eqns = [contraction_info[2] for contraction_info in path_info.
        contraction_list]
    tensors = {}
    input_labels = eq.split('->')[0].split(',')
    output_label = eq.split('->')[1]
    tensors = dict(zip(input_labels, args))
    for key, tensor in tensors.items():
        tensor = paddle.as_real(x=tensor)
        tensor = tensor.astype(dtype='float16')
        tensors[key] = tensor
    for partial_eq in partial_eqns:
        in_labels, out_label = partial_eq.split('->')
        in_labels = in_labels.split(',')
        in_tensors = [tensors[label] for label in in_labels]
        input_output = partial_eq.split('->')
        new_output = 'xy' + input_output[1]
        input_terms = input_output[0].split(',')
        new_inputs = [input_terms[0] + 'x', input_terms[1] + 'y']
        new_eqn = new_inputs[0] + ',' + new_inputs[1] + '->' + new_output
        tmp = tl.einsum(new_eqn, *in_tensors)
        result = paddle.stack(x=[tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0,
            ...] + tmp[0, 1, ...]], axis=-1)
        tensors[out_label] = result
    return paddle.as_complex(x=tensors[output_label])

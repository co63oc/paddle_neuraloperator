import tensorly as tl
from tensorly.tenalg.tenalg_utils import _validate_contraction_modes

tl.set_backend("numpy")
einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def tensor_dot_tucker(tensor, tucker, modes, batched_modes=()):
    """Batched tensor contraction between a dense tensor and a Tucker tensor on specified modes

    Parameters
    ----------
    tensor : DenseTensor
    tucker : TuckerTensor
    modes : int list or int
        modes on which to contract tensor1 and tensor2
    batched_modes : int or tuple[int]

    Returns
    -------
    contraction : tensor contracted with cp on the specified modes
    """
    modes_tensor, modes_tucker = _validate_contraction_modes(
        tl.shape(tensor), tucker.tensor_shape, modes
    )
    input_order = tensor.ndim
    weight_order = tucker.order
    batched_modes_tensor, batched_modes_tucker = _validate_contraction_modes(
        tl.shape(tensor), tucker.tensor_shape, batched_modes
    )
    sorted_modes_tucker = sorted(modes_tucker + batched_modes_tucker, reverse=True)
    sorted_modes_tensor = sorted(modes_tensor + batched_modes_tensor, reverse=True)
    rank_sym = [einsum_symbols[i] for i in range(weight_order)]
    tucker_sym = [einsum_symbols[i + weight_order] for i in range(weight_order)]
    tensor_sym = [einsum_symbols[i + 2 * weight_order] for i in range(tensor.ndim)]
    output_sym = tensor_sym + tucker_sym
    for m in sorted_modes_tucker:
        if m in modes_tucker:
            output_sym.pop(m + input_order)
    for m in sorted_modes_tensor:
        output_sym.pop(m)
    for i, e in enumerate(modes_tensor):
        tensor_sym[e] = tucker_sym[modes_tucker[i]]
    for i, e in enumerate(batched_modes_tensor):
        tensor_sym[e] = tucker_sym[batched_modes_tucker[i]]
    eq = "".join(tensor_sym)
    eq += "," + "".join(rank_sym)
    eq += "," + ",".join(f"{s}{r}" for s, r in zip(tucker_sym, rank_sym))
    eq += "->" + "".join(output_sym)
    return tl.einsum(eq, tensor, tucker.core, *tucker.factors)


def tensor_dot_cp(tensor, cp, modes):
    """Contracts a to CP tensors in factorized form

    Returns
    -------
    tensor = tensor x cp_matrix.to_matrix().T
    """
    try:
        cp_shape = cp.tensor_shape
    except AttributeError:
        cp_shape = cp.shape
    modes_tensor, modes_cp = _validate_contraction_modes(tl.shape(tensor), cp_shape, modes)
    tensor_order = tl.ndim(tensor)
    start = ord("b")
    eq_in = "".join(f"{chr(start + index)}" for index in range(tensor_order))
    eq_factors = []
    eq_res = "".join(eq_in[i] if i not in modes_tensor else "" for i in range(tensor_order))
    counter_joint = 0
    counter_free = 0
    for i in range(len(cp.factors)):
        if i in modes_cp:
            eq_factors.append(f"{eq_in[modes_tensor[counter_joint]]}a")
            counter_joint += 1
        else:
            eq_factors.append(f"{chr(start + tensor_order + counter_free)}a")
            eq_res += f"{chr(start + tensor_order + counter_free)}"
            counter_free += 1
    eq_factors = ",".join(f for f in eq_factors)
    eq = eq_in + ",a," + eq_factors + "->" + eq_res
    res = tl.einsum(eq, tensor, cp.weights, *cp.factors)
    return res

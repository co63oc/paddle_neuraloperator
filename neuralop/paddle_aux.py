
# This file is generated by PaConvert ToolKit, please Don't edit it!
import paddle

def view(self, *args, **kwargs):
    if args:
        if len(args)==1 and isinstance(args[0], (tuple, list, str)):
            return paddle.view(self, args[0])
        else:
            return paddle.view(self, list(args))
    elif kwargs:
        return paddle.view(self, shape_or_dtype = list(kwargs.values())[0])

setattr(paddle.Tensor, 'view', view)

def reshape(self, *args, **kwargs):
    if args:
        if len(args)==1 and isinstance(args[0], (tuple, list)):
            return paddle.reshape(self, args[0])
        else:
            return paddle.reshape(self, list(args))
    elif kwargs:
        assert 'shape' in kwargs
        return paddle.reshape(self, shape=kwargs['shape'])

setattr(paddle.Tensor, 'reshape', reshape)

def min(*args, **kwargs):
    if 'input' in kwargs:
        kwargs['x'] = kwargs.pop('input')

    out_v = None
    if 'out' in kwargs:
        out_v = kwargs.pop('out')

    if 'other' in kwargs:
        kwargs['y'] = kwargs.pop('other')
        ret = paddle.minimum(*args, **kwargs)
    elif len(args)==2 and isinstance(args[1], paddle.Tensor):
        ret = paddle.minimum(*args, **kwargs)
    else:
        if 'dim' in kwargs:
            kwargs['axis'] = kwargs.pop('dim')

        if 'axis' in kwargs or len(args) >= 2:
            if out_v:
                ret = paddle.min(*args, **kwargs), paddle.argmin(*args, **kwargs)
                paddle.assign(ret[0], out_v[0])
                paddle.assign(ret[1], out_v[1])
                return out_v
            else:
                ret = paddle.min(*args, **kwargs), paddle.argmin(*args, **kwargs)
                return ret
        else:
            ret = paddle.min(*args, **kwargs)
            return ret

    if out_v:
        paddle.assign(ret, out_v)
        return out_v
    else:
        return ret

def max(*args, **kwargs):
    if 'input' in kwargs:
        kwargs['x'] = kwargs.pop('input')

    out_v = None
    if 'out' in kwargs:
        out_v = kwargs.pop('out')

    if 'other' in kwargs:
        kwargs['y'] = kwargs.pop('other')
        ret = paddle.maximum(*args, **kwargs)
    elif len(args)==2 and isinstance(args[1], paddle.Tensor):
        ret = paddle.maximum(*args, **kwargs)
    else:
        if 'dim' in kwargs:
            kwargs['axis'] = kwargs.pop('dim')

        if 'axis' in kwargs or len(args) >= 2:
            if out_v:
                ret = paddle.max(*args, **kwargs), paddle.argmax(*args, **kwargs)
                paddle.assign(ret[0], out_v[0])
                paddle.assign(ret[1], out_v[1])
                return out_v
            else:
                ret = paddle.max(*args, **kwargs), paddle.argmax(*args, **kwargs)
                return ret
            return out_v
        else:
            ret = paddle.max(*args, **kwargs)
            return ret

    if out_v:
        paddle.assign(ret, out_v)
        return out_v
    else:
        return ret

def split(x, num_or_sections, axis=0):
    if isinstance(num_or_sections, int):
        return paddle.split(x, x.shape[axis]//num_or_sections, axis)
    else:
        return paddle.split(x, num_or_sections, axis)


def min_class_func(self, *args, **kwargs):
    if 'other' in kwargs:
        kwargs['y'] = kwargs.pop('other')
        ret = paddle.minimum(self, *args, **kwargs)
    elif len(args)==1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.minimum(self, *args, **kwargs)
    else:
        if 'dim' in kwargs:
            kwargs['axis'] = kwargs.pop('dim')

        if 'axis' in kwargs or len(args) >= 1:
            ret = paddle.min(self, *args, **kwargs), paddle.argmin(self, *args, **kwargs)
        else:
            ret = paddle.min(self, *args, **kwargs)

    return ret

def max_class_func(self, *args, **kwargs):
    if 'other' in kwargs:
        kwargs['y'] = kwargs.pop('other')
        ret = paddle.maximum(self, *args, **kwargs)
    elif len(args)==1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.maximum(self, *args, **kwargs)
    else:
        if 'dim' in kwargs:
            kwargs['axis'] = kwargs.pop('dim')

        if 'axis' in kwargs or len(args) >= 1:
            ret = paddle.max(self, *args, **kwargs), paddle.argmax(self, *args, **kwargs)
        else:
            ret = paddle.max(self, *args, **kwargs)

    return ret

setattr(paddle.Tensor, "min", min_class_func)
setattr(paddle.Tensor, "max", max_class_func)

def transpose_aux_func(dims,dim0, dim1):
    perm = list(range(dims))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return perm
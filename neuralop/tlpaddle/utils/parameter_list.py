import paddle

from neuralop import paddle_aux


class FactorList(paddle.nn.Layer):
    def __init__(self, parameters=None):
        super().__init__()
        self.keys = []
        self.counter = 0
        if parameters is not None:
            self.extend(parameters)

    def _unique_key(self):
        """Creates a new unique key"""
        key = f"factor_{self.counter}"
        self.counter += 1
        return key

    def append(self, element):
        key = self._unique_key()
        if paddle.is_tensor(x=element):
            if isinstance(element, paddle.base.framework.EagerParamBase):
                self.add_parameter(name=key, parameter=element)
            else:
                self.register_buffer(key, element)
        else:
            setattr(self, key, self.__class__(element))
        self.keys.append(key)

    def insert(self, index, element):
        key = self._unique_key()
        setattr(self, key, element)
        self.keys.insert(index, key)

    def pop(self, index=-1):
        item = self[index]
        self.__delitem__(index)
        return item

    def __getitem__(self, index):
        keys = self.keys[index]
        if isinstance(keys, list):
            return self.__class__([getattr(self, key) for key in keys])
        return getattr(self, keys)

    def __setitem__(self, index, value):
        setattr(self, self.keys[index], value)

    def __delitem__(self, index):
        delattr(self, self.keys[index])
        self.keys.__delitem__(index)

    def __len__(self):
        return len(self.keys)

    def extend(self, parameters):
        for param in parameters:
            self.append(param)

    def __iadd__(self, parameters):
        return self.extend(parameters)

    def __add__(self, parameters):
        instance = self.__class__(self)
        instance.extend(parameters)
        return instance

    def __radd__(self, parameters):
        instance = self.__class__(parameters)
        instance.extend(self)
        return instance

    def extra_repr(self) -> str:
        child_lines = []
        for k, p in self._parameters.items():
            size_str = "x".join(str(size) for size in tuple(p.shape))
            device_str = (
                "" if not p.place.is_gpu_place() else " (GPU {})".format(p.place.gpu_device_id())
            )
            parastr = "Parameter containing: [{} of size {}{}]".format(
                p.dtype, size_str, device_str
            )
            child_lines.append("  (" + str(k) + "): " + parastr)
        tmpstr = "\n".join(child_lines)
        return tmpstr


class ComplexFactorList(FactorList):
    def __getitem__(self, index):
        if isinstance(index, int):
            value = getattr(self, self.keys[index])
            if paddle.is_tensor(x=value):
                value = paddle_aux.as_complex(x=value)
            return value
        else:
            keys = self.keys[index]
            return self.__class__([paddle_aux.as_complex(x=getattr(self, key)) for key in keys])

    def __setitem__(self, index, value):
        if paddle.is_tensor(x=value):
            value = paddle.as_real(x=value)
        setattr(self, self.keys[index], value)

    def register_parameter(self, key, value):
        value = paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.as_real(x=value))
        super().add_parameter(name=key, parameter=value)

    def register_buffer(self, key, value):
        value = paddle.as_real(x=value)
        super().register_buffer(name=key, tensor=value)


class ParameterList(paddle.nn.Layer):
    def __init__(self, parameters=None):
        super().__init__()
        self.keys = []
        self.counter = 0
        if parameters is not None:
            self.extend(parameters)

    def _unique_key(self):
        """Creates a new unique key"""
        key = f"param_{self.counter}"
        self.counter += 1
        return key

    def append(self, element):
        key = self._unique_key()
        self.add_parameter(name=key, parameter=element)
        self.keys.append(key)

    def insert(self, index, element):
        key = self._unique_key()
        self.add_parameter(name=key, parameter=element)
        self.keys.insert(index, key)

    def pop(self, index=-1):
        item = self[index]
        self.__delitem__(index)
        return item

    def __getitem__(self, index):
        keys = self.keys[index]
        if isinstance(keys, list):
            return self.__class__([getattr(self, key) for key in keys])
        return getattr(self, keys)

    def __setitem__(self, index, value):
        self.add_parameter(name=self.keys[index], parameter=value)

    def __delitem__(self, index):
        delattr(self, self.keys[index])
        self.keys.__delitem__(index)

    def __len__(self):
        return len(self.keys)

    def extend(self, parameters):
        for param in parameters:
            self.append(param)

    def __iadd__(self, parameters):
        return self.extend(parameters)

    def extra_repr(self) -> str:
        child_lines = []
        for k, p in self._parameters.items():
            size_str = "x".join(str(size) for size in tuple(p.shape))
            device_str = (
                "" if not p.place.is_gpu_place() else " (GPU {})".format(p.place.gpu_device_id())
            )
            parastr = "Parameter containing: [{} of size {}{}]".format(
                p.dtype, size_str, device_str
            )
            child_lines.append("  (" + str(k) + "): " + parastr)
        tmpstr = "\n".join(child_lines)
        return tmpstr

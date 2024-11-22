import paddle
from typing import Dict
from ...utils import count_tensor_params
from .base_transforms import Transform, DictTransform


class Normalizer(Transform):

    def __init__(self, mean, std, eps=1e-06):
        self.mean = mean
        self.std = std
        self.eps = eps

    def transform(self, data):
        return (data - self.mean) / (self.std + self.eps)

    def inverse_transform(self, data):
        return data * (self.std + self.eps) + self.mean

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

    def cuda(self):
        self.mean = self.mean.cuda(blocking=True)
        self.std = self.std.cuda(blocking=True)

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


class UnitGaussianNormalizer(Transform):
    """
    UnitGaussianNormalizer normalizes data to be zero mean and unit std.
    """

    def __init__(self, mean=None, std=None, eps=1e-07, dim=None, mask=None):
        """
        mean : torch.tensor or None
            has to include batch-size as a dim of 1
            e.g. for tensors of shape ``(batch_size, channels, height, width)``,
            the mean over height and width should have shape ``(1, channels, 1, 1)``
        std : torch.tensor or None
        eps : float, default is 0
            for safe division by the std
        dim : int list, default is None
            if not None, dimensions of the data to reduce over to compute the mean and std.

            .. important::

                Has to include the batch-size (typically 0).
                For instance, to normalize data of shape ``(batch_size, channels, height, width)``
                along batch-size, height and width, pass ``dim=[0, 2, 3]``

        mask : torch.Tensor or None, default is None
            If not None, a tensor with the same size as a sample,
            with value 0 where the data should be ignored and 1 everywhere else

        Notes
        -----
        The resulting mean will have the same size as the input MINUS the specified dims.
        If you do not specify any dims, the mean and std will both be scalars.

        Returns
        -------
        UnitGaussianNormalizer instance
        """
        super().__init__()
        self.register_buffer(name='mean', tensor=mean)
        self.register_buffer(name='std', tensor=std)
        self.register_buffer(name='mask', tensor=mask)
        self.eps = eps
        if mean is not None:
            self.ndim = mean.ndim
        if isinstance(dim, int):
            dim = [dim]
        self.dim = dim
        self.n_elements = 0

    def fit(self, data_batch):
        self.update_mean_std(data_batch)

    def partial_fit(self, data_batch, batch_size=1):
        if 0 in list(tuple(data_batch.shape)):
            return
        count = 0
        n_samples = len(data_batch)
        while count < n_samples:
            samples = data_batch[count:count + batch_size]
            if self.n_elements:
                self.incremental_update_mean_std(samples)
            else:
                self.update_mean_std(samples)
            count += batch_size

    def update_mean_std(self, data_batch):
        self.ndim = data_batch.ndim
        if self.mask is None:
            self.n_elements = count_tensor_params(data_batch, self.dim)
            self.mean = paddle.mean(x=data_batch, axis=self.dim, keepdim=True)
            self.squared_mean = paddle.mean(x=data_batch ** 2, axis=self.
                dim, keepdim=True)
            self.std = paddle.std(x=data_batch, axis=self.dim, keepdim=True)
        else:
            batch_size = tuple(data_batch.shape)[0]
            dim = [(i - 1) for i in self.dim if i]
            shape = [s for i, s in enumerate(tuple(self.mask.shape)) if i
                 not in dim]
            self.n_elements = paddle.count_nonzero(x=self.mask, axis=dim
                ) * batch_size
            self.mean = paddle.zeros(shape=shape)
            self.std = paddle.zeros(shape=shape)
            self.squared_mean = paddle.zeros(shape=shape)
            data_batch[:, self.mask == 1] = 0
            self.mean[self.mask == 1] = paddle.sum(x=data_batch, axis=dim,
                keepdim=True) / self.n_elements
            self.squared_mean = paddle.sum(x=data_batch ** 2, axis=dim,
                keepdim=True) / self.n_elements
            self.std = paddle.std(x=data_batch, axis=self.dim, keepdim=True)

    def incremental_update_mean_std(self, data_batch):
        if self.mask is None:
            n_elements = count_tensor_params(data_batch, self.dim)
            dim = self.dim
        else:
            dim = [(i - 1) for i in self.dim if i]
            n_elements = paddle.count_nonzero(x=self.mask, axis=dim) * tuple(
                data_batch.shape)[0]
            data_batch[:, self.mask == 1] = 0
        self.mean = 1.0 / (self.n_elements + n_elements) * (self.n_elements *
            self.mean + paddle.sum(x=data_batch, axis=dim, keepdim=True))
        self.squared_mean = 1.0 / (self.n_elements + n_elements) * (self.
            n_elements * self.squared_mean + paddle.sum(x=data_batch ** 2,
            axis=dim, keepdim=True))
        self.n_elements += n_elements
        self.std = paddle.sqrt(x=self.squared_mean - self.mean ** 2
            ) * self.n_elements / (self.n_elements - 1)

    def transform(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def inverse_transform(self, x):
        return x * (self.std + self.eps) + self.mean

    def forward(self, x):
        return self.transform(x)

    def cuda(self):
        self.mean = self.mean.cuda(blocking=True)
        self.std = self.std.cuda(blocking=True)
        return self

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        return self

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    @classmethod
    def from_dataset(cls, dataset, dim=None, keys=None, mask=None):
        """Return a dictionary of normalizer instances, fitted on the given dataset

        Parameters
        ----------
        dataset : pytorch dataset
            each element must be a dict {key: sample}
            e.g. {'x': input_samples, 'y': target_labels}
        dim : int list, default is None
            * If None, reduce over all dims (scalar mean and std)
            * Otherwise, must include batch-dimensions and all over dims to reduce over
        keys : str list or None
            if not None, a normalizer is instanciated only for the given keys
        """
        for i, data_dict in enumerate(dataset):
            if not i:
                if not keys:
                    keys = data_dict.keys()
        instances = {key: cls(dim=dim, mask=mask) for key in keys}
        for i, data_dict in enumerate(dataset):
            for key, sample in data_dict.items():
                if key in keys:
                    instances[key].partial_fit(sample.unsqueeze(axis=0))
        return instances


class DictUnitGaussianNormalizer(DictTransform):
    """DictUnitGaussianNormalizer composes
    DictTransform and UnitGaussianNormalizer to normalize different
    fields of a model output tensor to Gaussian distributions w/
    mean 0 and unit variance.

        Parameters
        ----------
        normalizer_dict : Dict[str, UnitGaussianNormalizer]
            dictionary of normalizers, keyed to fields
        input_mappings : Dict[slice]
            slices of input tensor to grab per field, must share keys with above
        return_mappings : Dict[slice]
            _description_
        """

    def __init__(self, normalizer_dict: Dict[str, UnitGaussianNormalizer],
        input_mappings: Dict[str, slice], return_mappings: Dict[str, slice]):
        assert set(normalizer_dict.keys()) == set(input_mappings.keys()
            ), 'Error: normalizers and model input fields must be keyed identically'
        assert set(normalizer_dict.keys()) == set(return_mappings.keys()
            ), 'Error: normalizers and model output fields must be keyed identically'
        super().__init__(transform_dict=normalizer_dict, input_mappings=
            input_mappings, return_mappings=return_mappings)

    @classmethod
    def from_dataset(cls, dataset, dim=None, keys=None, mask=None):
        """Return a dictionary of normalizer instances, fitted on the given dataset

        Parameters
        ----------
        dataset : pytorch dataset
            each element must be a dict {key: sample}
            e.g. {'x': input_samples, 'y': target_labels}
        dim : int list, default is None
            * If None, reduce over all dims (scalar mean and std)
            * Otherwise, must include batch-dimensions and all over dims to reduce over
        keys : str list or None
            if not None, a normalizer is instanciated only for the given keys
        """
        for i, data_dict in enumerate(dataset):
            if not i:
                if not keys:
                    keys = data_dict.keys()
        instances = {key: cls(dim=dim, mask=mask) for key in keys}
        for i, data_dict in enumerate(dataset):
            for key, sample in data_dict.items():
                if key in keys:
                    instances[key].partial_fit(sample.unsqueeze(axis=0))
        return instances

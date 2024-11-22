import sys
sys.path.append('/nfs/github/paddle/paddle_neuraloperator/utils')
import paddle_aux
import paddle
from pathlib import Path
from typing import Optional, List, Union
import numpy as np
from .tensor_dataset import TensorDataset
from .pt_dataset import PTDataset


class Burgers1dTimeDataset(PTDataset):
    """
    Burgers1dTimeDataset wraps data from the viscous 
    Burger's equation in 1 spatial dimension.
    This dataset is not available for download online, but we
    provide a low-res version on 16 spatial points

    Attributes
    ----------
    train_db: torch.utils.data.Dataset of training examples
    test_db:  ""                       of test examples
    data_processor: neuralop.datasets.DataProcessor to process data examples
        optional, default is None
    """

    def __init__(self, root_dir: Union[Path, str], n_train: int, n_tests:
        list[int], train_resolution: int=16, test_resolutions: List[int]=[
        16], batch_size: int=32, test_batch_sizes: List[int]=32,
        temporal_subsample: Optional[int]=None, spatial_subsample: Optional
        [int]=None, pad: int=0, channel_dim: int=1, download: bool=True):
        """
        Parameters
        ----------
        root_dir : Union[Path, str]
            root at which to download data files
        dataset_name : str
            prefix of pt data files to store/access
        n_train : int
            number of train instances
        n_tests : List[int]
            number of test instances per test dataset
        batch_size : int
            batch size of training set
        test_batch_sizes : List[int]
            batch size of test sets
        train_resolution : int
            resolution of data for training set
        test_resolutions : List[int], optional
            resolution of data for testing sets, by default [16,32]
        encode_input : bool, optional
            whether to normalize inputs in provided DataProcessor,
            by default False
        encode_output : bool, optional
            whether to normalize outputs in provided DataProcessor,
            by default True
        encoding : str, optional
            parameter for input/output normalization. Whether
            to normalize by channel ("channel-wise") or 
            by pixel ("pixel-wise"), default "channel-wise"
        input_subsampling_rate : int or List[int], optional
            rate at which to subsample each input dimension, by default None
        output_subsampling_rate : int or List[int], optional
            rate at which to subsample each output dimension, by default None
        channel_dim : int, optional
            dimension of saved tensors to index data channels, by default 1
        """
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        if not root_dir.exists():
            root_dir.mkdir(parents=True)
        available_resolutions = [16, 128]
        assert train_resolution in available_resolutions, f'Resolutions available: {available_resolutions}, got {train_resolution}'
        for res in test_resolutions:
            assert res in available_resolutions, f'Resolutions available: {available_resolutions}, got {res}'
        super().__init__(root_dir=root_dir, n_train=n_train, n_tests=
            n_tests, batch_size=batch_size, test_batch_sizes=
            test_batch_sizes, train_resolution=train_resolution,
            test_resolutions=test_resolutions, input_subsampling_rate=
            spatial_subsample, output_subsampling_rate=[temporal_subsample,
            spatial_subsample], encode_input=True, encode_output=True,
            encoding='channel-wise', channel_dim=channel_dim, dataset_name=
            'burgers')


def load_burgers_1d(data_path, n_train, n_test, batch_train=32, batch_test=
    100, time=1, grid=[0, 1]):
    data_path = Path(data_path).joinpath('burgers.pt').as_posix()
    data = paddle.load(path=str(data_path))
    x_train = data[0:n_train, :, 0]
    x_test = data[n_train:n_train + n_test, :, 0]
    y_train = data[0:n_train, :, time]
    y_test = data[n_train:n_train + n_test, :, time]
    s = x_train.shape[-1]
    if grid is not None:
        grid = paddle.linspace(start=grid[0], stop=grid[1], num=s + 1)[0:-1
            ].view(1, -1)
        grid_train = grid.tile(repeat_times=[n_train, 1])
        grid_test = grid.tile(repeat_times=[n_test, 1])
        x_train = paddle.concat(x=(x_train.unsqueeze(axis=1), grid_train.
            unsqueeze(axis=1)), axis=1)
        x_test = paddle.concat(x=(x_test.unsqueeze(axis=1), grid_test.
            unsqueeze(axis=1)), axis=1)
    train_loader = paddle.io.DataLoader(dataset=paddle.io.TensorDataset([
        x_train, y_train]), batch_size=batch_train, shuffle=False)
    test_loader = paddle.io.DataLoader(dataset=paddle.io.TensorDataset([
        x_test, y_test]), batch_size=batch_test, shuffle=False)
    return train_loader, test_loader


def load_burgers_1dtime(data_path, n_train, n_test, batch_size=32,
    batch_size_test=100, temporal_length=101, spatial_length=128,
    temporal_subsample=1, spatial_subsample=1, pad=0):
    """
    Load burgers.mat data. Given the initial condition (t=0),
    predict timesteps 1 to temporal_length.
    """
    with np.load(data_path) as data:
        x_data = data['input']
        y_data = data['output']
        visc = data['visc']
    x_data = paddle.to_tensor(data=x_data.astype(np.float32))
    x_data = x_data[:, :spatial_length:spatial_subsample]
    y_data = paddle.to_tensor(data=y_data.astype(np.float32))
    y_data = y_data[:, :temporal_length:temporal_subsample, :spatial_length
        :spatial_subsample]
    visc = paddle.to_tensor(data=visc.astype(np.float32)).item()
    x_train = x_data[:n_train]
    y_train = y_data[:n_train]
    x_test = x_data[n_train:n_train + n_test]
    y_test = y_data[n_train:n_train + n_test]
    domain_lengths = [spatial_length / 128, (temporal_length - 1) / 100]
    domain_starts = [0.0, 0.0]
    spatial_length = spatial_length // spatial_subsample
    temporal_length = temporal_length // temporal_subsample
    if pad:
        x_train = paddle.nn.Pad1D(padding=pad, mode='replicate')(x_train)
        x_test = paddle.nn.Pad1D(padding=pad, mode='replicate')(x_test)
        spatial_length += 2 * pad
        temporal_length += 2 * pad
        incrs = [spatial_subsample / 128, temporal_subsample / 100]
        domain_lengths = [(d + incr * pad) for d, incr in zip(
            domain_lengths, incrs)]
        domain_starts = [(-incr * pad) for incr in incrs]
    grid_x = paddle.to_tensor(data=np.linspace(domain_starts[0],
        domain_lengths[0], spatial_length + 1)[:-1], dtype='float32')
    grid_t = paddle.to_tensor(data=np.linspace(domain_starts[1],
        domain_lengths[1], temporal_length), dtype='float32')
    grid_x = grid_x.reshape(1, 1, spatial_length)
    grid_t = grid_t.reshape(1, temporal_length, 1)
    x_train = x_train.reshape(n_train, 1, spatial_length).tile(repeat_times
        =[1, temporal_length, 1])
    x_test = x_test.reshape(n_test, 1, spatial_length).tile(repeat_times=[1,
        temporal_length, 1])
    x_train = paddle.stack(x=[x_train, grid_t.tile(repeat_times=[n_train, 1,
        spatial_length]), grid_x.tile(repeat_times=[n_train,
        temporal_length, 1])], axis=3)
    x_test = paddle.stack(x=[x_test, grid_t.tile(repeat_times=[n_test, 1,
        spatial_length]), grid_x.tile(repeat_times=[n_test, temporal_length,
        1])], axis=3)
    x_train = x_train.transpose(perm=[0, 3, 1, 2])
    x_test = x_test.transpose(perm=[0, 3, 1, 2])
    y_train = y_train.unsqueeze(axis=1)
    y_test = y_test.unsqueeze(axis=1)
    train_db = TensorDataset(x_train, y_train)
    train_loader = paddle.io.DataLoader(dataset=train_db, batch_size=
        batch_size, shuffle=False)
    test_db = TensorDataset(x_test, y_test)
    test_loader = paddle.io.DataLoader(dataset=test_db, batch_size=
        batch_size_test, shuffle=False)
    output_encoder = None
    test_loaders = {'test': test_loader}
    return train_loader, test_loaders, output_encoder

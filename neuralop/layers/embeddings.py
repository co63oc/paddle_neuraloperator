import sys
sys.path.append('/nfs/github/paddle/paddle_neuraloperator/utils')
import paddle_aux
import paddle
from abc import ABC, abstractmethod
from typing import List


class Embedding(paddle.nn.Layer, ABC):

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def out_channels(self):
        pass


class GridEmbedding2D(Embedding):
    """A simple positional embedding as a regular 2D grid
    """

    def __init__(self, in_channels: int, grid_boundaries=[[0, 1], [0, 1]]):
        """GridEmbedding2D applies a simple positional 
        embedding as a regular 2D grid

        Parameters
        ----------
        in_channels : int
            number of channels in input. Fixed for output channel interface
        grid_boundaries : list, optional
            coordinate boundaries of input grid, by default [[0, 1], [0, 1]]
        """
        super().__init__()
        self.in_channels = in_channels
        self.grid_boundaries = grid_boundaries
        self._grid = None
        self._res = None

    @property
    def out_channels(self):
        return self.in_channels + 2

    def grid(self, spatial_dims, device, dtype):
        """grid generates 2D grid needed for pos encoding
        and caches the grid associated with MRU resolution

        Parameters
        ----------
        spatial_dims : torch.size
             sizes of spatial resolution
        device : literal 'cpu' or 'cuda:*'
            where to load data
        dtype : str
            dtype to encode data

        Returns
        -------
        torch.tensor
            output grids to concatenate 
        """
        if self._grid is None or self._res != spatial_dims:
            grid_x, grid_y = regular_grid_2d(spatial_dims, grid_boundaries=
                self.grid_boundaries)
            grid_x = grid_x.to(device).to(dtype).unsqueeze(axis=0).unsqueeze(
                axis=0)
            grid_y = grid_y.to(device).to(dtype).unsqueeze(axis=0).unsqueeze(
                axis=0)
            self._grid = grid_x, grid_y
            self._res = spatial_dims
        return self._grid

    def forward(self, data, batched=True):
        if not batched:
            if data.ndim == 3:
                data = data.unsqueeze(axis=0)
        batch_size = tuple(data.shape)[0]
        x, y = self.grid(tuple(data.shape)[-2:], data.place, data.dtype)
        out = paddle.concat(x=(data, x.expand(shape=[batch_size, -1, -1, -1
            ]), y.expand(shape=[batch_size, -1, -1, -1])), axis=1)
        if not batched and batch_size == 1:
            return out.squeeze(axis=0)
        else:
            return out


class GridEmbeddingND(paddle.nn.Layer):
    """A positional embedding as a regular ND grid
    """

    def __init__(self, in_channels: int, dim: int=2, grid_boundaries=[[0, 1
        ], [0, 1]]):
        """GridEmbeddingND applies a simple positional 
        embedding as a regular ND grid

        Parameters
        ----------
        in_channels : int
            number of channels in input
        dim : int
            dimensions of positional encoding to apply
        grid_boundaries : list, optional
            coordinate boundaries of input grid along each dim, by default [[0, 1], [0, 1]]
        """
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim
        assert self.dim == len(grid_boundaries
            ), f'Error: expected grid_boundaries to be            an iterable of length {self.dim}, received {grid_boundaries}'
        self.grid_boundaries = grid_boundaries
        self._grid = None
        self._res = None

    @property
    def out_channels(self):
        return self.in_channels + self.dim

    def grid(self, spatial_dims: list, device: str, dtype: paddle.dtype):
        """grid generates ND grid needed for pos encoding
        and caches the grid associated with MRU resolution

        Parameters
        ----------
        spatial_dims : torch.Size
             sizes of spatial resolution
        device : literal 'cpu' or 'cuda:*'
            where to load data
        dtype : str
            dtype to encode data

        Returns
        -------
        torch.tensor
            output grids to concatenate 
        """
        if self._grid is None or self._res != spatial_dims:
            grids_by_dim = regular_grid_nd(spatial_dims, grid_boundaries=
                self.grid_boundaries)
            grids_by_dim = [x.to(device).to(dtype).unsqueeze(axis=0).
                unsqueeze(axis=0) for x in grids_by_dim]
            self._grid = grids_by_dim
            self._res = spatial_dims
        return self._grid

    def forward(self, data, batched=True):
        """
        Params
        --------
        data: torch.Tensor
            assumes shape batch (optional), channels, x_1, x_2, ...x_n
        batched: bool
            whether data has a batch dim
        """
        if not batched:
            if data.ndim == self.dim + 1:
                data = data.unsqueeze(axis=0)
        batch_size = tuple(data.shape)[0]
        grids = self.grid(spatial_dims=tuple(data.shape)[2:], device=data.
            place, dtype=data.dtype)
        grids = [x.tile(repeat_times=[batch_size, *([1] * (self.dim + 1))]) for
            x in grids]
        out = paddle.concat(x=(data, *grids), axis=1)
        return out


class SinusoidalEmbedding(Embedding):
    """
    SinusoidalEmbedding provides a unified sinusoidal positional embedding
    in the styles of Transformers :ref:`[1]` and Neural Radiance Fields (NERFs) :ref:`[2]`.

    Parameters
    ----------
    in_channels : int
        Number of input channels to embed
    num_freqs : int, optional
        Number of frequencies in positional embedding.
        By default, set to the number of input channels
    embedding : {'transformer', 'nerf'}
        Type of embedding to apply. For a function with N input channels, 
        each channel value p is embedded via a function g with 2L channels 
        such that g(p) is a 2L-dim vector. For 0 <= k < L:

        * 'transformer' for transformer-style encoding.

            g(p)_k = sin((p / max_positions) ^ {k / N})

            g(p)_{k+1} = cos((p / max_positions) ^ {k / N})

        * 'nerf' : NERF-style encoding.  

            g(p)_k = sin(2^(k) * Pi * p)

            g(p)_{k+1} = cos(2^(k) * Pi * p)

    max_positions : int, optional
        Maximum number of positions for the encoding, default 10000
        Only used if `embedding == transformer`.

    References
    -----------
    .. _[1]: 

    Vaswani, A. et al (2017)
        "Attention Is All You Need". 
        NeurIPS 2017, https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf. 

    .. _[2]: 
    
    Mildenhall, B. et al (2020)
        "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis".
        ArXiv, https://arxiv.org/pdf/2003.08934. 
    """

    def __init__(self, in_channels: int, num_frequencies: int=None,
        embedding_type: str='transformer', max_positions: int=10000):
        super().__init__()
        self.in_channels = in_channels
        self.num_frequencies = num_frequencies
        allowed_embeddings = ['nerf', 'transformer']
        assert embedding_type in allowed_embeddings, f'Error: embedding_type expected one of {allowed_embeddings}, received {embedding_type}'
        self.embedding_type = embedding_type
        if self.embedding_type == 'transformer':
            assert max_positions is not None, 'Error: max_positions must have an int value for                 transformer embedding.'
        self.max_positions = max_positions

    @property
    def out_channels(self):
        """
        out_channels: required property for linking/composing model layers 
        """
        return 2 * self.num_frequencies * self.in_channels

    def forward(self, x):
        """
        Parameters 
        -----------
        x: torch.Tensor, shape (n_in, self.in_channels) or (batch, n_in, self.in_channels)
        """
        assert x.ndim in [2, 3
            ], f'Error: expected inputs of shape (batch, n_in, {self.in_channels})            or (n_in, channels), got inputs with ndim={x.ndim}, shape={tuple(x.shape)}'
        if x.ndim == 2:
            batched = False
            x = x.unsqueeze(axis=0)
        else:
            batched = True
        batch_size, n_in, _ = tuple(x.shape)
        if self.embedding_type == 'nerf':
            freqs = 2 ** paddle.arange(start=0, end=self.num_frequencies
                ) * numpy.pi
        elif self.embedding_type == 'transformer':
            freqs = paddle.arange(start=0, end=self.num_frequencies
                ) / self.in_channels
            freqs = (1 / self.max_positions) ** freqs
        freqs = paddle.einsum('bij, k -> bijk', x, freqs)
        freqs = paddle.stack(x=(freqs.sin(), freqs.cos()), axis=-1)
        freqs = freqs.view(batch_size, n_in, -1)
        if not batched:
            freqs = freqs.squeeze(axis=0)
        return freqs


class RotaryEmbedding2D(paddle.nn.Layer):

    def __init__(self, dim, min_freq=1 / 64, scale=1.0):
        """
        Applying rotary positional embedding (https://arxiv.org/abs/2104.09864) to the input feature tensor.
        The crux is the dot product of two rotation matrices R(theta1) and R(theta2) is equal to R(theta2 - theta1).
        """
        super().__init__()
        inv_freq = 1.0 / 10000 ** (paddle.arange(start=0, end=dim, step=2).
            astype(dtype='float32') / dim)
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer(name='inv_freq', tensor=inv_freq, persistable=
            False)

    def forward(self, coordinates):
        """coordinates is tensor of [batch_size, num_points]"""
        coordinates = coordinates * (self.scale / self.min_freq)
        freqs = paddle.einsum('... i , j -> ... i j', coordinates, self.
            inv_freq)
        return paddle.concat(x=(freqs, freqs), axis=-1)

    @staticmethod
    def apply_1d_rotary_pos_emb(t, freqs):
        return apply_rotary_pos_emb(t, freqs)

    @staticmethod
    def apply_2d_rotary_pos_emb(t, freqs_x, freqs_y):
        """Split the last dimension of features into two equal halves
           and apply 1d rotary positional embedding to each half."""
        d = tuple(t.shape)[-1]
        t_x, t_y = t[..., :d // 2], t[..., d // 2:]
        return paddle.concat(x=(apply_rotary_pos_emb(t_x, freqs_x),
            apply_rotary_pos_emb(t_y, freqs_y)), axis=-1)


def regular_grid_2d(spatial_dims, grid_boundaries=[[0, 1], [0, 1]]):
    """
    Creates a 2 x height x width stack of positional encodings A, where
    A[:,i,j] = [[x,y]] at coordinate (i,j) on a (height, width) grid. 
    """
    height, width = spatial_dims
    xt = paddle.linspace(start=grid_boundaries[0][0], stop=grid_boundaries[
        0][1], num=height + 1)[:-1]
    yt = paddle.linspace(start=grid_boundaries[1][0], stop=grid_boundaries[
        1][1], num=width + 1)[:-1]
    grid_x, grid_y = paddle.meshgrid(xt, yt)
    grid_x = grid_x.tile(repeat_times=[1, 1])
    grid_y = grid_y.tile(repeat_times=[1, 1])
    return grid_x, grid_y


def regular_grid_nd(resolutions: List[int], grid_boundaries: List[List[int]
    ]=[[0, 1]] * 2):
    """regular_grid_nd generates a tensor of coordinate points that 
    describe a bounded regular grid.
    
    Creates a dim x res_d1 x ... x res_dn stack of positional encodings A, where
    A[:,c1,c2,...] = [[d1,d2,...dn]] at coordinate (c1,c2,...cn) on a (res_d1, ...res_dn) grid. 

    Parameters
    ----------
    resolutions : List[int]
        resolution of the output grid along each dimension
    grid_boundaries : List[List[int]], optional
        List of pairs [start, end] of the boundaries of the
        regular grid. Must correspond 1-to-1 with resolutions default [[0,1], [0,1]]

    Returns
    -------
    grid: tuple(Tensor)
    list of tensors describing positional encoding 
    """
    assert len(resolutions) == len(grid_boundaries
        ), 'Error: inputs must have same number of dimensions'
    dim = len(resolutions)
    meshgrid_inputs = list()
    for res, (start, stop) in zip(resolutions, grid_boundaries):
        meshgrid_inputs.append(paddle.linspace(start=start, stop=stop, num=
            res + 1)[:-1])
    grid = paddle.meshgrid(*meshgrid_inputs)
    grid = tuple([x.tile(repeat_times=[1] * dim) for x in grid])
    return grid


def rotate_half(x):
    """
    Split x's channels into two equal halves.
    """
    x = x.reshape(*tuple(x.shape)[:-1], 2, -1)
    x1, x2 = x.unbind(axis=-2)
    return paddle.concat(x=(-x2, x1), axis=-1)


def apply_rotary_pos_emb(t, freqs):
    """
    Apply rotation matrix computed based on freqs to rotate t.
    t: tensor of shape [batch_size, num_points, dim]
    freqs: tensor of shape [batch_size, num_points, 1]

    Formula: see equation (34) in https://arxiv.org/pdf/2104.09864.pdf
    """
    return t * freqs.cos() + rotate_half(t) * freqs.sin()

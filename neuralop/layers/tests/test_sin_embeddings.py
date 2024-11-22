import paddle
import math
import pytest
from ..embeddings import SinusoidalEmbedding
batch_size = 4
num_freqs = 3
in_channels = 3
n_in = 2
max_pos = 10000
if paddle.device.is_compiled_with_cuda():
    device = 'cuda'
else:
    device = 'cpu'


def test_NeRFEmbedding():
    nerf_embed = SinusoidalEmbedding(in_channels=in_channels,
        num_frequencies=3, embedding_type='nerf')
    unbatched_inputs = paddle.arange(end=in_channels) * paddle.to_tensor(data
        =[[1.0], [0.5]])
    embeds = nerf_embed(unbatched_inputs.to(device))
    true_outputs = paddle.zeros(shape=[n_in, in_channels * num_freqs * 2]).to(
        device)
    for channel in range(in_channels):
        for wavenumber in range(num_freqs):
            for i in range(2):
                idx = channel * (num_freqs * 2) + wavenumber * 2 + i
                freqs = 2 ** wavenumber * math.pi * unbatched_inputs[:, channel
                    ]
                if i == 0:
                    true_outputs[:, idx] = freqs.sin()
                else:
                    true_outputs[:, idx] = freqs.cos()
    assert paddle.allclose(x=embeds, y=true_outputs).item(), ''
    batched_inputs = paddle.stack(x=[paddle.arange(end=in_channels) *
        paddle.to_tensor(data=[[1.0], [0.5]])] * batch_size)
    embeds = nerf_embed(batched_inputs.to(device))
    true_outputs = paddle.zeros(shape=[batch_size, n_in, in_channels *
        num_freqs * 2]).to(device)
    for channel in range(in_channels):
        for wavenumber in range(num_freqs):
            for i in range(2):
                idx = channel * (num_freqs * 2) + wavenumber * 2 + i
                freqs = 2 ** wavenumber * math.pi * batched_inputs[:, :,
                    channel]
                if i == 0:
                    true_outputs[:, :, idx] = freqs.sin()
                else:
                    true_outputs[:, :, idx] = freqs.cos()
    assert paddle.allclose(x=embeds, y=true_outputs).item(), ''


def test_TransformerEmbedding():
    nerf_embed = SinusoidalEmbedding(in_channels=in_channels,
        num_frequencies=3, embedding_type='transformer', max_positions=max_pos)
    unbatched_inputs = paddle.arange(end=in_channels) * paddle.to_tensor(data
        =[[1.0], [0.5]])
    embeds = nerf_embed(unbatched_inputs.to(device))
    true_outputs = paddle.zeros(shape=[n_in, in_channels * num_freqs * 2]).to(
        device)
    for channel in range(in_channels):
        for wavenumber in range(num_freqs):
            for i in range(2):
                idx = channel * (num_freqs * 2) + wavenumber * 2 + i
                freqs = (1 / max_pos) ** (wavenumber / in_channels
                    ) * unbatched_inputs[:, channel]
                if i == 0:
                    true_outputs[:, idx] = freqs.sin()
                else:
                    true_outputs[:, idx] = freqs.cos()
    assert paddle.allclose(x=embeds, y=true_outputs).item(), ''
    batched_inputs = paddle.stack(x=[paddle.arange(end=in_channels) *
        paddle.to_tensor(data=[[1.0], [0.5]])] * batch_size)
    embeds = nerf_embed(batched_inputs.to(device))
    true_outputs = paddle.zeros(shape=[batch_size, n_in, in_channels *
        num_freqs * 2]).to(device)
    for channel in range(in_channels):
        for wavenumber in range(num_freqs):
            for i in range(2):
                idx = channel * (num_freqs * 2) + wavenumber * 2 + i
                freqs = (1 / max_pos) ** (wavenumber / in_channels
                    ) * batched_inputs[:, :, channel]
                if i == 0:
                    true_outputs[:, :, idx] = freqs.sin()
                else:
                    true_outputs[:, :, idx] = freqs.cos()
    assert paddle.allclose(x=embeds, y=true_outputs).item(), ''

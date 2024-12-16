import paddle
import pytest
from tensorly import tenalg

from ..gino import GINO

tenalg.set_backend("einsum")

# Fixed variables
in_channels = 3
out_channels = 2
projection_channels = 16
lifting_channels = 16
fno_n_modes = 8, 8, 8
# data parameters
n_in = 100
n_out = 100
latent_density = 8
fno_ada_in_dim = 1
fno_ada_in_features = 4


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("gno_coord_dim", [2, 3])
@pytest.mark.parametrize("gno_coord_embed_dim", [None, 32])
@pytest.mark.parametrize("fno_norm", [None, "ada_in"])
@pytest.mark.parametrize("gno_transform_type", ["linear", "nonlinear_kernelonly", "nonlinear"])
def test_gino(gno_transform_type, gno_coord_dim, gno_coord_embed_dim, batch_size, fno_norm):
    if paddle.device.cuda.device_count() >= 1:
        device = paddle.CUDAPlace(0)
    else:
        device = paddle.CPUPlace()
    model = GINO(
        in_channels=in_channels,
        out_channels=out_channels,
        gno_radius=0.3,  # make this large to ensure neighborhoods fit
        projection_channels=projection_channels,
        gno_coord_dim=gno_coord_dim,
        gno_coord_embed_dim=gno_coord_embed_dim,
        in_gno_mlp_hidden_layers=[16, 16],
        out_gno_mlp_hidden_layers=[16, 16],
        in_gno_transform_type=gno_transform_type,
        out_gno_transform_type=gno_transform_type,
        fno_n_modes=fno_n_modes[:gno_coord_dim],
        fno_norm=fno_norm,
        fno_ada_in_dim=fno_ada_in_dim,
        fno_ada_in_features=fno_ada_in_features,
        # keep the FNO model small for runtime
        fno_lifting_channels=lifting_channels,
    ).to(device)

    # create grid of latent queries on the unit cube
    latent_geom = paddle.stack(
        x=list(
            [
                i.T
                for i in paddle.meshgrid(
                    [paddle.linspace(start=0, stop=1, num=latent_density)] * gno_coord_dim
                )
            ]
        )
    )
    latent_geom = latent_geom.transpose(perm=[*list(range(1, gno_coord_dim + 1)), 0]).to(device)
    # create input geometry and output queries
    input_geom_shape = [n_in, gno_coord_dim]
    input_geom = paddle.randn(shape=input_geom_shape)
    output_queries_shape = [n_out, gno_coord_dim]
    output_queries = paddle.randn(shape=output_queries_shape)
    # create data and features
    x_shape = [batch_size, n_in, in_channels]
    x = paddle.randn(shape=x_shape)
    out_0 = x
    # require and retain grad to check for backprop
    out_0.stop_gradient = not True
    # out_0
    ada_in = paddle.randn(shape=[1])

    # Test forward pass
    out = model(
        x=x,
        input_geom=input_geom,
        latent_queries=latent_geom,
        output_queries=output_queries,
        ada_in=ada_in,
    )

    # Check output size
    assert list(tuple(out.shape)) == [batch_size, n_out, out_channels]

    # Check backward pass
    assert out.isfinite().astype("bool").all()
    if batch_size > 1:
        loss = out[0].sum()
    else:
        loss = out.sum()
    loss.backward()
    n_unused_params = 0
    for param in model.parameters():
        if param.grad is None:
            n_unused_params += 1
    assert n_unused_params == 0, f"{n_unused_params} parameters were unused!"
    if batch_size > 1:
        # assert f[1:] accumulates no grad
        if not x.grad[1:].nonzero().is_empty():
            assert not x.grad[1:].nonzero().astype("bool").any()

import paddle

from neuralop.tests.test_utils import DummyModel

from ..data_processors import DefaultDataProcessor
from ..data_processors import IncrementalDataProcessor
from ..normalizers import UnitGaussianNormalizer


def test_DefaultDataProcessor_pipeline():
    if paddle.device.is_compiled_with_cuda():
        device = paddle.CUDAPlace(0)
    else:
        device = paddle.CPUPlace()
    x = paddle.randn(shape=(1, 2, 64, 64))
    y = paddle.randn(shape=(1, 2, 64, 64))
    normalizer = UnitGaussianNormalizer(
        mean=paddle.zeros(shape=(1, 2, 1, 1)),
        std=paddle.ones(shape=(1, 2, 1, 1)),
        eps=1e-05,
    )
    pipeline = DefaultDataProcessor(in_normalizer=normalizer, out_normalizer=normalizer)
    data = {"x": x, "y": y}  # data on cpu at this point
    xform_data = pipeline.preprocess(data)
    # model outputs will be on device by default
    out = paddle.randn(shape=(1, 2, 64, 64)).to(device)
    _, inv_xform_data = pipeline.postprocess(out, xform_data)
    assert paddle.allclose(x=inv_xform_data["y"].cpu(), y=data["y"]).item(), ""


def test_DefaultDataProcessor_train_eval():
    if paddle.device.is_compiled_with_cuda():
        device = paddle.CUDAPlace(0)
    else:
        device = paddle.CPUPlace()
    model = DummyModel(features=10)
    normalizer = UnitGaussianNormalizer(
        mean=paddle.zeros(shape=(1, 2, 1, 1)),
        std=paddle.ones(shape=(1, 2, 1, 1)),
        eps=1e-05,
    )
    pipeline = DefaultDataProcessor(in_normalizer=normalizer, out_normalizer=normalizer)
    wrapped_model = pipeline.wrap(model).to(device)
    assert wrapped_model.device == device
    wrapped_model.train()
    assert wrapped_model.training
    assert wrapped_model.model.training

    wrapped_model.eval()
    assert not wrapped_model.training
    assert not wrapped_model.model.training


# ensure that the data processor incrementally increases the resolution
def test_incremental_resolution():
    if paddle.device.is_compiled_with_cuda():
        device = paddle.CUDAPlace(0)
    else:
        device = paddle.CPUPlace()
    x = paddle.randn(shape=(1, 2, 16, 16)).to(device)
    y = paddle.randn(shape=(1, 2, 16, 16)).to(device)
    indice_list = [2, 3]
    data_transform = IncrementalDataProcessor(
        in_normalizer=None,
        out_normalizer=None,
        device=device,
        subsampling_rates=[2],
        dataset_resolution=16,
        dataset_indices=indice_list,
        epoch_gap=10,
        verbose=True,
    )
    # model outputs will be on device by default
    x_new, y_new = data_transform.regularize_input_res(x, y)

    # This is cause x_new.shape = (1, 2, 8, 8)
    # This is cause y_new.shape = (1, 2, 8, 8)
    for i in indice_list:
        assert tuple(x_new.shape)[i] < tuple(x.shape)[i]
        assert tuple(y_new.shape)[i] < tuple(y.shape)[i]

import paddle
from ..normalizers import UnitGaussianNormalizer
from flaky import flaky


@flaky(max_runs=4, min_passes=3)
def test_UnitGaussianNormalizer_created_from_stats(eps=1e-06):
    x = paddle.rand(shape=[16, 3, 40, 50, 60]) * 2.5
    mean = paddle.mean(x=x, axis=[0, 2, 3, 4], keepdim=True)
    std = paddle.std(x=x, axis=[0, 2, 3, 4], keepdim=True)
    normalizer = UnitGaussianNormalizer(mean=mean, std=std, eps=eps)
    x_normalized = normalizer.transform(x)
    x_unnormalized = normalizer.inverse_transform(x_normalized)
    assert paddle.allclose(x=x_unnormalized, y=x).item(), ''
    assert paddle.mean(x=x_normalized) <= eps
    assert paddle.std(x=x_normalized) - 1 <= eps


@flaky(max_runs=4, min_passes=3)
def test_UnitGaussianNormalizer_from_data(eps=1e-06):
    x = paddle.rand(shape=[16, 3, 40, 50, 60]) * 2.5
    mean = paddle.mean(x=x, axis=[0, 2, 3, 4], keepdim=True)
    std = paddle.std(x=x, axis=[0, 2, 3, 4], keepdim=True)
    normalizer = UnitGaussianNormalizer(dim=[0, 2, 3, 4], eps=eps)
    normalizer.fit(x)
    assert paddle.allclose(x=normalizer.mean, y=mean).item(), ''
    assert paddle.allclose(rtol=eps, atol=eps, x=normalizer.std, y=std).item(
        ), ''
    x_normalized = normalizer.transform(x)
    x_unnormalized = normalizer.inverse_transform(x_normalized)
    assert paddle.allclose(x=x_unnormalized, y=x).item(), ''
    assert paddle.mean(x=x_normalized) <= eps
    assert paddle.std(x=x_normalized) - 1 <= eps
    assert paddle.allclose(x=normalizer.mean, y=mean).item(), ''
    assert paddle.allclose(rtol=eps, atol=eps, x=normalizer.std, y=std).item(
        ), ''


@flaky(max_runs=4, min_passes=3)
def test_UnitGaussianNormalizer_incremental_update(eps=1e-06):
    x = paddle.rand(shape=[16, 3, 40, 50, 60]) * 2.5
    mean = paddle.mean(x=x, axis=[0, 2, 3, 4], keepdim=True)
    std = paddle.std(x=x, axis=[0, 2, 3, 4], keepdim=True)
    normalizer = UnitGaussianNormalizer(dim=[0, 2, 3, 4], eps=eps)
    normalizer.partial_fit(x, batch_size=2)
    x_normalized = normalizer.transform(x)
    x_unnormalized = normalizer.inverse_transform(x_normalized)
    assert paddle.allclose(x=x_unnormalized, y=x).item(), ''
    assert paddle.mean(x=x_normalized) <= eps
    assert paddle.std(x=x_normalized) - 1 <= eps
    assert paddle.allclose(x=normalizer.mean, y=mean).item(), ''
    assert paddle.allclose(rtol=eps, atol=eps, x=normalizer.std, y=std).item(
        ), ''

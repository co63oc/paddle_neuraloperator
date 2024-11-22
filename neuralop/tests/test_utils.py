import os
import paddle
from ..utils import get_wandb_api_key, wandb_login
from ..utils import count_model_params, count_tensor_params
from pathlib import Path
import pytest
wandb_available = False
try:
    import wandb
    wandb_available = True
except ModuleNotFoundError:
    wandb_available = False
from neuralop.models.base_model import BaseModel


def test_count_model_params():


    class DumyModel(paddle.nn.Layer):

        def __init__(self, n_submodels=0, dtype='float32'):
            super().__init__()
            self.n_submodels = n_submodels
            self.param = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.randn(shape=(2, 3, 4), dtype=dtype))
            if n_submodels:
                self.model = DumyModel(n_submodels - 1, dtype=dtype)
    n_submodels = 2
    model = DumyModel(n_submodels=n_submodels)
    n_params = count_model_params(model)
    print(model)
    assert n_params == (n_submodels + 1) * 2 * 3 * 4
    model = DumyModel(n_submodels=n_submodels, dtype='complex64')
    n_params = count_model_params(model)
    print(model)
    assert n_params == 2 * (n_submodels + 1) * 2 * 3 * 4


def test_count_tensor_params():
    x = paddle.randn(shape=(2, 3, 4, 5, 6), dtype='float32')
    n_params = count_tensor_params(x)
    assert n_params == 2 * 3 * 4 * 5 * 6
    n_params = count_tensor_params(x, dims=[1, 3])
    assert n_params == 3 * 5
    x = paddle.randn(shape=(2, 3, 4, 5, 6), dtype='complex64')
    n_params = count_tensor_params(x)
    assert n_params == 2 * 3 * 4 * 5 * 6 * 2
    n_params = count_tensor_params(x, dims=[1, 3])
    assert n_params == 3 * 5 * 2


def test_get_wandb_api_key():
    os.environ.pop('WANDB_API_KEY', None)
    filepath = Path(__file__).parent.joinpath('test_config_key.txt').as_posix()
    key = get_wandb_api_key(filepath)
    assert key == 'my_secret_key'
    os.environ['WANDB_API_KEY'] = 'key_from_env'
    key = get_wandb_api_key(filepath)
    assert key == 'key_from_env'
    os.environ['WANDB_API_KEY'] = 'key_from_env'
    key = get_wandb_api_key('wrong_path')
    assert key == 'key_from_env'


def test_ArgparseConfig(monkeypatch):
    if wandb_available:

        def login(key):
            if key == 'my_secret_key':
                return True
            raise ValueError('Wrong key')
        monkeypatch.setattr(wandb, 'login', login)
        os.environ.pop('WANDB_API_KEY', None)
        filepath = Path(__file__).parent.joinpath('test_config_key.txt'
            ).as_posix()
        assert wandb_login(filepath) is None
        os.environ['WANDB_API_KEY'] = 'my_secret_key'
        assert wandb_login() is None
        os.environ['WANDB_API_KEY'] = 'wrong_key'
        assert wandb_login(key='my_secret_key') is None
        os.environ['WANDB_API_KEY'] = 'wrong_key'
        with pytest.raises(ValueError):
            wandb_login()


class DummyDataset(paddle.io.Dataset):

    def __init__(self, n_examples):
        super().__init__()
        self.X = paddle.randn(shape=(n_examples, 50))
        self.y = paddle.randn(shape=(n_examples, 1))

    def __getitem__(self, idx):
        return {'x': self.X[idx], 'y': self.y[idx]}

    def __len__(self):
        return tuple(self.X.shape)[0]


class DummyModel(BaseModel, name='Dummy'):
    """
    Simple linear model to mock-up our model API
    """

    def __init__(self, features, **kwargs):
        super().__init__()
        self.net = paddle.nn.Linear(in_features=features, out_features=1)

    def forward(self, x, **kwargs):
        """
        Throw out extra args as in FNO and other models
        """
        return self.net(x)

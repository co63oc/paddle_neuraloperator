import time
from pathlib import Path

import paddle
from configmypy import ConfigPipeline
from configmypy import YamlConfig
from tensorly import tenalg

from neuralop import get_model

tenalg.set_backend("einsum")


def test_from_config():
    """Test forward/backward from a config file"""
    config_name = "default"
    config_path = Path(__file__).parent.as_posix()
    pipe = ConfigPipeline(
        [YamlConfig("./test_config.yaml", config_name=config_name, config_folder=config_path)]
    )
    config = pipe.read_conf()
    config_name = pipe.steps[-1].config_name
    batch_size = config.data.batch_size
    size = config.data.size
    if paddle.device.cuda.device_count() >= 1:
        device = paddle.CUDAPlace(0)
    else:
        device = paddle.CPUPlace()
    model = get_model(config)
    model = model.to(device)
    in_data = paddle.randn(shape=[batch_size, 3, size, size]).to(device)
    print(model.__class__)
    print(model)
    t1 = time.time()
    out = model(in_data)
    t = time.time() - t1
    print(f"Output of size {tuple(out.shape)} in {t}.")
    loss = out.sum()
    loss.backward()

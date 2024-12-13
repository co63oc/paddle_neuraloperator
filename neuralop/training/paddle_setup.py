import paddle

import neuralop.mpu.comm as comm


def setup(config):
    """A convenience function to intialize the device, setup paddle settings and
    check multi-grid and other values. It sets up distributed communitation, if used.

    Parameters
    ----------
    config : dict
        this function checks:
        * config.distributed (use_distributed, seed)
        * config.data (n_train, batch_size, test_batch_sizes, n_tests, test_resolutions)

    Returns
    -------
    device, is_logger
        device : paddle.device
        is_logger : bool
    """
    if config.distributed.use_distributed:
        comm.init(config, verbose=config.verbose)
        # Set process 0 to log screen and wandb
        is_logger = comm.get_world_rank() == 0
        # Set device and random seed
        device = paddle.CUDAPlace(int(f"cuda:{comm.get_local_rank()}".replace("cuda:", "")))
        seed = config.distributed.seed + comm.get_data_parallel_rank()
        # Ensure every iteration has the same amount of data
        assert (
            config.data.n_train % config.data.batch_size == 0
        ), f"The number of training samples={config.data.n_train} cannot be divided by the batch_size={config.data.batch_size}."
        for j in range(len(config.data.test_batch_sizes)):
            assert (
                config.data.n_tests[j] % config.data.test_batch_sizes[j] == 0
            ), f"The number of training samples={config.data.n_tests[j]} cannot be divided by the batch_size={config.data.test_batch_sizes[j]} for test resolution {config.data.test_resolutions[j]}."
        # Ensure batch can be evenly split among the data-parallel group
        # NOTE: Distributed sampler NOT implemented: set model_parallel_size = # of GPUS
        assert (
            config.data.batch_size % comm.get_data_parallel_size() == 0
        ), f"Batch of size {config.data.batch_size} can be evenly split among the data-parallel group={comm.get_data_parallel_size()}."
        config.data.batch_size = config.data.batch_size // comm.get_data_parallel_size()
        # Ensure batch can be evenly split among the model-parallel group
        if config.patching.levels > 0:
            assert (
                config.data.batch_size
                * 2 ** (2 * config.patching.levels)
                % comm.get_model_parallel_size()
                == 0
            ), f"With MG patching, total batch-size of {config.data.batch_size * 2 ** (2 * config.patching.levels)} ({config.data.batch_size} times {2 ** (2 * config.patching.levels)}). However, this total batch-size cannot be evenly split among the {comm.get_model_parallel_size()} model-parallel groups."
            for b_size in config.data.test_batch_sizes:
                assert (
                    b_size * 2 ** (2 * config.patching.levels) % comm.get_model_parallel_size() == 0
                ), f"With MG patching, for test resolution of {config.data.test_resolutions[j]} the total batch-size is {config.data.batch_size * 2 ** (2 * config.patching.levels)} ({config.data.batch_size} times {2 ** (2 * config.patching.levels)}). However, this total batch-size cannot be evenly split among the {comm.get_model_parallel_size()} model-parallel groups."
    else:
        is_logger = True
        if paddle.device.cuda.device_count() >= 1:
            device = paddle.CUDAPlace(int("cuda:0".replace("cuda:", "")))
        else:
            device = paddle.CPUPlace()
        if "seed" in config.distributed:
            seed = config.distributed.seed
    # Set device, random seed and optimization
    if paddle.device.cuda.device_count() >= 1:
        paddle.device.set_device(
            device="gpu:" + str(device.index)
            if isinstance(device.index, int)
            else str(device.index).replace("cuda", "gpu")
        )
        if "seed" in config.distributed:
            paddle.seed(seed=seed)
        increase_l2_fetch_granularity()
        # try:
        #     torch.set_float32_matmul_precision('high')
        # except AttributeError:
        #     pass

    if "seed" in config.distributed:
        paddle.seed(seed=seed)
    return device, is_logger


def increase_l2_fetch_granularity():
    try:
        import ctypes

        _libcudart = ctypes.CDLL("libcudart.so")
        # Set device limit on the current device
        # cudaLimitMaxL2FetchGranularity = 0x05
        pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
        _libcudart.cudaDeviceSetLimit(ctypes.c_int(5), ctypes.c_int(128))
        _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(5))
        assert pValue.contents.value == 128
    except:  # noqa
        return
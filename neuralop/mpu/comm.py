import os
import paddle
import logging
import datetime as dt


class disable_logging(object):

    def __init__(self, level=logging.ERROR):
        logging.disable(level=level)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        logging.disable(level=logging.NOTSET)


_DATA_PARALLEL_GROUP = None
_MODEL_PARALLEL_GROUP = None


def get_world_size():
    if not paddle.distributed.is_initialized():
        return 1
    else:
        return paddle.distributed.get_world_size()


def get_world_rank():
    if not paddle.distributed.is_initialized():
        return 0
    else:
        return paddle.distributed.get_rank()


def get_local_rank():
    if not paddle.distributed.is_initialized():
        return 0
    else:
        return get_world_rank() % paddle.device.cuda.device_count()


def get_data_parallel_size():
    if not paddle.distributed.is_initialized():
        return 1
    else:
>>>>>>        return torch.distributed.get_world_size(group=_DATA_PARALLEL_GROUP)


def get_data_parallel_rank():
    if not paddle.distributed.is_initialized():
        return 0
    else:
        return paddle.distributed.get_rank(group=_DATA_PARALLEL_GROUP)


def get_data_parallel_group():
    assert paddle.distributed.is_initialized(
        ), 'Error, initialize torch.distributed first'
    return _DATA_PARALLEL_GROUP


def get_model_parallel_size():
    if not paddle.distributed.is_initialized(
        ) or _MODEL_PARALLEL_GROUP is None:
        return 1
    else:
>>>>>>        return torch.distributed.get_world_size(group=_MODEL_PARALLEL_GROUP)


def get_model_parallel_rank():
    if not paddle.distributed.is_initialized(
        ) or _MODEL_PARALLEL_GROUP is None:
        return 0
    else:
        return paddle.distributed.get_rank(group=_MODEL_PARALLEL_GROUP)


def get_model_parallel_group():
    assert paddle.distributed.is_initialized(
        ), 'Error, initialize torch.distributed first'
    return _MODEL_PARALLEL_GROUP


def init(config, verbose=False):
    if config.distributed == 'env':
        world_size = int(os.getenv('WORLD_SIZE', 1))
        world_rank = int(os.getenv('WORLD_RANK', 0))
        port = int(os.getenv('MASTER_PORT', 0))
        master_address = os.getenv('MASTER_ADDRESS')
    elif config.distributed.wireup_info == 'mpi':
        import socket
        from mpi4py import MPI
        mpi_comm = MPI.COMM_WORLD.Dup()
        world_size = mpi_comm.Get_size()
        world_rank = mpi_comm.Get_rank()
        my_host = '127.0.0.1'
        port = 29500
        master_address = mpi_comm.bcast(my_host, root=0)
        os.environ['MASTER_ADDRESS'] = master_address
        os.environ['MASTER_PORT'] = str(port)
    else:
        raise ValueError(
            f'Error, wireup-info {config.distributed.wireup_info} not supported'
            )
    local_rank = 0
    if world_size > 1:
        with disable_logging():
            if config.distributed.wireup_store == 'file':
                wireup_file_path = os.getenv('WIREUP_FILE_PATH')
>>>>>>                wireup_store = torch.distributed.FileStore(wireup_file_path,
                    world_size)
            elif config.distributed.wireup_store == 'tcp':
>>>>>>                wireup_store = torch.distributed.TCPStore(host_name=
                    master_address, port=port, world_size=world_size,
                    is_master=world_rank == 0, timeout=dt.timedelta(seconds
                    =900))
            paddle.distributed.init_parallel_env()
            world_size = get_world_size()
            world_rank = get_world_rank()
            local_rank = get_local_rank()
>>>>>>            torch.distributed.barrier(device_ids=[local_rank])
    is_logger = get_world_rank() == 0
    model_group_size = config.distributed.model_parallel_size
    data_group_size = world_size // model_group_size
    if is_logger:
        print(
            f'Using {world_size} in {model_group_size} x {data_group_size} decomposition (#model-ranks x #data-ranks)'
            )
    assert model_group_size <= world_size and world_size % model_group_size == 0, 'Error, please make sure matmul_parallel_size * spatial_parallel_size <= world size and that world size is evenly divisible by matmul_parallel_size * spatial_parallel_size'
    num_model_groups = world_size // model_group_size
    global _DATA_PARALLEL_GROUP
    global _MODEL_PARALLEL_GROUP
    if is_logger:
        print('Starting Wireup')
    if world_size > 1:
        if model_group_size > 1:
            model_groups = []
            for i in range(num_model_groups):
                start = i * model_group_size
                end = start + model_group_size
                model_groups.append(list(range(start, end)))
            data_groups = [sorted(list(i)) for i in zip(*model_groups)]
            if verbose and is_logger:
                print('Model Parallel Groups w/ respect to world rank:')
                for grp in model_groups:
                    print(grp)
            if verbose and is_logger:
                print('Data Parallel Groups w/ respect to world rank:')
                for grp in data_groups:
                    print(grp)
            with disable_logging():
                for grp in data_groups:
                    tmp_group = paddle.distributed.new_group(ranks=grp)
                    if world_rank in grp:
                        _DATA_PARALLEL_GROUP = tmp_group
                for grp in model_groups:
                    tmp_group = paddle.distributed.new_group(ranks=grp)
                    if world_rank in grp:
                        _MODEL_PARALLEL_GROUP = tmp_group
        else:
            with disable_logging():
                _MODEL_PARALLEL_GROUP = paddle.distributed.new_group(ranks=
                    [world_rank])
                _SPATIAL_PARALLEL_GROUP = _MODEL_PARALLEL_GROUP
                _MATMUL_PARALLEL_GROUP = _MODEL_PARALLEL_GROUP
                _DATA_PARALLEL_GROUP = paddle.distributed.new_group(ranks=
                    list(range(world_size)))
    if paddle.distributed.is_initialized():
>>>>>>        torch.distributed.barrier(device_ids=[local_rank])
    if is_logger:
        print('Finished Wireup')
    return

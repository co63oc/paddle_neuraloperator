import paddle
"""
Snippet to load all artifacts of training state as Modules
without constraining to use inside a default Trainer
"""
from typing import Union
from pathlib import Path


def load_training_state(save_dir: Union[str, Path], save_name: str, model:
    paddle.nn.Layer, optimizer: paddle.nn.Layer=None, scheduler: paddle.nn.
    Layer=None, regularizer: paddle.nn.Layer=None, map_location: dict=None
    ) ->dict:
    """load_training_state returns model and optional other training modules
    saved from prior training for downstream use

    Parameters
    ----------
    save_dir : Union[str, Path]
        directory from which to load training state (model, optional optimizer, scheduler, regularizer)
    save_name : str
        name of model to load
    model : nn.Module
        model to save
    optimizer : nn.Module, optional
        optimizer object to save, by default None
    scheduler : nn.Module, optional
        scheduler object to save, by default None
    regularizer : nn.Module, optional
        regularizer object to save, by default None
    map_location : dict, optional
        mapping dictionary keyed `{device_from: device_to}`, by default None
        dictionary instructs torch to load a model from a checkpoint on rank `device_from`
        and send it to `device_to`

    Returns
    -------
    dict of training state
        keyed `{'model': model, etc}`
        
    """
    if not map_location:
        if paddle.distributed.is_initialized():
            map_location = {'cuda:0': f'cuda:{paddle.distributed.get_rank}'}
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    model = model.from_checkpoint(save_dir, save_name, map_location=
        map_location)
    if optimizer is not None:
        optimizer_pth = save_dir / 'optimizer.pt'
        if optimizer_pth.exists():
            optimizer.set_state_dict(state_dict=paddle.load(path=str(
                optimizer_pth)))
        else:
            print(
                f'Warning: requested to load optimizer state, but no saved optimizer state exists in {save_dir}.'
                )
    if scheduler is not None:
        scheduler_pth = save_dir / 'scheduler.pt'
        if scheduler_pth.exists():
            scheduler.set_state_dict(state_dict=paddle.load(path=str(
                scheduler_pth)))
        else:
            print(
                f'Warning: requested to load scheduler state, but no saved scheduler state exists in {save_dir}.'
                )
    if regularizer is not None:
        regularizer_pth = save_dir / 'regularizer.pt'
        if regularizer_pth.exists():
            regularizer.set_state_dict(state_dict=paddle.load(path=str(
                regularizer_pth)))
        else:
            print(
                f'Warning: requested to load regularizer state, but no saved regularizer state exists in {save_dir}.'
                )
    return model


def save_training_state(save_dir: Union[str, Path], save_name: str, model:
    paddle.nn.Layer, optimizer: paddle.nn.Layer=None, scheduler: paddle.nn.
    Layer=None, regularizer: paddle.nn.Layer=None) ->None:
    """save_training_state returns model and optional other training modules
    saved from prior training for downstream use

    Parameters
    ----------
    save_dir : Union[str, Path]
        directory from which to load training state (model, optional optimizer, scheduler, regularizer)
    save_name : str
        name of model to load
    """
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    model.save_checkpoint(save_dir, save_name)
    if optimizer is not None:
        optimizer_pth = save_dir / 'optimizer.pt'
        paddle.save(obj=optimizer.state_dict(), path=optimizer_pth)
    if scheduler is not None:
        scheduler_pth = save_dir / 'scheduler.pt'
        paddle.save(obj=scheduler.state_dict(), path=scheduler_pth)
    if regularizer is not None:
        regularizer_pth = save_dir / 'regularizer.pt'
        paddle.save(obj=regularizer.state_dict(), path=regularizer_pth)
    print(f'Successfully saved training state to {save_dir}')

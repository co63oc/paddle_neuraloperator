# only import MeshDataModule if open3d is built locally
from importlib.util import find_spec

from .burgers import load_burgers_1dtime  # noqa
from .darcy import load_darcy_flow_small  # noqa
from .darcy import load_darcy_pt  # noqa
from .dict_dataset import DictDataset  # noqa
from .navier_stokes import load_navier_stokes_pt  # noqa
from .spherical_swe import load_spherical_swe  # noqa

if find_spec("open3d") is not None:
    from .mesh_datamodule import MeshDataModule  # noqa

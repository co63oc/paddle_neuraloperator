from .burgers import load_burgers_1dtime  # noqa
from .darcy import DarcyDataset  # noqa
from .darcy import load_darcy_flow_small  # noqa
from .dict_dataset import DictDataset  # noqa
from .navier_stokes import NavierStokesDataset  # noqa
from .navier_stokes import load_navier_stokes_pt  # noqa
from .pt_dataset import PTDataset  # noqa

# only import MeshDataModule if open3d is built locally
try:
    from .mesh_datamodule import MeshDataModule  # noqa
except ModuleNotFoundError:
    pass

# only import SphericalSWEDataset if paddle_harmonics is built locally
try:
    from .spherical_swe import load_spherical_swe  # noqa
except ModuleNotFoundError:
    pass

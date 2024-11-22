from .darcy import DarcyDataset, load_darcy_flow_small
from .navier_stokes import NavierStokesDataset, load_navier_stokes_pt
from .pt_dataset import PTDataset
from .burgers import load_burgers_1dtime
from .dict_dataset import DictDataset
try:
    from .mesh_datamodule import MeshDataModule
except ModuleNotFoundError:
    pass
try:
    from .spherical_swe import load_spherical_swe
except ModuleNotFoundError:
    pass

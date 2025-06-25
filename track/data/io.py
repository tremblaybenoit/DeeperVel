from typing import Union, List
import os
import xarray as xr
import numpy as np
from multiprocessing import Pool
import itertools
from track.utilities.logic import get_list
from tqdm import tqdm
import time
import tracemalloc


def read_xarray(filename, engine="scipy", combine="by_coords"):
    """ Reads a file using xarray.

    Parameters
    ----------
    filename: str. Path to file.
    engine: str. Engine to use for reading the NetCDF file.
    combine: str. Combine method.

    Returns
    -------
    data: xarray.Dataset. File as an xarray dataset
    """

    return xr.open_mfdataset(filename, engine=engine, combine=combine)


def read_netcdf(filename, engine="h5netcdf"):
    """ Reads a NetCDF file using xarray.

    Parameters
    ----------
    filename: str. Path to netCDF file.
    engine : str. Engine to use for reading the NetCDF file.

    Returns
    -------
    data: xarray.Dataset. NetCDF file as an xarray.
    """

    return read_xarray(filename, engine=engine)


class FitsDataset:
    """ Fits dataset class. """
    def __init__(self, path: Union[str, List[str]]):
        """
        Initialize the Fits dataset.

        Parameters
        ----------
        path: str. Path to the dataset.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()


class MURaMQSDataset:
    """ MURaMQS reader class."""

    def __init__(self, path: str, dataset: str='yz') -> None:
        """ Initialize MURaMQSDataset object.

        Parameters
        ----------
        path: str. Path to the dataset.
        dataset: str. Type of slice to read.

        Returns
        -------
        None.
        """
        super().__init__()

        # Path
        self.path = path

        # Baseline for 2D slices
        base_slice = {'nx': 1536, 'ny': 1536, 'nz': 1,
                      'dx': 16., 'dy': 16., 'dz': 16.,
                      'dstep': 50, 'dt': 10, 'nt': 361,
                      'iter_start': 0, 'iter_end': 18000}
        # Dataset configurations
        self.datasets = {
            'tau': {**base_slice,
                    'file_pattern': f'{path}/tau_slice_{{slice:05.3f}}.{{iter:06d}}',
                    'slices': [1, 1.e-1, 1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6],
                    },
            'yz': {**base_slice,
                   'file_pattern': f'{path}/yz_slice_{{slice:04d}}.{{iter:06d}}',
                   'slices': [0, 192, 384, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500],
                   },
            'xz': {**base_slice,
                   'file_pattern': f'{path}/xz_slice_{{slice:04d}}.{{iter:06d}}',
                   'slices': [0],
                   },
            'xy': {**base_slice,
                   'file_pattern': f'{path}/xy_slice_{{slice:04d}}.{{iter:06d}}',
                   'slices': [0],
                   },
            'cube': {'file_pattern': f'{path}/cube.{{iter:06d}}',
                     'nx': 1536, 'ny': 1536, 'nz': 512,
                     'dstep': 150, 'dt': 30, 'nt': 120,
                     'dx': 16., 'dy': 16., 'dz': 16.,
                     'iter_start': 0, 'iter_end': 18000,
                     },
        }

        # Slice type
        if dataset not in self.datasets.keys():
            raise ValueError(f"Dataset {dataset} not recognized.")
        else:
            self.dataset = self.datasets[dataset]
            self.dx = self.dataset['dx']
            self.dy = self.dataset['dy']
            self.dz = self.dataset['dz']
            self.nx = int(self.dataset['nx'])
            self.ny = int(self.dataset['ny'])
            self.nz = int(self.dataset['nz'])
            self.dstep = int(self.dataset['dstep'])
            self.dt = self.dataset['dt']
            self.nt = int(self.dataset['nt'])
            self.t = [t for t in range(self.nt)]
            self.iter_start = int(self.dataset['iter_start'])
            self.iter_end = int(self.dataset['iter_end'])
            self.iter = [i for i in range(self.iter_start, self.iter_end + 1, self.dstep)]
            self.slices = self.dataset['slices']

        # Variables meta data
        self.vars  = {
            'I500': {'file_pattern': f"{path}/I_out.{{iter:06d}}",
                     'dtype': np.float32,
                     'index': 0,
                     'units': {'scaling': 1.,
                               'label': r'[units]'},
                     },
            'rho' : {'file_pattern': self.dataset['file_pattern'],
                     'dtype': np.float32,
                     'index': 0,
                     'units': {'scaling': 1.,
                               'label': r'[units]'},
                    },
            'vx'  : {'file_pattern': self.dataset['file_pattern'],
                     'dtype': np.float32,
                     'index': 1,
                     'units': {'scaling': 1.e-5,
                               'label': r'[km s$^{-1}$]'},
                    },
            'vy'  : {'file_pattern': self.dataset['file_pattern'],
                     'dtype': np.float32,
                     'index': 2,
                     'units': {'scaling': 1.e-5,
                               'label': r'[km s$^{-1}$]'},
                    },
            'vz'  : {'file_pattern': self.dataset['file_pattern'],
                     'dtype': np.float32,
                     'index': 3,
                     'units': {'scaling': 1.e-5,
                               'label': r'[km s$^{-1}$]'},
                    },
            'eint': {'file_pattern': self.dataset['file_pattern'],
                     'dtype': np.float32,
                     'index': 4,
                     'units': {'scaling': 1.,
                               'label': r'[units]'},
                    },
            'Bx'  : {'file_pattern': self.dataset['file_pattern'],
                     'dtype': np.float32,
                     'index': 5,
                     'units': {'scaling': np.sqrt(4. * np.pi),
                               'label': r'[G]'},
                    },
            'By'  : {'file_pattern': self.dataset['file_pattern'],
                     'dtype': np.float32,
                     'index': 6,
                     'units': {'scaling': np.sqrt(4. * np.pi),
                               'label': r'[G]'},
                    },
            'Bz'  : {'file_pattern': self.dataset['file_pattern'],
                     'dtype': np.float32,
                     'index': 7,
                     'units': {'scaling': np.sqrt(4. * np.pi),
                               'label': r'[G]'},
                    },
            'Temp': {'file_pattern': self.dataset['file_pattern'],
                     'dtype': np.float32,
                     'index': 8,
                     'units': {'scaling': 1.,
                               'label': r'[units]'},
                    },
            'Pres': {'file_pattern': self.dataset['file_pattern'],
                     'dtype': np.float32,
                     'index': 9,
                     'units': {'scaling': 1.,
                               'label': r'[units]'},
                    },
        }

    def read_var(self, iter: int, slice: int, var: str,
                 x_min: int = 0, x_max: int = None, y_min: int = 0, y_max: int = None):
        """ Read a slice from a MURaM data file.

        Parameters
        ----------
        iter: int. Iteration to read.
        slice: int. Slice to read.
        var: str. Variable to read.
        x_min: int. Minimum x coordinate.
        x_max: int. Maximum x coordinate.
        y_min: int. Minimum y coordinate.
        y_max: int. Maximum y coordinate.

        Returns
        ----------
        data: np.ndarray. Data read from the file.
        """

        # Meta data
        if var in self.vars.keys():
            # Get meta data
            meta = self.vars[var]

            # Get filename
            filename = meta['file_pattern'].format(iter=iter, slice=slice)

            # Compute itemsize
            itemsize = np.dtype(meta['dtype']).itemsize
            # Compute dimensions
            x_max, y_max = x_max or self.nx, y_max or self.ny
            # Compute offset to variable
            offset = (4 + (meta['index'] * self.nx * self.ny)) * itemsize

            # Memory map the file and read variable
            data_tmp = np.memmap(filename, dtype=meta['dtype'], mode='r', offset=offset, shape=(1, self.nx, self.ny),
                                 order='F')
            # Extract patch of data
            return data_tmp[0, y_min:y_max, x_min:x_max] * meta['units']['scaling']
        else:
            raise ValueError(f"Variable {var} not recognized.")

    def read(self, t: Union[int, list[int]], slices: Union[int, list[int]], vars: Union[str, list[str]],
             x_min: Union[int, list[int]] = 0, nx: int = None, y_min: Union[int, list[int]] = 0, ny: int = None,
             num_workers: int = None) -> np.ndarray:
        """ Read slices from different timesteps in parallel.

        Parameters
        ----------
        t: list[int]. List of timesteps to read.
        vars: list[str]. List of variables to read.
        slices: int. Slice to read.
        x_min: int. Minimum x coordinate.
        nx: int. Length of x coordinate.
        y_min: int. Minimum y coordinate.
        ny: int. Length of y coordinate.
        num_workers: int. Number of workers to use for parallel processing.

        Returns
        ----------
        data: list[np.ndarray]. List of data read from the files.
        """

        # Format variables to lists
        t = [t] if isinstance(t, int) else t
        iters = [self.iter[i] for i in t]
        if isinstance(vars, str):
            vars = [vars]
        if isinstance(slices, int):
            slices = [slices]

        # Format x_min, x_max, y_min, y_max to lists
        if isinstance(x_min, int):
            x_min = [x_min]
        if isinstance(y_min, int):
            y_min = [y_min]
        # Check if lengths match
        if not (len(x_min) == len(y_min)):
            raise ValueError("Length of x_min, x_max, y_min, y_max must match.")
        nx = nx if nx is not None else self.nx
        ny = ny if ny is not None else self.ny
        x_max = [x + nx for x in x_min]
        y_max = [y + ny for y in y_min]

        # Create iterables based on lengths of coordinates
        if len(iters) == len(x_min):
            coordinates = zip((x_min, x_max, y_min, y_max), iters)
        else:
            coordinates = list(itertools.product(zip(x_min, x_max, y_min, y_max), iters))
        iterables = list(itertools.product(coordinates, slices, vars))
        args = [(i, s, v, x_min, x_max, y_min, y_max) for ((x_min, x_max, y_min, y_max), i), s, v in iterables]

        # Number of workers
        if num_workers is None:
            num_workers = os.cpu_count() // 2

        # Process data in parallel
        with Pool(num_workers) as p:
            data = np.stack(list(tqdm(p.starmap(self.read_var, args), total=len(args))), axis=0)

        # Reshape data
        data = (data.reshape(len(coordinates), len(slices), len(vars), ny, nx)).transpose(3, 4, 1, 0, 2)

        return data


if __name__ == '__main__':

    # Test the MURaMQSDataset class
    ex_path = "E:\\Data\\ISSI_Team_Flows\\Matthias\\SSD_25x8Mm_16_pdmp_1_ISSI_Flows\\2D"  # os.path.abspath("../E/Data/ISSI_Team_Flows/Matthias/SSD_25x8Mm_16_pdmp_1_ISSI_Flows/2D/")
    ex_slice_type = 'yz'
    ex_slice = [192, 400]
    ex_iter = [0, 3900, 4200]
    ex_vars = ['I500', 'vx', 'vy', 'vz', 'Bx', 'By', 'Bz']
    ex_x_min, ex_x_max, ex_y_min, ex_y_max = [0, 48, 96, 144], [48, 96, 144, 192], [0, 48, 96, 144], [48, 96, 144, 192]

    # Initialize the dataset
    dataset = MURaMQSDataset(ex_path, dataset=ex_slice_type)

    # Measure slice_reader
    tracemalloc.start()
    t0 = time.time()
    data1 = dataset.read(ex_iter, ex_slice, ex_vars,
                         x_min=ex_x_min, nx=48, y_min=ex_y_min, ny=48)
    #data1 = dataset._read(ex_iter, ex_slice, ex_vars, x_min=ex_x_min, x_max=ex_x_max,
    #                      y_min=ex_y_min, y_max=ex_y_max)
    t1 = time.time()
    mem1, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"slice_reader: shape={data1.shape}, time={t1-t0:.3f}s, memory={mem1/1024/1024:.2f} MB")


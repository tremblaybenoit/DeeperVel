import os
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List
from track.utilities.instantiators import instantiate
from tqdm import tqdm
import glob
# import xarray as xr
import sqlite3
from track.data.read_write import read_fstd
from track.data.read_write import SQLiteDataset, FSTDDataset
from track.utilities.logic import get_config_path


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(config: DictConfig) -> None:
    """ Train neural network based on set of configurations.

        Parameters
        ----------
        config: str. Main hydra configuration file containing all model hyperparameters.

        Returns
        -------
        None.
    """
    # Initialize data config
    data_config = config.data

    # print(data_config.pro.profiles[0]['variable'])
    print('Profiles Start -------------------------------------------------')
    pro = instantiate(data_config.state.file_type,
                      filename=os.path.join(data_config.dir, "SimTrialonTrlLev_000m"))
    # pro = instantiate(data_config.pro.file_type)
    # pro.read(os.path.join(data_config.dir, "SimTrialonTrlLev_000m"))
    # pro.summary()
    # pro.read(os.path.join(data_config.dir, "PertOnTrlLev_000m"))
    pro.summary()
    # emmw_values = pro.fetchvar('j').values
    # print(np.mean(hu_values), hu_values.shape)
    print('Profiles End   -------------------------------------------------')
    print(' ')
    print('Radiances Start ------------------------------------------------')
    # nb_points = 0
    rad = None
    filenames = [f"{data_config.radiance.file_pattern}_{node:04d}_{core:04d}"
                 for core in range(1, data_config.radiance.file_split.cores + 1)
                 for node in range(1, data_config.radiance.file_split.nodes + 1)]
    rad = instantiate(data_config.radiance.file_type, filename=filenames[0],
                      attachments=filenames[1:])
    rad.summary()
    y = rad.fetchvarsfromtable('HEADER', variables=['ID_OBS'])
    ny = len(y)
    z = np.reshape(rad.fetchvarsfromtable('DATA', variables=['OBSVALUE']), (ny, -1))
    print(z.shape)
    exit()

    for node in tqdm(range(1, data_config.radiance.file_split.nodes + 1)):
        for core in tqdm(range(1, data_config.radiance.file_split.cores + 1)):
            filename = os.path.join(data_config.dir, f"obsto_amsua_{node:04d}_{core:04d}")
            if node == 1 and core == 1:
                rad = instantiate(data_config.radiance.file_type, filename=filename)
            else:
                if rad is not None:
                    rad.attach(filename)
                else:
                    raise ValueError('First file not read.')
            # nb_points = nb_points + len(rad.fetchvarsfromtable('HEADER', variables=['ID_OBS']))
    nb_points = len(rad.fetchvarsfromtable('HEADER', variables=['ID_OBS']))
    print(f'Number of points: {nb_points}')

    y = rad.fetchvarsfromtable('HEADER', variables=['ID_OBS'])
    ny = len(y)
    x = np.reshape(rad.fetchvarsfromtable('DATA', variables=['ID_OBS']), (ny, -1))
    z = np.reshape(rad.fetchvarsfromtable('DATA', variables=['OBSVALUE']), (ny, -1))
    print(z[0, :])
    print(z[1, :])
    print(z[2, :])

    # rad.summary()
    rad.close()
    # rad = instantiate(data_config.radiance.file_type,
    #                   filename=os.path.join(data_config.dir, "obsto_amsua_0001_0001"))
    # rad2 = instantiate(data_config.radiance.file_type,
    #                    filename=os.path.join(data_config.dir, "obsto_amsua_0001_0016"))
    # rad3 = instantiate(data_config.radiance.file_type,
    #                    filename=os.path.join(data_config.dir, "obsto_amsua_0010_0001"))
    # rad4 = instantiate(data_config.radiance.file_type,
    #                    filename=os.path.join(data_config.dir, "obsto_amsua_0010_0016"))
    # rad.summary()
    # print(f'Number of points: {nb_points}')
    # lat = rad.fetchvarsfromtable('HEADER', variables=['LAT'])
    # lon = rad.fetchvarsfromtable('HEADER', variables=['LON'])
    # print('Latitude: ', np.amin(lat), np.amax(lat))
    # print('Longitude: ', np.amin(lon), np.amax(lon))
    # rad2.summary()
    # rad3.summary()
    # rad4.summary()
    # print(rad.fetchvarsfromtable('HEADER', variables=['ID_STN']))
    # print(rad.fetchvarsfromtable('DATA', variables=['ID_DATA']))
    # print(data_config.radiance.filters)
    # channels = [data_config.radiance.filters[i]['channel'] for i in range(len(data_config.radiance.filters))]
    # print(channels)
    exit()

    # Look for thermodynamic profiles
    print('Profiles Start -------------------------------------------------')
    '''
    print(f"Opening directory: {data_config.dir}")
    pro_files = glob.glob(os.path.join(data_config.dir, '*.nc*'))
    print(f"Number of files found: {len(pro_files)}")
    pro = [xr.open_dataset(file, engine="scipy") for file in pro_files]
    pro_vars = [state.variables.keys() for state in pro]
    print('------------------')
    for i in range(len(pro_files)):
        for j in pro_vars[i]:
            print(f'Variable: {j}')
            print(pro[i].variables[j])
            print(' ')
        print('------------------')
    hu = pro[1]['HU']
    print(hu)
    '''
    filename = os.path.join(data_config.dir, "SimTrialonTrlLev_000m")
    # pro = xr.open_dataset(filename, engine="fstd")
    # pro = read_fstd(filename)
    # print(pro)
    # pro_vars = [state.variables.keys() for state in pro]
    # pro_vars = list(pro.variables.keys())
    # print(pro_vars)
    # hu = pro['HU']
    # print(hu)
    # print(hu['time'])
    # x = hu.values
    # print(x.shape)

    pro = FSTDObj(filename)
    pro.summary()
    hu = pro.fetchvar('HU')
    print(hu)
    print(hu['time'])
    print(hu.values.shape)

    print('Profiles End   -------------------------------------------------')

    print(' ')
    print('Radiances Start ------------------------------------------------')
    # Radiances
    filename = os.path.join(data_config.dir, "obsto_amsua_0001_0001")

    obs = SQLiteObj(filename)
    obs.summary()

    vars = ['ID_OBS']
    header_id = obs.fetchvarsfromtable('HEADER', variables=vars)
    print(header_id[0:5])
    vars = ['ID_OBS']
    data_id = obs.fetchvarsfromtable('DATA', variables=vars)
    print(data_id[0:30])

    obs.close()
    print('Radiances End   ------------------------------------------------')




if __name__ == '__main__':
    """ Learn how to read MIDAS state and radiance files.

        Parameters
        ----------
        --config_path: str. Directory containing configuration file.
        --config_name: str. Configuration filename.
        +experiment: str. Experiment configuration filename to override default configuration.

        Returns
        -------
        checkpoint: Training weights & biases.
    """

    main()

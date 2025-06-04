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

    state_file = os.path.join(data_config.dir, "SimTrialonTrlLev_000m")
    radiance_files = [f"{data_config.dataset.radiance.file_pattern}_{node:04d}_{core:04d}"
                      for core in range(1, data_config.dataset.radiance.file_split.cores + 1)
                      for node in range(1, data_config.dataset.radiance.file_split.nodes + 1)]

    # Load MIDAS dataset
    midas = instantiate(data_config)
    # midas.setup(stage='train')

    exit()


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

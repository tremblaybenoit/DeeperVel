import os
import numpy as np
import pickle
from omegaconf import DictConfig, ListConfig
import pytorch_lightning as lightning
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Union, Dict
from track.utilities.instantiators import instantiate
from track.data.process import preprocess, postprocess
import logging


# Initialize logger
logger = logging.getLogger(__name__)


class BaseDataModule(lightning.LightningDataModule):

    def __init__(self, input: DictConfig, output: DictConfig, split: DictConfig = None,
                 batch_size: int = 32, num_workers: int = None, pin_memory: bool = True, shuffle: bool = True) -> None:
        """ Loads paired samples of input and output data.

            Parameters
            ----------
            input: DictConfig. Input data.
            output: DictConfig. Output data.
            split: DictConfig, optional. Training/validation/testing split, by default None.
            batch_size: int, optional. Batch size, by default 32
            num_workers: int, optional. Number of workers, by default None.
            pin_memory: bool, optional. Pin memory for faster data transfer, by default True.
            shuffle: bool, optional. Shuffle training data, by default True.

            Returns
            -------
            None.
        """

        #  Class inheritance
        super().__init__()

        # Number of cpus
        self.num_workers = num_workers if num_workers is not None else os.cpu_count() // 2
        # Neural network training batch size
        self.batch_size = batch_size
        # Pin memory for faster data transfer
        self.pin_memory = pin_memory
        # Split
        self.ds_split = split
        # Shuffle training data
        self.shuffle = shuffle

        # Configuration
        self.ds_input = input
        self.ds_output = output

        # Datasets
        self.ds_train = None
        self.ds_valid = None
        self.ds_test = None
        self.ds_pred = None

    def train_dataloader(self) -> DataLoader:
        """ Loads training set.

            Parameters
            ----------
            None.

            Returns
            -------
            Training set (inputs and outputs).

        """
        return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=True, shuffle=self.shuffle)

    def val_dataloader(self) -> DataLoader:
        """ Load validation set.

            Parameters
            ----------
            None.

            Returns
            -------
            Validation set (inputs & outputs).

        """
        return DataLoader(self.ds_valid, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        """ Load test set.

            Parameters
            ----------
            None.

            Returns
            -------
            Test set (inputs & outputs).

        """
        return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=True)

    def predict_dataloader(self) -> DataLoader:
        """ Load prediction set.

            Parameters
            ----------
            None.

            Returns
            -------
            Prediction set (inputs & outputs if available).

        """
        return DataLoader(self.ds_pred, batch_size=1, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=True)


class LazyDataModule(BaseDataModule):

    def __init__(self, input: DictConfig, output: DictConfig, split: DictConfig = None, augment: bool = None,
                 batch_size: int = 32, num_workers: int = None, pin_memory: bool = True, shuffle: bool = True) -> None:
        """ Loads paired data samples of radiances and thermodynamic profiles.

            Parameters
            ----------
            input: DictConfig. Input data configuration.
            output: DictConfig. Output data configuration.
            split: DictConfig, optional. Training/validation/testing split, by default None.
            batch_size: int, optional. Batch size, by default 32
            num_workers: int, optional. Number of workers, by default None.
            pin_memory: bool, optional. Pin memory for faster data transfer, by default True.
            shuffle: bool, optional. Shuffle training data, by default True.
            augment: bool, optional. Apply data augmentation, by default None.

            Returns
            -------
            None.

        """

        #  Class inheritance
        super().__init__(input=input, output=output, split=split, batch_size=batch_size,
                         num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle)

        # Transformations and scaling
        self.transform = True
        self.scaling = True
        self.augment = augment

    def setup(self, stage: str = None) -> None:
        """ Splits datasets into training, validation, testing, and prediction sets.

            Parameters
            ----------
            stage: str. Current operation: "train" for training, "test" for testing,
                        "predict" for inference.

            Returns
            -------
            None.

        """

        # Training/validation/testing datasets
        if stage in ["train", "test"]:

            # Read patches
            if os.path.exists(self.ds_split.patches):
                # Load from the file
                with open(self.ds_split.patches, 'rb') as file:
                    patches = pickle.load(file)

            # Training & validation sets
            if stage == "train":
                # Split patches
                patches_train = {
                    key: var[0:int(np.floor(self.ds_split.train * len(var)))] if isinstance(var, list) else var
                    for key, var in patches.items()
                }
                patches_valid = {
                    key: var[int(np.floor(self.ds_split.train * len(var))):
                             int(np.floor((self.ds_split.train + self.ds_split.valid) * len(var)))]
                    if isinstance(var, list) else var
                    for key, var in patches.items()
                }
                self.ds_train = LazyDataset(self.ds_input, output=self.ds_output, patches=patches_train, augment=self.augment,
                                            scaling=self.scaling, transform=self.transform)
                self.ds_valid = LazyDataset(self.ds_input, output=self.ds_output, patches=patches_valid, augment=self.augment,
                                            scaling=self.scaling, transform=self.transform)

            elif stage == "test":
                # Split patches
                patches_test = {
                    key: var[int(np.floor((self.ds_split.train + self.ds_split.valid) * len(var))):
                             int(np.floor((self.ds_split.train + self.ds_split.valid + self.ds_split.test) * len(var)))]
                    if isinstance(var, list) else var
                    for key, var in patches.items()
                }
                # Test set
                self.ds_test = LazyDataset(self.ds_input, output=self.ds_output, patches=patches_test,
                                           scaling=self.scaling, transform=self.transform)

        # Prediction dataset
        elif stage == "predict":
            self.ds_pred = LazyDataset(self.ds_input, scaling=self.scaling, transform=self.transform)

    def predict(self, input: DictConfig, patches: Dict = None) -> None:
        """ Load prediction dataset.

            Parameters
            ----------
            input: DictConfig. Input data configuration.
            patches: Dict, optional. Patches to read, by default None.

            Returns
            -------
            None.
        """

        # Prediction dataset
        self.ds_pred = LazyDataset(input, patches=patches, scaling=self.scaling, transform=self.transform)


class BaseDataset(Dataset):

    def __init__(self, config: DictConfig, patches: Dict = None, scaling: bool = False, transform: bool = False) \
            -> None:
        """ Loads and transforms paired data samples of radiances and thermodynamic profiles.

            Parameters
            ----------
            config: DictConfig. Configuration file containing data and statistics paths.
            patches: DictConfig, optional. Patches to read, by default None.
            scaling: bool, optional. Apply scaling, by default False.
            transform: bool, optional. Apply transformations, by default False.

            Returns
            -------
            None.
        """

        # Class inheritance
        super().__init__()

        # Read input variables, slices, and timesteps
        self.io = instantiate(config.io, _partial_=False)
        self.vars = config.variables
        self.slices = config.slices
        self.dt = config.dt if hasattr(config, 'dt') else [0]

        # If patches are provided, use them to set coordinates
        if patches is not None:
            self.x_min = patches['x_min']
            self.x_max = patches['x_max']
            self.y_min = patches['y_min']
            self.y_max = patches['y_max']
            self.nx = patches['nx']
            self.ny = patches['ny']
            self.t = [[dt + t] for t in patches['t'] for dt in self.dt]
        # Else use full FOV
        else:
            # If no patches are provided, set coordinates to full FOV
            self.t = [t for t in range(self.io.nt)]
            self.x_min = [0 for _ in range(self.io.nt)]
            self.x_max = [None for _ in range(self.io.nt)]
            self.y_min = [0 for _ in range(self.io.nt)]
            self.y_max = [None for _ in range(self.io.nt)]
            self.nx = self.io.nx
            self.ny = self.io.ny

        # If stats are provided, store them and instantiate scaling objects
        if config.stats is not None:
            # Verify if stats are available
            if os.path.exists(config.stats):
                # Load from the file
                with open(config.stats, 'rb') as file:
                    stats = pickle.load(file)
                    # Extract statistics along relevant channels only
                    self.input_stats = stats[self.vars.keys()]
            else:
                logger.error(f"Statistics file {config.stats} does not exist.")
                raise ValueError(f"Statistics file {config.stats} does not exist.")
        else:
            self.stats = None

        # Scaling and transformation
        self.scaling = scaling
        self.transform = transform

    def read(self, t: Union[int, list[int]] = None, slices: Union[int, list[int]] = None,
             vars: Union[str, list[str]] = None, x_min: Union[int, list[int]] = 0, nx: int = None,
             y_min: Union[int, list[int]] = 0, ny: int = None, num_workers=None) -> np.ndarray:
        """ Read data from the input/output data.

            Parameters
            ----------
            t: int or list of int. List of timesteps to read.
            slices: int or list of int. List of slices to read.
            vars: str or list of str. List of variables to read.
            x_min: int, optional. Minimum x-coordinate, by default 0.
            nx: int, optional. Width of the patch, by default None.
            y_min: int, optional. Minimum y-coordinate, by default 0.
            ny: int, optional. Height of the patch, by default None.
            num_workers: int, optional. Number of workers to use for reading data, by default None.
            Returns
            -------
            data: Dict[str, np.ndarray]. Dictionary containing data for each variable.
        """

        # If no specific iters, slices, or vars are provided, use the default ones
        t = self.t if t is None else t
        slices = self.slices if slices is None else slices
        vars = self.vars.keys() if vars is None else vars
        x_min = self.x_min if x_min is None else x_min
        y_min = self.y_min if y_min is None else y_min
        nx = self.nx if nx is None else nx
        ny = self.ny if ny is None else ny

        # Read input data
        data = self.io.read(t, slices, vars, x_min=x_min, nx=nx, y_min=y_min, ny=ny, num_workers=num_workers)

        return data

    def preprocess(self, data: np.ndarray, scaling: bool = True, transform: bool = True) -> np.ndarray:
        """ Apply transformations and scaling to the data.

            Parameters
            ----------
            data: np.ndarray. Data in its original format.
            scaling: bool, optional. Apply scaling, by default True.
            transform: bool, optional. Apply transformations, by default True.

            Returns
            -------
            data_prep: np.ndarray. Preprocessed data.
        """

        # Extract radiance data at specified index
        if len(self.vars) > 1:
            return np.stack([preprocess(data[..., v], self.vars[var], self.stats[var], scaling=scaling,
                                        transform=transform)
                             for v, var in enumerate(self.vars.keys())], axis=-1)
        else:
            return preprocess(data[..., self.vars.keys()[0]], self.vars[self.vars.keys()[0]],
                              self.stats[self.vars.keys()[0]], scaling=scaling, transform=transform)

    def __len__(self) -> int:
        """ Get the length of the dataset.

            Parameters
            ----------
            None.

            Returns
            -------
            int: Length of the dataset.
        """
        return len(self.t)

    def __getitem__(self, item: int):
        """ Get data and apply transformations.

            Parameters
            ----------
            item: int. Index of item to read.

            Returns
            -------
            Data: Float.
        """

        # Read data
        data = self.io.read(self.t[item], self.slices, self.vars.keys(), nx=self.nx, ny=self.ny,
                            x_min=self.x_min[item], y_min=self.y_min[item])

        # Apply transformations
        if self.scaling or self.transform:
            data = self.preprocess(data, scaling=self.scaling, transform=self.transform)

        return data


class LazyDataset(Dataset):

    def __init__(self, input: DictConfig, output: DictConfig = None, patches: Dict = None,
                 scaling: bool = False, transform: bool = False, augment: bool = False) -> None:
        """ Loads and transforms paired data samples of radiances and thermodynamic profiles.

            Parameters
            ----------
            input: DictConfig. Input data configuration.
            output: DictConfig, optional. Output data configuration, by default None.
            patches: Dict, optional. Patches to read, by default None.
            scaling: bool, optional. Apply scaling, by default False.
            transform: bool, optional. Apply transformations, by default False.
            augment: bool, optional. Apply data augmentation, by default False.

            Returns
            -------
            None.
        """

        # Class inheritance
        super().__init__()

        # Read input variables, slices, and timesteps
        self.input = BaseDataset(input.io, patches=patches)
        # If output is provided, read it
        if output is not None:
            self.output = BaseDataset(output.io, patches=patches)
        else:
            self.output = None

        # Scaling and transformation
        self.scaling = scaling
        self.transform = transform
        self.augment = augment

    def __len__(self) -> int:
        """ Get the length of the dataset.

            Parameters
            ----------
            None.

            Returns
            -------
            int: Length of the dataset.
        """
        return len(self.input.t)

    def __getitem__(self, item: int) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """ Get data and apply transformations.

            Parameters
            ----------
            item: int. Index of item to read.

            Returns
            -------
            Data: Float.
        """

        # Read data
        input_data = self.input[item]

        # If output is provided, read it
        if self.output is not None:
            # Read output data
            output_data = self.output[item]

            # Apply augmentation to input_data and output_data if specified
            if self.augment:
                # TODO: Implement data augmentation logic here
                pass

            # Transform output data if specified
            if self.scaling or self.transform:
                output_data = self.output.preprocess(output_data, scaling=self.scaling, transform=self.transform)
        else:
            output_data = None

        # Transform input data if specified
        if self.scaling or self.transform:
            input_data = self.input.preprocess(input_data, scaling=self.scaling, transform=self.transform)

        # Return input and output data
        if output_data is not None:
            return input_data.reshape(self.input.ny, self.input.nx, -1), output_data.reshape(self.input.ny, self.input.nx, -1)
        else:
            return input_data.reshape(self.input.ny, self.input.nx, -1)  # Reshape to (ny, nx, channels) if no output data is available

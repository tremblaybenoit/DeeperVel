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
from track.data.transformations import geometric_augmentation
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


# Initialize logger
logger = logging.getLogger(__name__)


def get_item(args):
    ds, i = args
    return ds[i]

def load_all(ds):
    with ProcessPoolExecutor() as executor:
        return list(tqdm(
            executor.map(get_item, [(ds, i) for i in range(len(ds))], chunksize=32),
            total=len(ds),
            desc="Loading dataset"
        ))


class BaseDataModule(lightning.LightningDataModule):

    def __init__(self, input: Union[DictConfig, ListConfig], output: Union[DictConfig, ListConfig],
                 split: DictConfig = None, batch_size: int = 32, num_workers: int = None, pin_memory: bool = True,
                 shuffle: bool = True) -> None:
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

    def __init__(self, input: DictConfig, output: DictConfig, split: DictConfig = None, augment: bool = False,
                 scaling: bool = True, transform: bool = True,
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
        self.transform = transform
        self.scaling = scaling
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

        # Load patches
        if stage in ["train", "test"] and hasattr(self.ds_split, 'patches'):
            # Read patches
            if os.path.exists(self.ds_split.patches):
                # Load from the file
                with open(self.ds_split.patches, 'rb') as file:
                    patches = pickle.load(file)

        # Training & validation sets
        if stage == "train":
            # If split is provided, use it to create training and validation sets
            patches_train = {
                key: var[:self.ds_split.train] if isinstance(var, list) else var for key, var in patches.items()
            }
            patches_valid = {
                key: var[self.ds_split.train:self.ds_split.train + self.ds_split.valid]
                if isinstance(var, list) else var for key, var in patches.items()
            }
            logger.info(f"Loading training set samples")
            self.ds_train = MultiDataset(self.ds_input, output=self.ds_output, augment=self.augment,
                                        scaling=self.scaling, transform=self.transform, x_min=patches_train['x_min'],
                                        nx=patches_train['nx'], y_min=patches_train['y_min'], ny=patches_train['ny'],
                                        t=patches_train['t'])
            logger.info(f"Loading validation set samples")
            self.ds_valid = MultiDataset(self.ds_input, output=self.ds_output,  # augment=self.augment,
                                        scaling=self.scaling, transform=self.transform, x_min=patches_valid['x_min'],
                                        nx=patches_valid['nx'], y_min=patches_valid['y_min'], ny=patches_valid['ny'],
                                        t=patches_valid['t'])

        elif stage == "test":

            # Test set
            if hasattr(self.ds_split, 'test') and self.ds_split.test is not None:
                # Use the test set from the split
                patches_test = {
                    key: var[self.ds_split.train+self.ds_split.valid:self.ds_split.train+self.ds_split.valid+self.ds_split.test]
                    if isinstance(var, list) else var for key, var in patches.items()
                }
                self.ds_test = MultiDataset(self.ds_input, output=self.ds_output, scaling=self.scaling,
                                           transform=self.transform, x_min=patches_test['x_min'],
                                           nx=patches_test['nx'], y_min=patches_test['y_min'], ny=patches_test['ny'],
                                           t=patches_test['t'])
            else:
                # Use all available data
                self.ds_test = MultiDataset(self.ds_input, output=self.ds_output, scaling=self.scaling,
                                           transform=self.transform)

        # Prediction dataset
        elif stage == "predict":
            self.ds_pred = MultiDataset(self.ds_input, scaling=self.scaling, transform=self.transform)

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
        self.ds_pred = LazyDataset(input, scaling=self.scaling, transform=self.transform)


class BaseDataset(Dataset):

    def __init__(self, config: DictConfig, scaling: bool = False, transform: bool = False,
                 t: Union[list, int] = None, x_min: Union[list, int] = None, nx: Union[list, int] = None,
                 y_min: Union[list, int] = None, ny: Union[list, int] = None) -> None:
        """ Loads and transforms paired data samples of radiances and thermodynamic profiles.

            Parameters
            ----------
            config: DictConfig. Configuration file containing data and statistics paths.
            scaling: bool, optional. Apply scaling, by default False.
            transform: bool, optional. Apply transformations, by default False.
            t: list or int, optional. List of timesteps to read, by default None.
            x_min: list or int, optional. Minimum x-coordinate, by default None.
            nx: list or int, optional. Width of the patch, by default None.
            y_min: list or int, optional. Minimum y-coordinate, by default None.
            ny: list or int, optional. Height of the patch, by default None.

            Returns
            -------
            None.
        """

        # Class inheritance
        super().__init__()

        # Read input variables, slices, and timesteps
        self.io = instantiate(config.dataset.io, _partial_=False)
        self.vars = config.variables
        self.slices = config.slices
        self.dt = config.dt if hasattr(config, 'dt') else [0]

        # Patches
        self.t = [t for t in range(self.io.nt)] if t is None else [[dt_i + t_i for dt_i in self.dt] for t_i in t]
        self.x_min = [0 for _ in range(len(self.t))] if x_min is None else x_min
        self.y_min = [0 for _ in range(len(self.t))] if y_min is None else y_min
        self.nx = self.io.nx if nx is None else nx
        self.ny = self.io.ny if ny is None else ny

        # If stats are provided, store them and instantiate scaling objects
        if hasattr(config.dataset, 'statistics'):
            # Verify if stats are available
            if os.path.exists(config.dataset.statistics.path):
                # Load from the file
                with open(config.dataset.statistics.path, 'rb') as file:
                    stats = pickle.load(file)
                    # Extract statistics along relevant channels only
                    self.stats = {slice: {var: stats[slice][var] for var in list(self.vars.keys())} for slice in self.slices}
            else:
                logger.error(f"Statistics file {config.dataset.statistics.path} does not exist.")
                raise ValueError(f"Statistics file {config.dataset.statistics.path} does not exist.")
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
        vars = list(self.vars.keys()) if vars is None else vars
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
        return np.concatenate([
            np.stack([
                preprocess(data[:, :, s_idx, :, v_idx], self.vars[var],
                           self.stats[s][var], scaling=scaling, transform=transform)
                for v_idx, var in enumerate(list(self.vars.keys()))
            ], axis=-1).reshape((data.shape[0], data.shape[1], 1, data.shape[3], data.shape[4]))  # Stack over variables
            for s_idx, s in enumerate(self.slices if isinstance(self.slices, ListConfig) else [self.slices])
        ], axis=2)  # Stack over slices

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
        data = self.io.read(self.t[item], self.slices, list(self.vars.keys()), nx=self.nx, ny=self.ny,
                            x_min=self.x_min[item], y_min=self.y_min[item], num_workers=1)

        # Apply transformations
        if self.scaling or self.transform:
            data = self.preprocess(data, scaling=self.scaling, transform=self.transform)

        return data


class LazyDataset(Dataset):

    def __init__(self, input: DictConfig, output: DictConfig = None, scaling: bool = False, transform: bool = False,
                 augment: bool = False, t: Union[list, int] = None, x_min: Union[list, int] = None,
                 nx: Union[list, int] = None, y_min: Union[list, int] = None, ny: Union[list, int] = None) -> None:
        """ Loads and transforms paired data samples of radiances and thermodynamic profiles.

            Parameters
            ----------
            input: DictConfig. Input data configuration.
            output: DictConfig, optional. Output data configuration, by default None.
            scaling: bool, optional. Apply scaling, by default False.
            transform: bool, optional. Apply transformations, by default False.
            augment: bool, optional. Apply data augmentation, by default False.
            t: list or int, optional. List of timesteps to read, by default None.
            x_min: list or int, optional. Minimum x-coordinate, by default None.
            nx: list or int, optional. Width of the patch, by default None.
            y_min: list or int, optional. Minimum y-coordinate, by default None.
            ny: list or int, optional. Height of the patch, by default None.

            Returns
            -------
            None.
        """

        # Class inheritance
        super().__init__()

        # Read input variables, slices, and timesteps
        self.input = BaseDataset(input.dataset.io, t=t, x_min=x_min, nx=nx, y_min=y_min, ny=ny)
        # If output is provided, read it
        if output is not None:
            self.output = BaseDataset(output.dataset.io, t=t, x_min=x_min, nx=nx, y_min=y_min, ny=ny)
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
                # Combinations for geometric augmentation
                combinations = [
                    (0, None),  # identity
                    (1, None),  # rot90
                    (2, None),  # rot180
                    (3, None),  # rot270
                    (0, 0),  # flip x
                    (0, 1),  # flip y
                    (1, 0),  # rot90 + flip x
                    (1, 1),  # rot90 + flip y
                ]
                n_flip, n_rot90 = combinations[np.random.randint(0, 8)]
                axes_rot90 = (0, 1)
                # Apply geometric augmentation to input and output data
                input_data = geometric_augmentation(input_data, self.input.vars.keys(),
                                                    n_flip=n_flip, n_rot90=n_rot90, axes_rot90=axes_rot90)
                output_data = geometric_augmentation(output_data, self.output.vars.keys(),
                                                    n_flip=n_flip, n_rot90=n_rot90, axes_rot90=axes_rot90)

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


class MultiLazyDataset(Dataset):
    def __init__(self, input: ListConfig, output: ListConfig = None, scaling: bool = False,
                 transform: bool = False, augment: bool = False, t: Union[list, int] = None,
                 x_min: Union[list, int] = None, nx: Union[list, int] = None, y_min: Union[list, int] = None,
                 ny: Union[list, int] = None) -> None:
        """ Loads and transforms paired data samples.

            Parameters
            ----------
            input: ListConfig. List of input data configurations.
            output: ListConfig, optional. List of output data configurations, by default None.
            scaling: bool, optional. Apply scaling, by default False.
            transform: bool, optional. Apply transformations, by default False.
            augment: bool, optional. Apply data augmentation, by default False.
            t: list or int, optional. List of timesteps to read, by default None.
            x_min: list or int, optional. Minimum x-coordinate, by default None.
            nx: list or int, optional. Width of the patch, by default None.
            y_min: list or int, optional. Minimum y-coordinate, by default None.
            ny: list or int, optional. Height of the patch, by default None.

            Returns
            -------
            None.
        """

        # Class inheritance
        super().__init__()

        # Read input variables, slices, and timesteps
        self.inputs = [BaseDataset(cfg, t=t, x_min=x_min, nx=nx, y_min=y_min, ny=ny) for cfg in input]
        self.outputs = [BaseDataset(cfg, t=t, x_min=x_min, nx=nx, y_min=y_min, ny=ny) for cfg in output] if output is not None else None
        # Transformations and scaling
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
        return len(self.inputs[0].t)

    def __getitem__(self, item: int) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """ Get data and apply transformations.

            Parameters
            ----------
            item: int. Index of item to read.

            Returns
            -------
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: Input data, and optionally output data.
        """

        # Read data from all input datasets
        input_data = [ds[item] for ds in self.inputs]

        # If output datasets are provided, read them as well
        if self.outputs is not None:
            # Read data from all output datasets
            output_data = [ds[item] for ds in self.outputs]

            # Apply augmentations if specified
            if self.augment:
                # Combinations for geometric augmentation
                combinations = [
                    (0, None),  # identity
                    (1, None),  # rot90
                    (2, None),  # rot180
                    (3, None),  # rot270
                    (0, 0),  # flip x
                    (0, 1),  # flip y
                    (1, 0),  # rot90 + flip x
                    (1, 1),  # rot90 + flip y
                ]
                n_flip, n_rot90 = combinations[np.random.randint(0, 8)]
                axes_rot90 = (0, 1)
                # Apply augmentation to input and output data
                input_data = [geometric_augmentation(data, list(ds.vars.keys()), n_flip=n_flip, n_rot90=n_rot90,
                                                     axes_rot90=axes_rot90)
                              for ds, data in zip(self.inputs, input_data)]
                output_data = [geometric_augmentation(data, list(ds.vars.keys()), n_flip=n_flip, n_rot90=n_rot90,
                                                      axes_rot90=axes_rot90)
                               for ds, data in zip(self.outputs, output_data)]
        else:
            output_data = None

        # Preprocess input and output data if scaling or transformation is specified
        if self.scaling or self.transform:
            # Preprocess input data
            input_data = [ds.preprocess(data, scaling=self.scaling, transform=self.transform) for ds, data in zip(self.inputs, input_data)]
            # Preprocess output data if available
            if output_data is not None:
                output_data = [ds.preprocess(data, scaling=self.scaling, transform=self.transform) for ds, data in zip(self.outputs, output_data)]

        # Combine input and output data into a single array
        input_combined = np.concatenate([data.reshape(ds.ny, ds.nx, -1) for ds, data in zip(self.inputs, input_data)], axis=-1)
        # If output data is available, combine it as well
        if output_data is not None:
            # Combine output data into a single array
            output_combined = np.concatenate([data.reshape(ds.ny, ds.nx, -1) for ds, data in zip(self.outputs, output_data)], axis=-1)

            return input_combined, output_combined
        # If no output data is available, return only input data
        else:
            return input_combined


class MultiDataset(Dataset):
    def __init__(self, input: ListConfig, output: ListConfig = None, scaling: bool = False,
                 transform: bool = False, augment: bool = False, t: Union[list, int] = None,
                 x_min: Union[list, int] = None, nx: Union[list, int] = None, y_min: Union[list, int] = None,
                 ny: Union[list, int] = None) -> None:
        """ Loads and transforms paired data samples into memory.

            Parameters
            ----------
            input: ListConfig. List of input data configurations.
            output: ListConfig, optional. List of output data configurations, by default None.
            scaling: bool, optional. Apply scaling, by default False.
            transform: bool, optional. Apply transformations, by default False.
            augment: bool, optional. Apply data augmentation, by default False.
            t: list or int, optional. List of timesteps to read, by default None.
            x_min: list or int, optional. Minimum x-coordinate, by default None.
            nx: list or int, optional. Width of the patch, by default None.
            y_min: list or int, optional. Minimum y-coordinate, by default None.
            ny: list or int, optional. Height of the patch, by default None.

            Returns
            -------
            None.
        """

        # Class inheritance
        super().__init__()

        # Read input variables, slices, and timesteps
        self.inputs = [BaseDataset(cfg, t=t, x_min=x_min, nx=nx, y_min=y_min, ny=ny) for cfg in input]
        self.outputs = [BaseDataset(cfg, t=t, x_min=x_min, nx=nx, y_min=y_min, ny=ny) for cfg in output] if output is not None else None
        # Transformations and scaling
        self.scaling = scaling
        self.transform = transform
        self.augment = augment

        # Read all data into memory
        # self.input_data = [[ds[i] for i in range(len(ds))] for ds in self.inputs]
        # self.output_data = [[ds[i] for i in range(len(ds))] for ds in self.outputs] if self.outputs is not None else None
        self.input_data = [load_all(ds) for ds in tqdm(self.inputs)]
        self.output_data = [load_all(ds) for ds in tqdm(self.outputs)] if self.outputs is not None else None
        self.length = len(self.input_data[0])

    def __len__(self) -> int:
        """ Get the length of the dataset.

            Parameters
            ----------
            None.

            Returns
            -------
            int: Length of the dataset.
        """
        return self.length

    def __getitem__(self, item: int) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """ Get data and apply transformations.

            Parameters
            ----------
            item: int. Index of item to read.

            Returns
            -------
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: Input data, and optionally output data.
        """

        # Extract data from memory
        input_data = [data[item].astype('float64') for data in self.input_data]
        output_data = [data[item].astype('float64') for data in self.output_data] if self.output_data is not None else None

        # Apply augmentation if specified
        if self.augment:
            combinations = [
                (0, None),  # identity
                (1, None),  # rot90
                (2, None),  # rot180
                (3, None),  # rot270
                (0, 1),  # flip x
                (0, 0),  # flip y
                (1, 1),  # rot90 + flip x
                (1, 0),  # rot90 + flip y
            ]
            n_rot90, n_flip = combinations[np.random.randint(0, 8)]
            axes_rot90 = (0, 1)
            input_data = [geometric_augmentation(data, list(ds.vars.keys()), n_flip=n_flip, n_rot90=n_rot90, axes_rot90=axes_rot90)
                          for ds, data in zip(self.inputs, input_data)]
            if output_data is not None:
                output_data = [geometric_augmentation(data, list(ds.vars.keys()), n_flip=n_flip, n_rot90=n_rot90, axes_rot90=axes_rot90)
                               for ds, data in zip(self.outputs, output_data)]

        # Apply scaling/transform if specified
        if self.scaling or self.transform:
            input_data = [ds.preprocess(data, scaling=self.scaling, transform=self.transform) for ds, data in zip(self.inputs, input_data)]
            if output_data is not None:
                output_data = [ds.preprocess(data, scaling=self.scaling, transform=self.transform) for ds, data in zip(self.outputs, output_data)]

        # Combine input and output data into a single array
        input_combined = np.concatenate([data.reshape(ds.ny, ds.nx, -1) for ds, data in zip(self.inputs, input_data)], axis=-1)
        if output_data is not None:
            output_combined = np.concatenate([data.reshape(ds.ny, ds.nx, -1) for ds, data in zip(self.outputs, output_data)], axis=-1)
            return input_combined.transpose(2, 0, 1), output_combined.transpose(2, 0, 1)
        else:
            return input_combined.transpose(2, 0, 1)

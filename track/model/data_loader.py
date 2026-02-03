import os
import numpy as np
from omegaconf import DictConfig, ListConfig
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Union
from track.utilities.instantiators import instantiate
import logging
from track.data.transformations import identity, geometric_augmentation, augment_vector, augment_scalar
import torch


# Initialize logger
logger = logging.getLogger(__name__)


class BaseDataloader(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = None,
                 persistent_workers: bool = True, pin_memory: bool = True, shuffle: bool = True) -> None:
        """ Base dataloader class.

        Parameters
        ----------
        batch_size : int. Batch size for the dataloader.
        num_workers : int. Number of workers for the dataloader.
        persistent_workers : bool. If True, the data loader will keep workers alive between epochs.
        pin_memory : bool. If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
        shuffle : bool. If True, the data loader will shuffle the data at every epoch.

        Returns
        -------
        None.
        """

        #  Class inheritance
        super().__init__()

        # Number of cpus
        self.num_workers = num_workers if num_workers is not None else os.cpu_count() // 2
        # Persistent workers for faster data loading
        self.persistent_workers = persistent_workers
        # Neural network training batch size
        self.batch_size = batch_size
        # Pin memory for faster data transfer
        self.pin_memory = pin_memory
        # Shuffle data at every epoch
        self.shuffle = shuffle

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
            Training set (inputs & outputs).

        """
        return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers, shuffle=self.shuffle)

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
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

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
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

    def predict_dataloader(self) -> DataLoader:
        """ Load prediction set.

            Parameters
            ----------
            None.

            Returns
            -------
            Prediction set (inputs & outputs if available).

        """
        return DataLoader(self.ds_pred, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)


class Dataloader(BaseDataloader):
    def __init__(self, stage: DictConfig, batch_size: int = 32, num_workers: int = None,
                 persistent_workers: bool = True, pin_memory: bool = True) -> None:
        """ Dataloader for the CRTM dataset.

        Parameters
        ----------
        stage: DictConfig. Configuration object for the dataset at each stage (train, valid, test, pred).
        batch_size : int. Batch size for the dataloader.
        num_workers : int. Number of workers for the dataloader.
        persistent_workers : bool. If True, the data loader will not shut down the worker processes after a dataset has been consumed.
        pin_memory : bool. If True, the data loader will copy Tensors into CUDA pinned memory before returning them.

        Returns
        -------
        None.
        """

        #  Class inheritance
        super().__init__(batch_size=batch_size, num_workers=num_workers, persistent_workers=persistent_workers,
                         pin_memory=pin_memory)

        # Data sets
        self.stage = stage

    def setup(self, stage: str):
        """ Set up the dataset for training, validation, testing, or prediction.

            Parameters
            ----------
            stage : str. Stage of the model ('train', 'valid', 'test', 'predict').

            Returns
            -------
            None.
        """

        # Load datasets
        if stage == 'train':
            # Training/validation data
            self.ds_train, self.ds_valid = instantiate(self.stage.train), instantiate(self.stage.valid)
        elif stage == 'test':
            # Test/prediction data
            self.ds_test = instantiate(self.stage.test)
        elif stage == 'pred':
            # Prediction data
            self.ds_pred = instantiate(self.stage.predict)


class InMemoryDataset(Dataset):
    """ Dataset class that loads all data into memory at initialization. """

    def __init__(self, input: DictConfig, target: DictConfig = None, results: DictConfig = None,
                 augment: bool = False, patches: dict = None) -> None:
        """ Initialize the dataset.

            Parameters
            ----------
            input: DictConfig. Configuration object for the input variables.
            target: DictConfig. Configuration object for the target variables.
            results: DictConfig. Configuration object for the results.
            augment: bool, optional. Apply data augmentation, by default False.
            patches: dict, optional. Dictionary containing patch information, by default None.

            Returns
            -------
            None.
        """

        # Class inheritance
        super().__init__()

        # Store patches
        self.patches = instantiate(patches) if patches is not None else None
        # Store input and output configurations
        self.input, self.target = input, target
        # Store results configuration
        self.results = results
        # Data augmentation
        self.augment = augment

        # Preload all input data into memory
        self.input_data = [
            self._load_and_slice(self.input, i)
            for i in range(self._get_length(self.input))
        ]
        # Preload all target data if present
        self.target_data = [
            self._load_and_slice(self.target, i)
            for i in range(self._get_length(self.target))
        ] if self.target is not None else None

    def _get_length(self, config):
        """ Get the length of the dataset for a given config. """
        if self.patches is not None:
            return len(self.patches['t'])
        return len(config.variables[list(config.variables.keys())[0]].path)

    def __len__(self) -> int:
        """ Get the length of the dataset. """
        return len(self.input_data)

    def _load_and_slice(self, config: DictConfig, item: int) -> tuple[list, list, list]:
        """ Load and slice data based on the configuration and item index.

            Parameters
            ----------
            config: DictConfig. Configuration object for the variables.
            item: int. Index of the item to load.

            Returns
            -------
            tuple: (list of variable names, list of data arrays, list of normalization functions).
        """

        # Initialize an empty list to store the data
        var_list = list(config.variables.keys())
        data, norm = [], []
        # Loop through slices, variables, and time deltas to load and slice the data
        has_slices = hasattr(config, 'slices')
        slices = config.slices if has_slices else [None]
        for s in slices:
            for dt in config.dt:
                for var in var_list:
                    # Determine the index based on whether patches are used
                    idx = self.patches['t'][item] + dt if self.patches is not None else item + dt
                    # Load the data from the specified path
                    path = config.variables[var].path[s][idx] if has_slices else config.variables[var].path[idx]
                    # Instantiate the loading function and load the data
                    arr = instantiate(config.variables[var].load, path=path)
                    f_norm = instantiate(config.variables[var].normalization) \
                        if hasattr(config.variables[var], 'normalization') else identity
                    # If patches are used, slice the data accordingly
                    if self.patches is not None and 'x_min' in self.patches and 'y_min' in self.patches:
                        y_min, ny = self.patches['y_min'][item], self.patches['ny']
                        x_min, nx = self.patches['x_min'][item], self.patches['nx']
                        arr = arr[y_min:y_min + ny, x_min:x_min + nx]
                    # Store variable names, data, and normalization functions
                    data.append(arr)
                    norm.append(f_norm)
        # Stack the data along a new axis and return it
        return var_list, data, norm

    def __getitem__(self, idx: int):
        """ Get data from memory.

            Parameters
            ----------
            idx: int. Index of item to read.

            Returns
            -------
            Data: Float.
        """

        # Extract input and target data
        input_vars, input_data, input_norm = self.input_data[idx]
        target_vars, target_data, target_norm = self.target_data[idx] if self.target_data is not None else None

        # Apply data augmentation if enabled
        if self.augment:
            # Randomly select a combination of transformations
            idx = np.random.randint(0, 8)
            combinations = [
                (0, None), (1, None), (2, None), (3, None),
                (0, 1), (0, 0), (1, 1), (1, 0),
            ]
            n_rot90, flip = combinations[idx]
            augment_parameters = {'n_rot90': n_rot90, 'flip': flip, 'axes_rot90': (0, 1)}
            # Apply geometric augmentation to input and target data
            input_data = geometric_augmentation(input_data, input_vars, **augment_parameters)
            if target_data is not None:
                target_data = geometric_augmentation(target_data, target_vars, **augment_parameters)

        # Apply normalization functions to input and target data
        input_data = np.stack([f(d) for d, f in zip(input_data, input_norm)])
        target_data = np.stack([f(d) for d, f in zip(target_data, target_norm)]) if target_data is not None else None

        if target_data is not None:
            return input_data, target_data
        else:
            return input_data


class InMemoryDatasets(Dataset):
    """Dataset class for stacking multiple InMemoryDataset outputs per sample."""

    def __init__(self, input: ListConfig, target: ListConfig = None, results: DictConfig = None,
                 augment: bool = False, patches: dict = None) -> None:
        """ Dataset for stacking multiple InMemoryDataset instances.

            Parameters
            ----------
            input: ListConfig. List of configurations for input variables.
            target: ListConfig, optional. List of configurations for target variables, by default None.
            results: DictConfig. Configuration object for the results.
            augment: bool, optional. Apply data augmentation, by default False.
            patches: dict, optional. Dictionary containing patch information, by default None.

            Returns
            -------
            None.
        """

        # Class inheritance
        super().__init__()
        # Create a list of InMemoryDataset instances for each input-target pair
        if target is not None:
            self.datasets = [InMemoryDataset(i, t, results, augment, patches) for i, t in zip(input, target)]
        else:
            self.datasets = [InMemoryDataset(i, target, results, augment, patches) for i in input]
        # Ensure all datasets have the same length
        lengths = [len(ds) for ds in self.datasets]
        if not all(l == lengths[0] for l in lengths):
            raise ValueError("All InMemoryDataset instances must have the same length for stacking.")
        self.length = lengths[0]

    def __len__(self) -> int:
        """ Get the length of the stacked dataset.

            Returns
            -------
            int. Length of the dataset.
        """
        return self.length

    def __getitem__(self, idx: int):
        """ Get stacked data from multiple datasets."""

        # Retrieve samples from each dataset
        samples = [ds[idx] for ds in self.datasets]
        # If targets are present, each sample is a tuple (input, target)
        if isinstance(samples[0], tuple):
            inputs, targets = zip(*samples)
            return np.concatenate(inputs, axis=0), np.concatenate(targets, axis=0)
        # If no targets, each sample is just input
        else:
            return np.concatenate(samples, axis=0)


class LazyDataset(Dataset):
    """ Dataset class for lazy loading a single dataset."""

    def __init__(self, input: DictConfig, target: DictConfig = None, results: DictConfig = None, augment: bool=False,
                 patches: dict=None) -> None:
        """ Initialize the dataset.

            Parameters
            ----------
            input: DictConfig. Configuration object for the input variables.
            target: DictConfig. Configuration object for the target variables.
            results: DictConfig. Configuration object for the results.
            augment: bool, optional. Apply data augmentation, by default False.

            Returns
            -------
            None.
        """

        # Class inheritance
        super().__init__()

        # Store patches
        self.patches = instantiate(patches) if patches is not None else None
        # Store input and output configurations
        self.input, self.target = input, target
        # Store results configuration
        self.results = results
        # Data augmentation
        self.augment = augment

    def __len__(self) -> int:
        """ Get the length of the dataset."""

        # Return the length of the dataset based on the patches
        if self.patches is not None:
            return len(self.patches['t'])
        # If no patches are provided, return the length of the input variable paths
        return len(self.input.variables[list(self.input.variables.keys())[0]].path)

    def _load_and_slice(self, config: DictConfig, item: int, augment_parameters: dict = None) -> np.ndarray:
        """ Load and slice data based on the configuration and item index.

            Parameters
            ----------
            config: DictConfig. Configuration object for the variables.
            item: int. Index of the item to load.
            augment_parameters: dict, optional. Parameters for data augmentation, by default None.

            Returns
            -------
            tuple: (list of variable names, list of data arrays, list of normalization functions).
        """

        # Initialize an empty list to store the data
        var_list = list(config.variables.keys())
        data = []
        # Loop through slices, variables, and time deltas to load and slice the data
        has_slices = hasattr(config, 'slices')
        slices = config.slices if has_slices else [None]
        for s in slices:
            for dt in config.dt:
                sub_data = []
                sub_norm = []
                for var in var_list:
                    # Determine the index based on whether patches are used
                    idx = self.patches['t'][item] + dt if self.patches is not None else item + dt
                    # Load the data from the specified path
                    path = config.variables[var].path[s][idx] if has_slices else config.variables[var].path[idx]
                    # Instantiate the loading function and load the data
                    arr = instantiate(config.variables[var].load, path=path)
                    f_norm = instantiate(config.variables[var].normalization) \
                        if hasattr(config.variables[var], 'normalization') else identity
                    # If patches are used, slice the data accordingly
                    if self.patches is not None and 'x_min' in self.patches and 'y_min' in self.patches:
                        y_min, ny = self.patches['y_min'][item], self.patches['ny']
                        x_min, nx = self.patches['x_min'][item], self.patches['nx']
                        arr = arr[y_min:y_min + ny, x_min:x_min + nx]
                    # Append to sub-lists
                    sub_data.append(arr)
                    sub_norm.append(f_norm)
                # Apply data augmentation if parameters are provided
                if augment_parameters is not None:
                    sub_data = geometric_augmentation(sub_data, var_list, **augment_parameters)
                # Apply normalization functions and append to the main data list
                sub_data = [f(d) for d, f in zip(sub_data, sub_norm)]
                data.extend(sub_data)
        # Stack the data along a new axis and return it
        return np.stack(data)

    def __getitem__(self, item: int) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """ Get data and apply transformations.

            Parameters
            ----------
            item: int. Index of item to read.

            Returns
            -------
            Data: Float.
        """

        # Apply data augmentation if enabled
        if self.augment:
            # Randomly select a combination of transformations
            idx = np.random.randint(0, 8)
            combinations = [
                (0, None), (1, None), (2, None), (3, None),
                (0, 1), (0, 0), (1, 1), (1, 0),
            ]
            n_rot90, flip = combinations[idx]
            augment_parameters = {'n_rot90': n_rot90, 'flip': flip, 'axes_rot90': (0, 1)}
        else:
            augment_parameters = None

        # Load and slice input and target data
        input_data = self._load_and_slice(self.input, item, augment_parameters=augment_parameters)
        target_data = self._load_and_slice(self.target, item, augment_parameters=augment_parameters) \
            if self.target is not None else None

        # Return the data, transposed to have channels first
        if target_data is not None:
            return input_data, target_data
        else:
            return input_data


class LazyDatasets(Dataset):
    """Dataset class for stacking multiple LazyDataset outputs per sample."""

    def __init__(self, input: ListConfig, target: ListConfig = None, results: DictConfig = None, augment: bool = False,
                 patches: dict = None) -> None:
        """ Dataset for stacking multiple LazyDataset instances.

            Parameters
            ----------
            input: ListConfig. List of configurations for input variables.
            target: ListConfig, optional. List of configurations for target variables, by default None.
            results: DictConfig. Configuration object for the results.
            augment: bool, optional. Apply data augmentation, by default False.
            patches: dict, optional. Dictionary containing patch information, by default None.

            Returns
            -------
            None.
        """

        # Class inheritance
        super().__init__()
        # Create a list of LazyDataset instances for each input-target pair
        if target is not None:
            self.datasets = [LazyDataset(i, t, results, augment, patches) for i, t in zip(input, target)]
        else:
            self.datasets = [LazyDataset(i, target, results, augment, patches) for i in input]
        # Ensure all datasets have the same length
        lengths = [len(ds) for ds in self.datasets]
        if not all(l == lengths[0] for l in lengths):
            raise ValueError("All LazyDataset instances must have the same length for stacking.")
        self.length = lengths[0]

    def __len__(self) -> int:
        """ Get the length of the stacked dataset.

            Returns
            -------
            int. Length of the dataset.
        """
        return self.length

    def __getitem__(self, idx: int):
        """ Get stacked data from multiple datasets."""

        # Retrieve samples from each dataset
        samples = [ds[idx] for ds in self.datasets]
        # If targets are present, each sample is a tuple (input, target)
        if isinstance(samples[0], tuple):
            inputs, targets = zip(*samples)
            return np.concatenate(inputs, axis=0), np.concatenate(targets, axis=0)
        # If no targets, each sample is just input
        else:
            return np.concatenate(samples, axis=0)



class TorchDataset(Dataset):
    """ Dataset using torch shared memory tensors. """
    def __init__(self, input: DictConfig, target: DictConfig = None, results: DictConfig = None) -> None:
        """ Initialize the dataset.

            Parameters
            ----------
            input : DictConfig. Configuration for input variables.
            target : DictConfig. Configuration for target variables (optional).
            results : DictConfig. Configuration for results (optional).

            Returns
            -------
            None.
        """

        def _load(var_dict: DictConfig) -> torch.Tensor:
            """ Load and normalize variables from configuration.

                Parameters
                ----------
                var_dict : DictConfig. Configuration for variable.

                Returns
                -------
                torch.Tensor. Loaded and normalized tensor in shared memory.
            """
            arr = load_var(var_dict)
            # if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
            t = torch.from_numpy(arr)
            t.share_memory_()
            return t

        # Input loading
        self.input, self.input_keys = {}, []
        # We store the keys in a list once so __getitem__ doesn't have to look them up
        input_keys = list(input.keys())
        for key in input_keys:
            data = _load(input[key])
            self.input[key] = data
            self.input_keys.append(key)

        # 2. Automate Target Block Loading
        self.target, self.target_keys = {}, []
        self.has_targets = target is not None
        if self.has_targets:
            target_keys = list(target.keys())
            for key in target_keys:
                data = _load(target[key])
                self.target[key] = data
                self.target_keys.append(key)

        # Store results config
        self.results = results

    def __len__(self):
        """ Get the length of the dataset."""
        return int(self.input[self.input_keys[0]].shape[0])

    def __getitem__(self, idx: int) -> dict:
        """ Get item from dataset.

            Parameters
            ----------
            idx : int. Index of the item to retrieve.

            Returns
            -------
            Dataset object.
        """

        # Use dictionary comprehension over pre-cached keys
        out = {'input': {k: self.input[k][idx] for k in self.input_keys}}

        # Add target dictionary only if it exists
        if self.has_targets:
            out['target'] = {k: self.target[k][idx] for k in self.target_keys}

        return out


class AugmentedTorchDataset(Dataset):
    """ Dataset using torch shared memory tensors. """
    def __init__(self, input: DictConfig, target: DictConfig = None, results: DictConfig = None, augment: bool=False,
                 vector_keys: list = None) -> None:
        """ Initialize the dataset.

            Parameters
            ----------
            input : DictConfig. Configuration for input variables.
            target : DictConfig. Configuration for target variables (optional).
            results : DictConfig. Configuration for results (optional).
            augment: bool, optional. Apply data augmentation, by default False.

            Returns
            -------
            None.
        """

        # Vector keys
        vector_keys = vector_keys if vector_keys is not None else [("vx", "vy"), ("Bx", "By"), ("Ex", "Ey")]

        # Input data
        self.input_vector, self.input_vector_keys, self.input_vector_norm, self.input_scalar, self.input_scalar_norm \
            = self._organize_data(input, vector_keys)
        # Target data
        self.target_vector, self.target_vector_keys, self.target_vector_norm, self.target_scalar, self.target_scalar_norm \
            = self._organize_data(target, vector_keys)
        self.has_target = len(self.target_vector) > 0 or len(self.target_scalar) > 0

        # Dataset length
        self._len = int(self.input_scalar[0].shape[0]) if len(self.input_scalar) > 0 \
            else int(self.input_vector[0][0].shape[0])
        # Store results config
        self.results = results
        # Data augmentation
        self.augment = augment

    def __len__(self):
        """ Get the length of the dataset."""
        return self._len

    @staticmethod
    def _organize_data(config: DictConfig, vector_keys: list) -> Tuple[list, list, list, list, list]:
        """ Organize data into vector and scalar components.

            Parameters
            ----------
            config: DictConfig. Configuration for variables.

            Returns
            -------
            v_data: list. List of vector data tuples.
            v_norm: list. List of vector normalization functions.
            s_data: list. List of scalar data.
            s_norm: list. List of scalar normalization functions.
        """

        def _load(var_dict: DictConfig) -> torch.Tensor:
            """ Load and normalize variables from configuration.

                Parameters
                ----------
                var_dict : DictConfig. Configuration for variable.

                Returns
                -------
                torch.Tensor. Loaded and normalized tensor in shared memory.
            """
            arr = load_var(var_dict)
            # if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
            t = torch.from_numpy(arr)
            t.share_memory_()
            return t

        # Return empty lists if config is None
        if config is None:
            return [], [], [], [], []

        # Organize vector data
        v_data, v_keys, v_norm = [], [], []
        for key1, key2 in vector_keys:
            if key1 in config and key2 in config:
                v_keys.append((key1, key2))
                v_data.append((_load(config[key1]), _load(config[key2])))
                v_norm.append((instantiate(config[key1].normalization), instantiate(config[key2].normalization)))
            elif key1 in config or key2 in config:
                raise ValueError(f"Both components of vector {(key1, key2)} must be present.")
        # Organize scalar data
        s_keys = [k for k in config.keys() if all(k not in pair for pair in v_keys)]
        s_data = [_load(config[k]) for k in s_keys]
        s_norm = [instantiate(config[k].normalization) for k in s_keys]

        # Return organized data
        return v_data, v_keys, v_norm, s_data, s_norm

    @staticmethod
    def _process_sample(item: int, v_data, v_norm, s_data, s_norm, augment_parameters: dict = None) -> torch.Tensor:
        """ Process a single sample with optional augmentation.

            Parameters
            ----------
            item: int. Index of the item to process.
            v_data: list. List of vector data tuples.
            v_norm: list. List of vector normalization functions.
            s_data: list. List of scalar data.
            s_norm: list. List of scalar normalization functions.
            augment_parameters: dict, optional. Parameters for data augmentation, by default None.

            Returns
            -------
            torch.Tensor. Processed sample tensor.
        """

        # Initialize an empty list to store the data
        data = []
        # Process vector data
        for (vx, vy), (fx, fy) in zip(v_data, v_norm):
            x_comp = vx[item]
            y_comp = vy[item]
            if augment_parameters is not None:
                x_comp, y_comp = augment_vector(x_comp, y_comp, **augment_parameters)
            data.append(fx(x_comp))
            data.append(fy(y_comp))
        # Process scalar data
        for scalar, fscalar in zip(s_data, s_norm):
            scalar_data = scalar[item]
            if augment_parameters is not None:
                scalar_data = augment_scalar(scalar_data, **augment_parameters)
            data.append(fscalar(scalar_data))
        # Stack tensors and return
        return torch.stack(data, dim=0)

    def __getitem__(self, item: int) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
        """ Get item from dataset.

            Parameters
            ----------
            item : int. Index of the item to retrieve.

            Returns
            -------
            Dataset object.
        """

        # Apply data augmentation if enabled
        if self.augment:
            # Randomly select a combination of transformations
            idx = np.random.randint(0, 8)
            combinations = [
                (0, None), (1, None), (2, None), (3, None),
                (0, 1), (0, 0), (1, 1), (1, 0),
            ]
            n_rot90, flip = combinations[idx]
            augment_parameters = {'n_rot90': n_rot90, 'flip': flip, 'axes_rot90': (0, 1)}
        else:
            augment_parameters = None

        # Load input data, starting with vectors then scalars
        input_data =  self._process_sample(item, self.input_vector, self.input_vector_norm,
                                           self.input_scalar, self.input_scalar_norm, augment_parameters)

        # Add target data if it exists
        if self.has_target:
            target_data = self._process_sample(item, self.target_vector, self.target_vector_norm,
                                               self.target_scalar, self.target_scalar_norm, augment_parameters)
            return input_data, target_data

        return input_data


class AugmentedLazyTorchDataset(AugmentedTorchDataset):
    """ Dataset using torch shared memory tensors and lazy loading. """
    def __init__(self, input: DictConfig, target: DictConfig = None, results: DictConfig = None, augment: bool=False,
                 vector_keys: list = None) -> None:
        """ Initialize the dataset.

            Parameters
            ----------
            input : DictConfig. Configuration for input variables.
            target : DictConfig. Configuration for target variables (optional).
            results : DictConfig. Configuration for results (optional).
            augment: bool, optional. Apply data augmentation, by default False.

            Returns
            -------
            None.
        """

        # Class inheritance
        super().__init__(input, target=target, results=results, augment=augment, vector_keys=vector_keys)

    @staticmethod
    def _organize_data(config: DictConfig, vector_keys: list) -> tuple[list, list, list, list, list]:
        """ Organize data into vector and scalar components.

            Parameters
            ----------
            config: DictConfig. Configuration for variables.

            Returns
            -------
            v_data: list. List of vector data tuples.
            v_norm: list. List of vector normalization functions.
            s_data: list. List of scalar data.
            s_norm: list. List of scalar normalization functions.
        """

        # Return empty lists if config is None
        if config is None:
            return [], [], [], [], []

        # Organize vector data
        v_data, v_keys, v_norm = [], [], []
        for key1, key2 in vector_keys:
            if key1 in config and key2 in config:
                v_keys.append((key1, key2))
                v_data.append((instantiate(config[key1].load), instantiate(config[key2].load)))
                v_norm.append((instantiate(config[key1].normalization), instantiate(config[key2].normalization)))
            elif key1 in config or key2 in config:
                raise ValueError(f"Both components of vector {(key1, key2)} must be present.")
        # Organize scalar data
        s_keys = [k for k in config.keys() if all(k not in pair for pair in v_keys)]
        s_data = [instantiate(config[k].load) for k in s_keys]
        s_norm = [instantiate(config[k].normalization) for k in s_keys]

        # Return organized data
        return v_data, v_keys, v_norm, s_data, s_norm

    @staticmethod
    def _process_sample(item: int, v_data, v_norm, s_data, s_norm, augment_parameters: dict = None) -> torch.Tensor:
        """ Process a single sample with optional augmentation.

            Parameters
            ----------
            item: int. Index of the item to process.
            v_data: list. List of vector data tuples.
            v_norm: list. List of vector normalization functions.
            s_data: list. List of scalar data.
            s_norm: list. List of scalar normalization functions.
            augment_parameters: dict, optional. Parameters for data augmentation, by default None.

            Returns
            -------
            torch.Tensor. Processed sample tensor.
        """

        # Initialize an empty list to store the data
        data = []
        # Process vector data
        for (vx, vy), (fx, fy) in zip(v_data, v_norm):
            x_comp = vx(item)
            y_comp = vy(item)
            if augment_parameters is not None:
                x_comp, y_comp = augment_vector(x_comp, y_comp, **augment_parameters)
            data.append(fx(x_comp))
            data.append(fy(y_comp))
        # Process scalar data
        for scalar, fscalar in zip(s_data, s_norm):
            scalar_data = scalar(item)
            if augment_parameters is not None:
                scalar_data = augment_scalar(scalar_data, **augment_parameters)
            data.append(fscalar(scalar_data))
        # Stack tensors and return
        return torch.stack(data, dim=0)

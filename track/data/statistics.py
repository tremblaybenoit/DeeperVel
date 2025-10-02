from typing import Union, Dict
import numpy as np
import pickle
import torch
import hydra
from omegaconf import DictConfig
from track.utilities.instantiators import instantiate
from track.utilities.logic import get_config_path
from tqdm import tqdm
import gc
import logging

# Initialize logger
logger = logging.getLogger(__name__)


def read_statistics(path: str, tensor: bool = False, dtype: str = 'float32') -> dict:
    """ Read statistics from a file.

        Parameters
        ----------
        path: str. Path to the file containing statistics.
        tensor: bool. If True, returns statistics as torch tensors, otherwise as numpy arrays.
        dtype: str. Data type of the torch tensors (if tensor=True).

        Returns
        -------
        Dictionary containing statistics of the dataset.
    """

    # Load statistics from file
    with open(path, 'rb') as file:
        stats = pickle.load(file)

    # Convert statistics to torch tensors if required
    if tensor:
        # Loop through each variable in stats and convert numpy arrays to torch tensors
        stats = {var: {key: torch.tensor(value, dtype=getattr(torch, dtype)) if isinstance(value, np.ndarray) else value
                       for key, value in var_stats.items()} for var, var_stats in stats.items()}
    else:
        # Ensure all statistics have the correct dtype
        stats = {var: {key: value.astype(dtype) if isinstance(value, np.ndarray) else value
                       for key, value in var_stats.items()} for var, var_stats in stats.items()}

    return stats


def read_statistics_var(path: str, var: str, tensor: bool = False, dtype: str = 'float32') -> dict:
    """ Read statistics of a specific variable from a file.

        Parameters
        ----------
        path: str. Path to the file containing statistics.
        var: str. Variable to read statistics for.
        tensor: bool. If True, returns statistics as torch tensors, otherwise as numpy arrays.
        dtype: str. Data type of the torch tensors (if tensor=True).

        Returns
        -------
        Dictionary containing statistics of the specified variable.
    """

    # Load statistics from file
    stats = read_statistics(path, dtype=dtype)[var]

    # Convert statistics to torch tensors if required
    if tensor:
        stats = {key: torch.tensor(value, dtype=getattr(torch, dtype)) if isinstance(value, np.ndarray) else value
                 for key, value in stats.items()}

    # Return statistics for the specified variable
    return stats


def read_statistics_slice_var(path: str, slice: Union[int, float], var: str, tensor: bool = False, dtype: str = 'float32') -> dict:
    """ Read statistics of a specific variable from a file.

        Parameters
        ----------
        path: str. Path to the file containing statistics.
        slice: Union[int, float]. Slice from which to read statistics.
        var: str. Variable to read statistics for.
        tensor: bool. If True, returns statistics as torch tensors, otherwise as numpy arrays.
        dtype: str. Data type of the torch tensors (if tensor=True).

        Returns
        -------
        Dictionary containing statistics of the specified variable.
    """

    # Load statistics from file
    stats = read_statistics(path, dtype=dtype)[slice][var]

    # Convert statistics to torch tensors if required
    if tensor:
        stats = {key: torch.tensor(value, dtype=getattr(torch, dtype)) if isinstance(value, np.ndarray) else value
                 for key, value in stats.items()}

    # Return statistics for the specified variable
    return stats


def combine_statistics(stats: list[dict[str, Union[np.ndarray, int]]]) -> dict[str, np.ndarray]:
    """ Combine statistics of multiple datasets based on the mathematical definition of mean, var, stdev, etc.
        The datasets make come from different sources, e.g. different instruments, different heights, etc.
        Thus, they may have been computed from a different number of samples.

        Parameters
        ----------
        stats: List[Dict[str, Union[np.ndarray, int]]]. List of statistics of different datasets.

        Returns
        -------
        Dictionary containing combined statistics (as if in a single dataset).
    """
    logger.info("Combining statistics of all datasets...")

    # Initialize combined statisticsL min, max, mean
    n_samples = np.sum([stat["n_samples"] for stat in stats], axis=0)
    combined_stats = {"min": np.min([stat["min"] for stat in stats], axis=0),
                      "max": np.max([stat["max"] for stat in stats], axis=0),
                      "mean": (np.sum([stat["n_samples"] * stat["mean"] for stat in stats], axis=0) / n_samples)}

    # Combine variance (based on its mathematical definition)
    combined_stats["variance"] = \
        (np.sum([stat["n_samples"] * (stat["variance"] + (stat["mean"] - combined_stats["mean"])**2)
                 for stat in stats], axis=0) / n_samples)

    # Combine stdev (based on its mathematical definition)
    combined_stats["stdev"] = np.sqrt(combined_stats["variance"])

    return combined_stats


def compute_statistics_in_chunks(data: Union[np.ndarray, torch.Tensor], axis: Union[int, tuple] = None,
                                 n_chunks: int = 10) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    """ Compute statistics of a large dataset by computing statistics of its parts and combining them.

        Parameters
        ----------
        data: np.ndarray or torch.Tensor. Dataset to compute statistics on.
        axis: int or tuple. Axis to compute statistics along.
        n_chunks: int. Number of chunks to split the dataset into.

        Returns
        -------
        Dictionary containing statistics of the dataset.
    """

    # Split data into chunks
    if isinstance(data, torch.Tensor):
        chunks = torch.chunk(data, n_chunks, dim=axis)
    else:
        chunks = np.array_split(data, n_chunks, axis=axis)

    # Compute statistics of each chunk
    stats = [statistics(chunk, axis=axis) for chunk in chunks]

    # Combine statistics
    return combine_statistics(stats)


def statistics(data: Union[np.ndarray, torch.Tensor], axis: Union[int, tuple] = None) \
        -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    """ Compute statistics of a given dataset.

        Parameters
        ----------
        data: np.ndarray or torch.Tensor. Dataset to compute statistics on.
        axis: int or tuple. Axis to compute statistics along.

        Returns
        -------
        Dictionary containing statistics of the dataset.
    """

    # Convert torch tensor to a numpy array
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    # Compute statistics
    return {
        "mean": np.mean(data, axis=axis),
        "stdev": np.std(data, axis=axis),
        "min": np.min(data, axis=axis),
        "max": np.max(data, axis=axis),
        "median": np.median(data, axis=axis),
        "variance": np.var(data, axis=axis),
        "n_samples": np.sum(~np.isnan(data), axis=axis)
    }


def compute_statistics(input: DictConfig, output: DictConfig = None) -> dict:
    """ Compute statistics of a given dataset.

        Parameters
        ----------
        input: DictConfig. Main hydra configuration file containing all model hyperparameters.
        output: DictConfig. Main hydra configuration file containing all model hyperparameters.

        Returns
        -------
        None.
    """

    def process_variable(data: np.ndarray, variable: str, stats_dict: dict):
        """ Compute statistics of a variable and combine with previous statistics if any.

        Parameters
        ---------
        data: np.ndarray. Data of the variable to compute statistics on.
        variable: str. Name of the variable.
        stats_dict: dict. Dictionary to store statistics.

        Returns
        -------
        None.
        """
        stat = statistics(data, axis=tuple(range(data.ndim - 1)))
        if variable in stats_dict:
            stats_dict[variable] = combine_statistics([stats_dict[variable], stat])
        else:
            stats_dict[variable] = stat

    def apply_transverse_invariance(stats_dict: dict, variables: dict):
        """ Apply transverse invariance to the statistics of the variables.

        Parameters
        ----------
        stats_dict: dict. Dictionary containing statistics of the variables.
        variables: dict. Dictionary containing variable configurations.

        Returns
        -------
        None.
        """
        for vpair in [("vx", "vy"), ("Bx", "By")]:
            if all(v in variables for v in vpair):
                negatives = [
                    {"mean": -stats_dict[v]["mean"],
                     "stdev": stats_dict[v]["stdev"],
                     "min": -stats_dict[v]["max"],
                     "max": -stats_dict[v]["min"],
                     "median": -stats_dict[v]["median"],
                     "variance": stats_dict[v]["variance"],
                     "n_samples": stats_dict[v]["n_samples"]}
                    for v in vpair
                ]
                combined_stats = combine_statistics([stats_dict[vpair[0]], stats_dict[vpair[1]], *negatives])
                stats_dict[vpair[0]] = stats_dict[vpair[1]] = combined_stats

    # Initialize statistics dictionary
    stats = {}

    # Retrieve path to files
    # path = instantiate(input.path)

    # Loop over variables
    for variable, variable_config in input.variables.items():
        logger.info(f"Computing statistics of {variable} dataset out of {len(list(input.variables.keys()))}...")
        # Retrieve path to files
        path = instantiate(variable_config.path)
        # If path is a dict, it means we have different levels (e.g. heights)
        if isinstance(path, dict):
            # Loop over levels
            for key, p in path.items():
                logger.info(f"Computing statistics of {variable} at level {key}...")
                # Initialize level in stats if not already present
                if key not in stats:
                    stats[key] = {}
                # If path contents is a list, loop over files
                filenames = p if isinstance(p, list) else [p]
                # Loop over files
                for filename in tqdm(filenames, desc=f"Processing files for {variable} at level {key}"):
                    # Load data
                    data = instantiate(variable_config['load'], path=filename)
                    # Compute statistics and combine with previous statistics if any
                    process_variable(data, variable, stats[key])
                    # Free memory
                    data = None
                    gc.collect()
        # If path is a list, loop over files
        else:
            # If path contents is a list, loop over files
            filenames = path if isinstance(path, list) else [path]
            # Loop over files
            for filename in filenames:
                # Load data
                data = instantiate(variable_config['load'], path=filename)
                # Compute statistics and combine with previous statistics if any
                process_variable(data, variable, stats)
                # Free memory
                data = None
                gc.collect()

    # Apply transverse invariance after all variables are processed
    if input.transverse_invariance:
        if any(isinstance(v, dict) for v in stats.values()):
            # If stats is organized by levels (dict of dicts)
            for key in stats:
                apply_transverse_invariance(stats[key], input.variables)
        else:
            apply_transverse_invariance(stats, input.variables)

    # Save statistics to file
    if output is not None:
        logger.info(f"Saving statistics to file {output.path}.")
        with open(output.path, 'wb') as file:
            # noinspection PyTypeChecker
            pickle.dump(stats, file)

    return stats


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """ Compute statistics of a given dataset.

        Parameters
        ----------
        config: DictConfig. Configuration file containing data and statistics paths.

        Returns
        -------
        None.
    """

    # If statistics is part of the preparation steps:
    if hasattr(config.preparation, "statistics"):
        for dataset, config_statistics in config.preparation.statistics.items():
            logger.info(f"Computing statistics of {dataset} dataset...")
            _ = instantiate(config_statistics)

    return


if __name__ == '__main__':
    """ Compute statistics of a given dataset.

        Parameters
        ----------
        --config_path: str. Directory containing configuration file.
        --config_name: str. Configuration filename.
        +experiment: str. Experiment configuration filename to override default configuration.

        Returns
        -------
        pickle file containing data statistics.
    """

    main()

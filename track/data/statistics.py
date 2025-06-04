from typing import Union, Dict, List
import numpy as np
import pickle
import torch
import hydra
from omegaconf import DictConfig
from track.utilities.instantiators import instantiate
from track.utilities.logic import get_config_path


def combine_statistics(stats: List[Dict[str, Union[np.ndarray, torch.Tensor, int]]]) \
        -> Dict[str, Union[np.ndarray, torch.Tensor, int]]:
    """ Combine statistics of multiple dataset based on the mathematical definition of the operations
        (mean, min, max, standard deviation, variance). As the datasets come from different sources, their
        statistics may have been computed on different number of samples.

        Parameters
        ----------
        stats: list of dictionaries. List of statistics of the different datasets.

        Returns
        -------
        Dictionary containing combined statistics.
    """

    # Initialize combined statistics (starting with min, max, and mean)
    n_samples = np.sum([stat["n_samples"] for stat in stats], axis=0)
    combined_stats = {"min": np.min([stat["min"] for stat in stats], axis=0),
                      "max": np.max([stat["max"] for stat in stats], axis=0),
                      "mean": np.sum([stat["mean"] * stat["n_samples"] for stat in stats], axis=0) / n_samples}

    # Combine variance (based on its mathematical definition)
    combined_stats["variance"] = (
            np.sum([stat["n_samples"] * (stat["variance"] + (stat["mean"] - combined_stats["mean"]) ** 2)
                    for stat in stats], axis=0) / n_samples)

    # Combined standard deviation (based on its mathematical definition)
    combined_stats["stdev"] = np.sqrt(combined_stats["variance"])

    # Return combined statistics
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


def compute_statistics(config: DictConfig, variables: DictConfig) -> None:
    """ Compute statistics of a given dataset.

        Parameters
        ----------
        config: DictConfig. Configuration file containing data and statistics paths.
        variables: list of str. Variables to compute statistics on.

        Returns
        -------
        pickle file containing data statistics.
    """

    # Reader class
    io = instantiate(config.input.data.io, _partial_=False)
    # Load data
    io.open(config.input.data.path)
    # Compute statistics per variable
    stats = {}

    # Loop sequentially over variables for memory efficiency
    for variable, variable_config in variables.items():
        # Reader class
        io = instantiate(config.input.data.io, _partial_=False)
        # Apply transformations
        if hasattr(variable_config, "transform"):
            data = variable_config.transform(*[io.data[i] for i in variables[v].use])
        else:
            # If no transformation is applied, just use the data
            if len(variables[v].use) == 1:
                data = io.data[variables[v].use[0]]
            # Else concatenate the data
            else:
                data = np.concatenate([io.data[i] for i in variables[v].use], axis=-1)

        # Compute statistics
        stats[v] = statistics(data, axis=tuple(range(data.ndim - 1)))
        # Clear memory
        data = None

    # Save statistics to a file
    with open(config.output.data.path, 'wb') as file:
        # noinspection PyTypeChecker
        pickle.dump(stats, file)
    # Clear memory
    io = None

    return


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

    # Initialize data config
    data_config = config.data

    # If statistics is part of the preparation steps:
    if hasattr(data_config.preparation, "statistics"):
        # Compute statistics
        compute_statistics(data_config, variables=data_config.variables)

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

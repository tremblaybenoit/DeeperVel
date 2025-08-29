import numpy as np
from typing import Dict, Union, Tuple
from omegaconf import DictConfig
from track.utilities.instantiators import instantiate
from track.utilities.logic import get_config_path


def preprocess(data: np.ndarray, config: DictConfig, stats: Dict = None, scaling: bool=True, transform: bool=True) \
        -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """ Read data and apply transformations.

        Parameters
        ----------
        data: np.ndarray. Data in its original format.
        config: DictConfig. Variables to read.
        stats: DictConfig. Statistics to use for scaling.
        scaling: bool, optional. Apply scaling, by default True.
        transform: bool, optional. Apply transformations, by default True.

        Returns
        -------
        data_prep: np.ndarray. Preprocessed data.
    """

    #  If there is no scaling and no transformation, return the data as is
    if not scaling and not transform:
        return data
    else:

        # Apply transformations
        if transform:
            data_prep = instantiate(config.transform, data=data, _partial_=False)
        else:
            data_prep = data

        # Apply scaling
        if scaling:
            # Verify if stats are available
            if stats is not None:
                data_prep = instantiate(config.scaling, data=data_prep, stats=stats, inverse_transform=False,
                                        _partial_=False)
            else:
                raise ValueError("No statistics available for scaling.")

        # Stack state data
        return data_prep


def postprocess(data: np.ndarray, config: DictConfig, stats: Dict = None, scaling: bool=True, transform: bool=True) \
        -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """ Transform data back to the original format.

        Parameters
        ----------
        data: np.ndarray. Preprocessed data.
        config: DictConfig. Variables to read.
        stats: DictConfig. Statistics to use for scaling.
        scaling: bool, optional. Apply scaling, by default True.
        transform: bool, optional. Apply transformations, by default True.

        Returns
        -------
        data: np.ndarray. Postprocessed data.
    """

    #  If there is no scaling and no transformation, return the data as is
    if not scaling and not transform:
        return data
    else:
        # TODO: Implement inverse transformations, if relevant?
        # Apply transformations
        #if transform:
        #    data_prep = config.transform(data)
        #else:
        #    data_prep = data

        # Apply scaling
        if scaling:
            # Verify if stats are available
            if stats is not None:
                data_prep = instantiate(config.scaling, data=data, stats=stats, inverse_transform=True,
                                        _partial_=False)
            else:
                raise ValueError("No statistics available for scaling.")
        else:
            data_prep = data

        # Stack state data
        return data_prep
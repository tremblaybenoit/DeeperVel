import hydra
from omegaconf import DictConfig, OmegaConf
from track.utilities.logic import get_config_path
from track.utilities.instantiators import instantiate
from track.data.colocate import colocate
import pickle
import numpy as np
from track.data.transformations import geometric_augmentation


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """ Train neural network based on a set of configurations.

        Parameters
        ----------
        config: str. Main hydra configuration file containing all model hyperparameters.

        Returns
        -------
        None.
    """

    # TODO: Test colocation
    colocations = colocate(config.data.preparation.colocate)

    # TODO: Read statistics
    if hasattr(config.data.preparation, "statistics"):
        # Read statistics from the pickle file
        with open(config.data.dataset.yz.statistics.path, 'rb') as file:
            statistics = pickle.load(file)

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

    # TODO: Test data loader (train)
    data_loader = instantiate(config.data.loader)
    data_loader.setup(stage='train')

    # Read one patch from the data loader
    patch = data_loader.ds_train[0][0]
    # Print its stats
    print(f"Patch stats per variable:",
          {i: {'min': np.min(patch[..., i]), 'max': np.max(patch[..., i])} for i in range(patch.shape[-1])})
    breakpoint()


    # TODO: Test data loader (test)
    data_loader2 = instantiate(config.data)
    data_loader2.setup(stage='test')

    # TODO: Test statistics (with transverse invariance)

    # TODO: Test augmentation

    # TODO: Test plotting routines

    return


if __name__ == '__main__':
    """ Train neural network to track plasma motions (or other physical quantities).

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

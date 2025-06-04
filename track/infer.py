import hydra
from omegaconf import DictConfig
import torch
from track.train import Tracker
from track.utilities.logic import get_config_path
torch.set_default_dtype(torch.float64)


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """ Use neural network to track plasma motions (or other physical quantities).

        Parameters
        ----------
        config: str. Main hydra configuration file containing all model hyperparameters.

        Returns
        -------
        None.
    """
    # Initialize trainer object and train
    Tracker(config).track()


if __name__ == '__main__':
    """ Use neural network to track plasma motions (or other physical quantities).

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
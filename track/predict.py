import hydra
import logging
from omegaconf import DictConfig
from track.utilities.instantiators import instantiate
from track.utilities.logic import get_config_path
from track.track import Tracker
import torch
torch.set_float32_matmul_precision('high')

# Initialize logger
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """ Make prediction using a neural network and its set of configurations.

        Parameters
        ----------
        config: str. Main hydra configuration file containing all model hyperparameters.

        Returns
        -------
        None.
    """

    # Initialize the trainer object
    logger.info("Initializing tracker...")
    tracker = Tracker(config)

    # Evaluate on the test set
    logger.info("Predict using tracker...")
    pred = tracker.predict()

    # Save predictions
    logger.info("Saving predictions to file...")
    # TODO: Add proper config
    save_function = instantiate(config.output.save_function)
    save_function(pred)


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

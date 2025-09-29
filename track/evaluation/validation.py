import numpy as np
import logging
import hydra
from omegaconf import DictConfig
from track.utilities.logic import get_config_path
from track.utilities.instantiators import instantiate
from track.utilities.plot import save_plot

# Initialize logger
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """ Validation of the neural network using test set results.

        Parameters
        ----------
        config: str. Main hydra configuration file containing all model hyperparameters.

        Returns
        -------
        None.
    """

    # Load test set results
    logger.info("Load test set results...")
    pred = {}
    for result_name, result_config in config.loader.stage.test.results.items():
        pred[result_name] = instantiate(result_config.load).astype(np.float32)

    # Load test set references
    logger.info("Load test set references...")
    ref = {}
    for var_name, var_config in config.loader.stage.test.target.items():
        ref[var_name] = np.array(instantiate(var_config.load)).astype(np.float32)

    # Plot
    logger.info("Plot comparison...")
    # TODO: Add plots


if __name__ == '__main__':
    """ Evaluate results from the test set.

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

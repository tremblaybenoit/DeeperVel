import numpy as np
import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as lightning
from track.data.process import postprocess
from track.utilities.instantiators import instantiate, instantiate_list
from track.utilities.logic import get_config_path
from track.utilities.logger import TrainerLogger
torch.set_float32_matmul_precision('high')


# Initialize logger
logger = logging.getLogger(__name__)


class Tracker:
    """Class for tracking plasma motions (or other physical quantities) using neural networks."""

    def __init__(self, config: DictConfig) -> None:
        """ Initialization of the trainer and its configuration.

            Parameters
            ----------
            config: Hydra configuration object.

            Returns
            -------
            None.

        """

        # Load config object and resolve paths
        OmegaConf.resolve(config)
        self.config = config
        self.checkpoint_path = (config.callbacks.model_checkpoint.dirpath +
                                f"/{config.callbacks.model_checkpoint.filename}.ckpt")

        # Initialization
        self.data_loader = None
        self.callbacks = None
        self.trainer_logger = None
        self.trainer = None
        self.model = None

        # For reproducibility, set the randomizer seed if provided
        if self.config.get("seed"):
            lightning.seed_everything(self.config.task_seed, workers=True)

    def setup(self, loader_config, stage: str='train') -> None:
        """ Setup trainer object.

            Parameters
            ----------
            loader_config: DictConfig. Configuration object for the data loader.
            stage: str. Stage of the training process.
                        Options are 'train', 'test', or 'predict'.

            Returns
            -------
            None.
        """

        # Data loader
        logger.info("Initializing data loader...")
        self.data_loader = instantiate(loader_config)
        # Generate training/validation/test sets
        self.data_loader.setup(stage=stage)

        # Trainer loggers and callbacks: Only activated during training
        if stage in ['train', 'test']:
            # Configure logger
            if self.trainer_logger is None:
                TrainerLogger(self.config.logger).configure()
                logger.info("Initializing logger(s)...")
                self.trainer_logger = instantiate_list(self.config.get("logger"), "logger")
                logger.info("Done with loggers, Initializing logger(s)...")

            # Callbacks
            if self.callbacks is None:
                logger.info("Initializing callback(s)...")
                self.callbacks = instantiate_list(self.config.get("callbacks"), "callbacks")

        else:
            self.trainer_logger, self.callbacks = False, None

        # Trainer
        logger.info("Waking up trainer...")
        self.trainer = instantiate(self.config.trainer, callbacks=self.callbacks, logger=self.trainer_logger)

    def train(self) -> None:
        """ Loads data, loggers, callbacks, trainer, and then trains and tests the model.
            Saves the training weights and biases in a checkpoint file.

            Parameters
            ----------
            None. Relies on self.config.

            Returns
            -------
            None. The model checkpoint (.ckpt) is stored in self.config.paths.checkpoint_dir.
        """

        # Data loader and trainer setup
        self.setup(self.config.loader, stage='train')

        # Model
        logger.info("Initializing model...")
        self.model = instantiate(self.config.model)

        # Train the model from scratch or continue training⚡
        resume_ckpt = self.config.get("resume_from_checkpoint", None)
        if resume_ckpt and os.path.exists(resume_ckpt):
            logger.info(f"Resuming training from checkpoint: {resume_ckpt}")
            self.trainer.fit(self.model, self.data_loader, ckpt_path=resume_ckpt)
        else:
            logger.info("Training...")
            self.trainer.fit(self.model, self.data_loader)
        print("Training complete.")

        # Save optimal model checkpoint
        print("Saving model checkpoint...")
        save_dictionary = OmegaConf.to_container(self.config.copy())
        save_dictionary['model'] = self.model
        torch.save(save_dictionary, self.checkpoint_path)

    def test(self) -> None:
        """ Loads data, loggers, callbacks, trainer, and then tests the model. Saves the test results in a file.

            Parameters
            ----------
            None. Relies on self.config.

            Returns
            -------
            None. The test results are stored in self.config.paths.checkpoint_dir.
        """

        # Data loader and trainer setup
        self.setup(self.config.loader, stage='test')

        # Load model from a checkpoint
        if self.model is None:
            logger.info("Loading checkpoint...")
            state = torch.load(self.checkpoint_path, weights_only=False)
            self.model = state['model']

        # Evaluate on the test set
        logger.info("Running against test set...")
        results = self.trainer.test(self.model, self.data_loader)
        # Extract results
        test_results = self.model.test_results
        # Reshape based on state variables and heights
        # TODO: Reshape properly
        predictions = test_results.reshape(test_results.shape[0],
                                           len(self.data_loader.ds_predict.state_variables), -1)

        # Postprocess predictions
        for v, variable in enumerate(self.data_loader.ds_predict.state_variables):
            # Undo transformations and save
            predictions[:, v, :] = postprocess(predictions[:, v, :],
                                               self.data_loader.ds_predict.state_variables[variable],
                                               self.data_loader.ds_predict.state_stats[variable])

        # Save test results to file
        logger.info("Saving test results...")
        io = instantiate(self.config.data.dataset.state.files.data.io, _partial_=False)
        io.open(os.path.join(self.config.callbacks.model_checkpoint.dirpath, 'test'), mode='w')
        # Reshape based on state variables and heights
        self.model.test_results = self.model.test_results.reshape(self.model.test_results.shape[0],
                                                                  len(self.data_loader.ds_test.state.variables), -1)

        # Save test results
        for v, variable in enumerate(self.data_loader.ds_test.state.variables):
            # Undo transformations and save
            io[variable] = postprocess(self.model.test_results[:, v, :],
                                       self.data_loader.ds_test.state.variables[variable],
                                       self.data_loader.ds_test.state.stats[variable])

    def predict(self, loader_config: DictConfig) -> np.ndarray:
        """ Predicts the output of the model on a given dataset.

            Parameters
            ----------
            loader_config: DictConfig. Configuration object for the data to predict on.

            Returns
            -------
            None.
        """

        # Data loader and trainer setup
        self.setup(loader_config, stage='predict')

        # Load model from a checkpoint
        if self.model is None:
            logger.info("Loading checkpoint...")
            state = torch.load(self.checkpoint_path, weights_only=False)
            self.model = state['model']

        # Predict on dataset
        logger.info("Predicting on dataset...")
        predictions = torch.cat(self.trainer.predict(self.model, self.data_loader), dim=0).cpu().numpy()
        # Reshape based on state variables and heights
        # TODO: Reshape properly
        predictions = predictions.reshape(predictions.shape[0],
                                          len(self.data_loader.ds_predict.state_variables), -1)

        # Postprocess predictions
        for v, variable in enumerate(self.data_loader.ds_predict.state_variables):
            # Undo transformations and save
            predictions[:, v, :] = postprocess(predictions[:, v, :],
                                               self.data_loader.ds_predict.state_variables[variable],
                                               self.data_loader.ds_predict.state_stats[variable])

        return predictions


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

    # Initialize the trainer object
    logger.info("Initializing tracker...")
    tracker = Tracker(config)

    # Train the model
    logger.info("Training tracker...")
    tracker.train()


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

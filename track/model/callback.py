from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
from track.utilities.plot import fig_scatterplots, fig_patches
import numpy as np
import wandb
import tempfile
import os


class FigureLogger(Callback):
    """
    Callback to log figures at the end of each validation epoch.
    """
    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(self, trainer, model):
        """
        Logs figures at the end of each validation epoch.

        Parameters
        ----------
        trainer: pytorch_lightning.Trainer. The trainer instance.
        model: pytorch_lightning.LightningModule. The model instance.

        Returns
        -------
        None. The figures are logged to the logger associated with the trainer.
        """

        # Extract results from the model
        current_epoch = trainer.current_epoch
        if trainer.sanity_checking is False:
            # Data
            target = np.concatenate(model.valid_target, axis=0).transpose(0, 2, 3, 1)
            result = np.concatenate(model.valid_result, axis=0).transpose(0, 2, 3, 1)

            # Generate figures
            fig0 = fig_scatterplots(target, result, title_prefix=f"Epoch {current_epoch:02d}")
            fig1 = fig_patches(target, result, title_prefix=f"Epoch {current_epoch:02d}",
                               img_range=(-2.5, 2.5))

            # Save figures to a buffer
            for logger in trainer.loggers if hasattr(trainer, "loggers") else [trainer.logger]:
                # TensorBoard
                if logger.__class__.__name__.lower().startswith("tensorboard"):
                    logger.experiment.add_figure(
                        tag=f"Scatterplots",
                        figure=fig0,
                        global_step=current_epoch
                    )
                    logger.experiment.add_figure(
                        tag=f"Patches",
                        figure=fig1,
                        global_step=current_epoch
                    )
                # WandB
                elif logger.__class__.__name__.lower().startswith("wandb"):
                    logger.experiment.log({
                        f"Scatterplots/Epoch_{current_epoch:02d}": wandb.Image(fig0),
                        f"Patches/Epoch_{current_epoch:02d}": wandb.Image(fig1)
                    })
                # MLflow
                elif logger.__class__.__name__.lower().startswith("mlflow"):
                    for name, fig in [
                        (f"Scatterplots_Epoch_{current_epoch:02d}.png", fig0),
                        (f"Patches_Epoch_{current_epoch:02d}.png", fig1)
                    ]:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            filename = os.path.join(tmpdir, name)
                            fig.savefig(filename)
                            logger.experiment.log_artifact(logger.run_id, filename, artifact_path="figures")

            # Close figures to free memory
            plt.close(fig0)
            plt.close(fig1)
            # breakpoint()
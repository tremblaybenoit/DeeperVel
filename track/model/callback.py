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

        # If in sanity checking, skip logging
        if trainer.sanity_checking:
            return

        # Extract results from the model
        current_epoch = trainer.current_epoch

        # Data
        target = model.valid_target
        result = model.valid_result

        # List of figures to log
        figs = []
        # Tags for each figure
        tags = ["Scatterplots", "Patches"]

        # Generate figures
        figs.append(fig_scatterplots(target, result, title_prefix=f"Epoch {current_epoch:02d}"))
        figs.append(fig_patches(target, result, title_prefix=f"Epoch {current_epoch:02d}", img_range=(-2.5, 2.5)))

        # Save figures to a buffer
        for logger in trainer.loggers if hasattr(trainer, "loggers") else [trainer.logger]:
            # TensorBoard
            if logger.__class__.__name__.lower().startswith("tensorboard"):
                for tag, fig in zip(tags, figs):
                    logger.experiment.add_figure(tag=tag, figure=fig, global_step=current_epoch)
            # WandB
            elif logger.__class__.__name__.lower().startswith("wandb"):
                logger.experiment.log(
                    {f"{tag}/Epoch_{current_epoch:02d}": wandb.Image(fig) for tag, fig in zip(tags, figs)})
            # MLflow
            elif logger.__class__.__name__.lower().startswith("mlflow"):
                for tag, fig in zip(tags, figs):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        filename = os.path.join(tmpdir, f"{tag}_Epoch_{current_epoch:02d}.png")
                        fig.savefig(filename)
                        logger.experiment.log_artifact(logger.run_id, filename, artifact_path="figures")

        # Close figures to free memory
        plt.close('all')
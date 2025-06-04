from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
import torch
from track.utilities.plot import scatterplot
import numpy as np
import wandb
import tempfile


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
        vx_target = torch.stack(model.valid_vx_target, dim=0).cpu().numpy()
        vx_pred = torch.stack(model.valid_vx_pred, dim=0).cpu().numpy()
        vy_target = torch.stack(model.valid_vy_target, dim=0).cpu().numpy()
        vy_pred = torch.stack(model.valid_vy_pred, dim=0).cpu().numpy()
        prof_err = (prof_pred - prof_target) / (np.std(prof_target, axis=0) + np.finfo(float).eps)
        clrsky = torch.stack(model.valid_clrsky, dim=0).cpu().numpy()

        # Generate figures
        fig0 = fig_vertical_profiles(prof_target, prof_pred, title=f"Epoch {current_epoch:02d} - Vertical profiles")
        fig1 = fig_vertical_profile(prof_err, title=f"Epoch {current_epoch:02d} - Vertical profile errors")
        fig2 = fig_rmse_bars(bt_target, bt_pred, clrsky,
                             title=[f"Epoch {current_epoch:02d} - Forward model errors",
                                    f"Epoch {current_epoch:02d} - Normalized forward model errors"])

        # Save figures to a buffer
        for logger in trainer.loggers if hasattr(trainer, "loggers") else [trainer.logger]:
            # TensorBoard
            if hasattr(logger, "experiment") and hasattr(logger.experiment, "add_figure"):
                logger.experiment.add_figure(
                    tag=f"VerticalProfiles/Epoch_{current_epoch:02d}",
                    figure=fig0,
                    global_step=current_epoch
                )
                logger.experiment.add_figure(
                    tag=f"VerticalErrors/Epoch_{current_epoch:02d}",
                    figure=fig1,
                    global_step=current_epoch
                )
                logger.experiment.add_figure(
                    tag=f"RadianceRMSE/Epoch_{current_epoch:02d}",
                    figure=fig2,
                    global_step=current_epoch
                )
            # WandB
            elif logger.__class__.__name__.lower().startswith("wandb"):
                logger.experiment.log({
                    f"VerticalProfiles/Epoch_{current_epoch:02d}": wandb.Image(fig0),
                    f"VerticalErrors/Epoch_{current_epoch:02d}": wandb.Image(fig1),
                    f"RadianceRMSE/Epoch_{current_epoch:02d}": wandb.Image(fig2)
                })
            # MLflow
            elif logger.__class__.__name__.lower().startswith("mlflow"):
                for name, fig in [
                    (f"VerticalProfiles_Epoch_{current_epoch:02d}.png", fig0),
                    (f"VerticalErrors_Epoch_{current_epoch:02d}.png", fig1),
                    (f"RadianceRMSE_Epoch_{current_epoch:02d}.png", fig2)
                ]:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                        fig.savefig(tmpfile.name)
                        logger.experiment.log_artifact(tmpfile.name, artifact_path="figures")

        # Close figures to free memory
        plt.close(fig0)
        plt.close(fig1)
        plt.close(fig2)
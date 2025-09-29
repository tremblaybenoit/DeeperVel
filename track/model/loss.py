import torch


def mse(pred, target):
    """ Compute MSE loss between prediction and target.

    Parameters
    ----------
    pred: torch.Tensor. Prediction tensor.
    target: torch.Tensor. Target tensor.

    Returns
    -------
    loss: torch.Tensor. MSE loss between prediction and target.

    """

    # Compute MSE loss
    loss = (pred - target)**2

    return loss


class MSE(torch.nn.Module):
    """ Mean Squared Error loss module."""
    def __init__(self):
        """ Initialize the MSE module.

        Returns
        -------
        None.
        """
        super().__init__()

    def to(self, device):
        """ Move the module to a specified device.

        Parameters
        ----------
        device: torch.device. Device to move the module to.
        """
        super().to(device)
        return self

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """ Compute the mean squared error between predicted and target tensors.

        Parameters
        ----------
        pred: torch.Tensor. Predicted tensor.
        target: torch.Tensor. True values.

        Returns
        -------
        torch.Tensor. Mean squared error over the batch.
        """
        return mse(pred, target)
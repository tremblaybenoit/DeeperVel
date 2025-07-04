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
    breakpoint()

    # Compute MSE loss
    loss = torch.nanmean((pred - target)**2)

    return loss
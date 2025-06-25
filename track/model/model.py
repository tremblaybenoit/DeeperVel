import sys
import torch
from typing import Callable, Tuple, Union
from pytorch_lightning import LightningModule
import torch.nn as nn


def same_padding(kernel_size: int, stride: int) -> int:
    """ Calculate padding to keep the same input and output size.

        Parameters
        ----------
        kernel_size: int. Size of the convolutional kernel.
        stride: int. Stride of the convolutional layer.

        Returns
        -------
        padding: int. Padding to apply.
    """
    return ((stride - 1) + kernel_size - 1) // 2


class ResidualBlock(nn.Module):
    """Residual block for neural network architecture."""
    def __init__(self, n_filters: int = 64, kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 activation: nn.Module = nn.ReLU()) -> None:
        """ Initialize residual block.

            Parameters
            ----------
            n_filters: int. Number of filters in the convolutional layers.
            kernel_size: int. Size of the convolutional kernel.
            stride: int. Stride of the convolutional layers.
            padding: int. Padding for the convolutional layers.
            activation: nn.Module. Activation function to use.

            Returns
            -------
            None.
        """

        # Class inheritance
        super().__init__()

        # Components
        self.conv1 = nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.relu = activation
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(n_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Pass forward through residual block.

            Parameters
            ----------
            x: tensor. Inputs.

            Returns
            -------
            y: tensor. Outputs.
        """

        # Sequence of operations
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class BaseModel(LightningModule):

    def __init__(self, model: nn.Module, optimizer: Callable = torch.optim.Adam, loss_func: Callable = nn.MSELoss(),
                 log_valid: bool = False) \
            -> None:
        """ Initialize base neural network model. Enables class inheritance.

            Parameters
            ----------
            model: nn. Neural network architecture.
            optimizer: Callable (partially instantiated). Choice of optimizer and corresponding parameters.
            loss_func: Callable (partially instantiated). Loss function.
            log_valid: bool; default=False. Flag to log validation metrics.

            Returns
            -------
            None.
        """

        # Class inheritance
        super().__init__()
        # Neural network architecture
        self.model = model
        # Optimizer initialization
        self.optimizer = optimizer
        # Loss function
        self.loss_func = loss_func

        # Test set results
        self.test_pred = []
        # Validation set results
        if log_valid:
            self.valid_pred = []
            self.valid_target = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Pass forward through neural network architecture.

            Parameters
            ----------
            x: tensor. Inputs.

            Returns
            -------
            y: tensor. Outputs.
        """
        return self.model(x)

    def base_step(self, batch: torch.Tensor, batch_nb: int, stage: str) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """ Perform training/validation/test step.

            Parameters
            ----------
            batch: tensor. Batch from the training set.
            batch_nb: int. Index of the batch out of the training set.
            stage: str. Current operation: "train", "valid", or "test".

            Returns
            -------
            Loss value: tensor.
        """

        # Extract data from batch
        x, y = batch
        # Forward pass
        y_pred = self(x)
        # Compute loss function
        loss = self.loss_func(y_pred, y)

        # Compute metrics (for diagnostic purposes)
        epsilon = sys.float_info.min
        # Mean relative absolute error
        rae = torch.nanmean(torch.abs((y - y_pred) / (torch.abs(y) + epsilon)) * 100)
        # Mean absolute error
        mae = torch.nanmean(torch.abs(y - y_pred))

        # Log metrics
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{stage}_MAE", mae, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{stage}_RAE", rae, on_epoch=True, prog_bar=True, logger=True)

        # For test set...
        if stage == 'test':
            # Store test outputs
            self.test_results.append(y_pred)
        # For validation set...
        elif stage == 'valid':
            # Store validation outputs and targets
            self.valid_pred.append(y_pred)
            self.valid_target.append(y)

        return loss

    def training_step(self, batch: torch.Tensor, batch_nb: int) -> torch.Tensor:
        """ Perform training step.

            Parameters
            ----------
            batch: tensor. Batch from the training set.
            batch_nb: int. Index of the batch out of the training set.

            Returns
            -------
            Loss value: tensor.
        """

        return self.base_step(batch, batch_nb, stage='train')

    def validation_step(self, batch: torch.Tensor, batch_nb: int) -> torch.Tensor:
        """ Perform validation step.

            Parameters
            ----------
            batch: tensor. Batch from the validation set.
            batch_nb: int. Index of the batch out of the validation set.

            Returns
            -------
            Loss value: tensor.
        """

        return self.base_step(batch, batch_nb, stage='valid')

    def test_step(self, batch: torch.Tensor, batch_nb: int) -> torch.Tensor:
        """ Perform test step.

            Parameters
            ----------
            batch: tensor. Batch from the test set.
            batch_nb: int. Index of the batch out of the test set.

            Returns
            -------
            Loss value: tensor.
        """

        return self.base_step(batch, batch_nb, stage='test')

    def on_validation_epoch_start(self) -> None:
        """ Perform validation epoch start.

            Parameters
            ----------
            None.

            Returns
            -------
            None.
        """

        # Aggregate validation results and convert to numpy array
        self.valid_pred = []
        self.valid_target = []

    def on_validation_epoch_end(self) -> None:
        """ Perform validation epoch end.

            Parameters
            ----------
            None.

            Returns
            -------
            None.
        """

        # Clear the lists for the next epoch
        if self.log_valid:
            self.valid_pred.clear()
            self.valid_target.clear()

    def on_test_epoch_start(self) -> None:
        """ Perform test epoch start.

            Parameters
            ----------
            None.

            Returns
            -------
            None.
        """

        # Aggregate test results and convert to numpy array
        self.test_pred = []

    def on_test_epoch_end(self) -> None:
        """ Perform test epoch end.

            Parameters
            ----------
            None.

            Returns
            -------
            None.
        """

        # Aggregate test results and convert to numpy array
        self.test_pred = torch.cat(self.test_pred).cpu().numpy()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """ Instantiate optimizer.

            Parameters
            ----------
            None. Target and parameters are passed from self.optmizer_config.

            Returns
            -------
            Optimizer instance.
        """

        # Instantiate from config object
        return self.optimizer(self.parameters())


class DeepVelModel(BaseModel):
    """
    DeepVel neural network model (Asensio Ramos et al., 2017).
    """
    def __init__(self, n_in_channels: int, n_out_channels: int, n_filters: int = 64, kernel_size: int = 3,
                 n_conv_layers: int = 20, stride: int = 1, padding: int = None, activation: nn.Module = nn.ReLU(),
                 optimizer: Callable = torch.optim.Adam, loss_func: Callable = nn.MSELoss()) -> None:
        """ Initialize DeepVel neural network model.

            Parameters
            ----------
            n_in_channels: int. Number of input channels.
            n_out_channels: int. Number of output channels.
            n_filters: int. Number of filters in the convolutional layers.
            kernel_size: int. Size of the convolutional kernel.
            n_conv_layers: int. Number of convolutional layers in residual block.
            stride: int. Stride of the convolutional layers.
            padding: int. Padding for the convolutional layers.
            activation: nn.Module. Activation function to use.
            optimizer: DictConfig. Choice of optimizer and corresponding parameters.
            loss_func: DictConfig. Loss function.

            Returns
            -------
            None.
        """

        # Padding
        if padding is None:
            padding = same_padding(kernel_size, stride)

        # Residual blocks
        residuals = [ResidualBlock(n_filters, kernel_size=kernel_size, stride=stride,
                                   padding=padding, activation=activation)] * n_conv_layers
        # Complete model
        model = nn.Sequential(
            nn.Conv2d(n_in_channels, n_filters, kernel_size=kernel_size, stride=stride, padding=padding),
            activation,
            *residuals,
            nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(n_filters),
            nn.Conv2d(n_filters, n_out_channels, kernel_size=1, stride=stride, padding=padding))

        # Class inheritance
        super().__init__(model=model, optimizer=optimizer, loss_func=loss_func)


class DeepVelUModel(BaseModel):
    """
    DeepVelU neural network model (Tremblay & Attié, 2020).
    """
    def __init__(self, n_in_channels: int, n_out_channels: int, n_filters: int = 64, kernel_size: int = 3,
                 n_conv_layers: int = 20, stride: int = 1, padding: int = None, activation: nn.Module = nn.ReLU(),
                 optimizer: Callable = torch.optim.Adam, loss_func: Callable = nn.MSELoss()) -> None:
        """ Initialize DeepVelU neural network model.

            Parameters
            ----------
            n_in_channels: int. Number of input channels.
            n_out_channels: int. Number of output channels.
            n_filters: int. Number of filters in the convolutional layers.
            kernel_size: int. Size of the convolutional kernel.
            n_conv_layers: int. Number of convolutional layers in residual block.
            stride: int. Stride of the convolutional layers.
            padding: int. Padding for the convolutional layers.
            activation: nn.Module. Activation function to use.
            optimizer: DictConfig. Choice of optimizer and corresponding parameters.
            loss_func: DictConfig. Loss function.

            Returns
            -------
            None.
        """

        # Padding
        if padding is None:
            padding = same_padding(kernel_size, stride)

        # Architecture
        model = None

        # Class inheritance
        super().__init__(model=model, optimizer=optimizer, loss_func=loss_func)


class DeeperVelModel(BaseModel):
    """
    DeeperVel neural network model (Tremblay & Rempel, in prep.).
    """
    def __init__(self, n_in_channels: int, n_out_channels: int, n_filters: int = 64, kernel_size: int = 3,
                 n_conv_layers: int = 20, stride: int = 1, padding: int = None, activation: nn.Module = nn.ReLU(),
                 optimizer: Callable = torch.optim.Adam, loss_func: Callable = nn.MSELoss()) -> None:
        """ Initialize DeeperVel neural network model.

            Parameters
            ----------
            n_in_channels: int. Number of input channels.
            n_out_channels: int. Number of output channels.
            n_filters: int. Number of filters in the convolutional layers.
            kernel_size: int. Size of the convolutional kernel.
            n_conv_layers: int. Number of convolutional layers in residual block.
            stride: int. Stride of the convolutional layers.
            padding: int. Padding for the convolutional layers.
            activation: nn.Module. Activation function to use.
            optimizer: DictConfig. Choice of optimizer and corresponding parameters.
            loss_func: DictConfig. Loss function.

            Returns
            -------
            None.
        """

        # Padding
        if padding is None:
            padding = same_padding(kernel_size, stride)

        # Residual blocks
        residuals = [ResidualBlock(n_filters, kernel_size=kernel_size, stride=stride,
                                   padding=padding, activation=activation)] * n_conv_layers
        # Complete model
        model = nn.Sequential(
            nn.Conv2d(n_in_channels, n_filters, kernel_size=4*kernel_size, stride=stride, padding=padding),
            nn.Conv2d(n_filters, n_filters, kernel_size=2*kernel_size, stride=stride, padding=padding),
            activation,
            *residuals,
            nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(n_filters),
            nn.Conv2d(n_filters, n_out_channels, kernel_size=1, stride=stride, padding=padding))

        # Class inheritance
        super().__init__(model=model, optimizer=optimizer, loss_func=loss_func)


import sys
import torch
from typing import Callable, Tuple, Union, Any
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
import torch.nn as nn
import numpy as np
from track.utilities.instantiators import instantiate


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
                 activation: Callable = None) -> None:
        """ Initialize residual block.

            Parameters
            ----------
            n_filters: int. Number of filters in the convolutional layers.
            kernel_size: int. Size of the convolutional kernel.
            stride: int. Stride of the convolutional layers.
            padding: int. Padding for the convolutional layers.
            activation: nn.Module. Activation function to use. If None, ReLU is used.

            Returns
            -------
            None.
        """

        # Class inheritance
        super().__init__()

        # Components
        self.conv1 = nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.relu1 = instantiate(activation) if activation else nn.ReLU()
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.relu2 = instantiate(activation) if activation else nn.ReLU()

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
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu2(out)
        return out


class BaseModel(LightningModule):

    def __init__(self, optimizer: Callable = None, lr_scheduler: Callable = None,
                 loss_func: Callable = None, log_valid: bool = False) \
            -> None:
        """ Initialize the base neural network model. Enables class inheritance.

            Parameters
            ----------
            optimizer: Callable. Choice of optimizer and corresponding parameters.
            lr_scheduler: Callable. Choice of learning rate scheduler and corresponding parameters.
            loss_func: Callable. Loss function to use.
            log_valid: bool; default=False. Flag to log validation metrics.

            Returns
            -------
            None.
        """

        # Class inheritance
        super().__init__()

        # Optimizer initialization
        self.optimizer = optimizer
        # Learning rate scheduler
        self.lr_scheduler = lr_scheduler
        # Loss function
        self.loss_func = loss_func
        # Store hyperparameters
        self.save_hyperparameters(ignore=['optimizer', 'lr_scheduler', 'loss_func'])

        # Test set results
        self.test_result = []
        # Validation set results
        self.log_valid = log_valid
        if self.log_valid:
            self.valid_result = []
            self.valid_target = []

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
        rae = torch.nanmedian(torch.abs((y - y_pred) / (torch.abs(y) + epsilon)) * 100)
        # Mean absolute error
        mae = torch.nanmean(torch.abs(y - y_pred))

        # Log metrics
        if stage in ['train', 'valid']:
            self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"{stage}_MAE", mae, on_epoch=True, prog_bar=False, logger=True)
            self.log(f"{stage}_RAE", rae, on_epoch=True, prog_bar=False, logger=True)

        # For test set...
        if stage == 'test':
            # Store test outputs
            self.test_result.append(y_pred.detach().cpu().numpy())
        # For validation set...
        elif stage == 'valid':
            # Store validation outputs and targets
            self.valid_result.append(y_pred.detach().cpu().numpy())
            self.valid_target.append(y.detach().cpu().numpy())

        return loss

    def training_step(self, batch: torch.Tensor, batch_nb: int) -> torch.Tensor:
        """ Perform the training step.

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
        """ Perform the validation step.

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
        """ Perform the test step.

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

        # Clear the lists for the next epoch
        self.valid_result = []
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
            self.valid_result.clear()
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

        # Clear the list for the next epoch
        self.test_result.clear()

    def on_test_epoch_end(self) -> None:
        """ Perform the test epoch end.

            Parameters
            ----------
            None.

            Returns
            -------
            None.
        """

        # Aggregate test results
        self.test_result = np.concatenate(self.test_result, axis=0)

    def configure_optimizers(self) -> Union[dict[str, Union[torch.optim.Optimizer, dict[str, Any]]], None]:
        """ Instantiate optimizer.

            Parameters
            ----------
            None. Target and parameters are passed from self.optmizer_config.

            Returns
            -------
            Optimizer instance.
        """

        if self.optimizer is not None:

            # Instantiate optimizer
            optimizer = instantiate(self.optimizer, params=self.parameters())

            # Check if learning rate scheduler is defined
            if self.lr_scheduler is not None:

                # Instantiate learning rate scheduler
                lr_scheduler = instantiate(self.lr_scheduler, optimizer=optimizer)

                # Check if the learning rate scheduler is specifically reducing on plateau
                reduce_on_plateau = isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
                print('Reduce on plateau:', reduce_on_plateau)

                # Instantiate from the config object
                return {'optimizer': optimizer,
                        'lr_scheduler': {'scheduler': lr_scheduler,
                                         'interval': 'epoch',
                                         'monitor': 'valid_loss',
                                         'frequency': 1,
                                         'reduce_on_plateau': reduce_on_plateau,
                                         }
                        }
            return optimizer
        return None


class DeepVelModel(BaseModel):
    """
    DeepVel neural network model (Asensio Ramos et al., 2017).
    """
    def __init__(self, n_in_channels: int, n_out_channels: int, n_filters: int = 64, kernel_size: int = 3,
                 n_conv_layers: int = 20, stride: int = 1, padding: int = None, activation: Callable = None,
                 optimizer: Callable = None, lr_scheduler: Callable = None, loss_func: Callable = None,
                 log_valid: bool = False) -> None:
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
            activation: Callable. Activation function of the convolutional layers.
            optimizer: Callable. Choice of optimizer and corresponding parameters.
            lr_scheduler: Callable. Choice of learning rate scheduler and corresponding parameters.
            loss_func: Callable. Loss function to use.
            log_valid: bool; default=False. Flag to log validation metrics.

            Returns
            -------
            None.
        """

        # Class inheritance
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler, loss_func=loss_func, log_valid=log_valid)

        # Padding
        if padding is None:
            padding = same_padding(kernel_size, stride)

        # Residual blocks
        residuals = [ResidualBlock(n_filters, kernel_size=kernel_size, stride=stride, padding=padding,
                                   activation=activation) for _ in range(n_conv_layers)]
        # Complete model
        self.model = nn.Sequential(
            nn.Conv2d(n_in_channels, n_filters, kernel_size=kernel_size, stride=stride, padding=padding),
            instantiate(activation) if activation else nn.ReLU(),
            *residuals,
            nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(n_filters),
            nn.Conv2d(n_filters, n_out_channels, kernel_size=kernel_size, stride=stride, padding=padding))

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


class DeepVelUModel(BaseModel):
    """
    DeepVelU neural network model (Tremblay & Attié, 2020).
    """
    def __init__(self, n_in_channels: int, n_out_channels: int, n_filters: int = 64, kernel_size: int = 3,
                 depth: int = 3, dropout: float = 0.5, activation: Callable = None, optimizer: Callable = None,
                 lr_scheduler: Callable = None, loss_func: Callable = None, log_valid: bool = False) -> None:
        """ Initialize DeepVelU neural network model.

            Parameters
            ----------
            n_in_channels: int. Number of input channels.
            n_out_channels: int. Number of output channels.
            n_filters: int. Number of filters in the convolutional layers.
            kernel_size: int. Size of the convolutional kernel.
            depth: int. Depth of the U-Net architecture.
            dropout: float. Dropout rate for the convolutional layers.
            activation: Callable. Activation function of the convolutional layers.
            optimizer: Callable. Choice of optimizer and corresponding parameters.
            lr_scheduler: Callable. Choice of learning rate scheduler and corresponding parameters.
            loss_func: Callable. Loss function to use.
            log_valid: bool; default=False. Flag to log validation metrics.

            Returns
            -------
            None.
        """

        # Class inheritance
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler, loss_func=loss_func, log_valid=log_valid)

        # Depth of the U-Net architecture
        self.depth = depth
        # Layers for each block
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        # Upsampling layer
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Down block
        in_ch = n_in_channels
        for i in range(depth):
            # Output channels for the current block
            out_ch = n_filters if i == 0 else n_filters * 2
            # Create down block
            block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=kernel_size // 2),
                nn.BatchNorm2d(out_ch),
                instantiate(activation, _partial_=True) if activation is not None else nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv2d(out_ch, out_ch, kernel_size, stride=2, padding=kernel_size // 2),
                nn.BatchNorm2d(out_ch),
                instantiate(activation, _partial_=True) if activation is not None else nn.ReLU(inplace=True),
            )
            # Store the block
            self.down_blocks.append(block)
            # Update input channels for the next block
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_ch, in_ch*2, kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm2d(in_ch*2),
            instantiate(activation, _partial_=True) if activation is not None else nn.ReLU(inplace=True),
            nn.Conv2d(in_ch*2, in_ch*2, kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm2d(in_ch*2),
            instantiate(activation, _partial_=True) if activation is not None else nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Up path
        up_in_ch = in_ch*2  # after bottleneck
        down_channels = [n_filters if i == 0 else n_filters * 2 for i in range(depth)]
        for i in range(depth):
            # Skip channels
            skip_ch = down_channels[-(i + 1)]
            # Output channels for the current block
            out_ch = n_filters if i == depth-1 else n_filters * 2
            # Create up block
            block = nn.Sequential(
                nn.Conv2d(up_in_ch + skip_ch, out_ch, kernel_size, stride=1, padding=kernel_size//2),
                nn.BatchNorm2d(out_ch),
                instantiate(activation, _partial_=True) if activation is not None else nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size, stride=1, padding=kernel_size//2),
                nn.BatchNorm2d(out_ch),
                instantiate(activation, _partial_=True) if activation is not None else nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            # Store the block
            self.up_blocks.append(block)
            # Update input channels for the next block
            up_in_ch = out_ch

        # Final convolution layer
        self.final_conv = nn.Conv2d(n_filters, n_out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Pass forward through neural network architecture.

            Parameters
            ----------
            x: tensor. Inputs.

            Returns
            -------
            y: tensor. Outputs.
        """

        # Initialize skips and output tensor
        skips = []
        out = x

        # Down path
        for block in self.down_blocks:
            out = block(out)
            skips.append(out)

        # Bottleneck
        out = self.bottleneck(out)

        # Up path
        for i, block in enumerate(self.up_blocks):
            out = self.upsample(out)
            out = torch.cat([out, skips[-(i+1)]], dim=1)
            out = block(out)
        # Final convolution layer
        return self.final_conv(out)


class DeeperVelModel(BaseModel):
    """
    DeeperVel neural network model (Tremblay & Rempel, in prep.).
    """
    def __init__(self, n_in_channels: int, n_out_channels: int, n_filters: int = 64, kernel_size: int = 3,
                 n_conv_layers: int = 20, stride: int = 1, padding: int = None, activation: Callable = None,
                 optimizer: Callable = None, lr_scheduler: Callable = None, loss_func: Callable = None,
                 log_valid: bool = False) -> None:
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
            activation: Callable. Activation function of the convolutional layers.
            optimizer: Callable. Choice of optimizer and corresponding parameters.
            lr_scheduler: Callable. Choice of learning rate scheduler and corresponding parameters.
            loss_func: Callable. Loss function to use.
            log_valid: bool; default=False. Flag to log validation metrics.

            Returns
            -------
            None.
        """

        # Class inheritance
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler, loss_func=loss_func, log_valid=log_valid)

        # Padding
        if padding is None:
            padding = same_padding(kernel_size, stride)

        # Residual blocks
        residuals = [ResidualBlock(n_filters, kernel_size=kernel_size, stride=stride,
                                   padding=padding, activation=activation) for _ in range(n_conv_layers)]
        # Complete model
        self.model = nn.Sequential(
            nn.Conv2d(n_in_channels, n_filters, kernel_size=4*kernel_size, stride=stride, padding=padding),
            nn.Conv2d(n_filters, n_filters, kernel_size=2*kernel_size, stride=stride, padding=padding),
            instantiate(activation, _partial_=True) if activation else nn.ReLU(),
            *residuals,
            nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(n_filters),
            nn.Conv2d(n_filters, n_out_channels, kernel_size=1, stride=stride, padding=padding))

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

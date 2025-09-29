from typing import Union, Dict
import numpy as np
import torch


def broadcast(var1: Union[np.ndarray, torch.Tensor], var2: Union[np.ndarray, torch.Tensor]) \
        -> Union[np.ndarray, torch.Tensor]:
    """ Broadcast var2 to the same dimensions as var1.
        Works for both numpy arrays and torch tensors.

        Parameters
        ----------
        var1: numpy arr or torch tensor. Reference.
        var2: numpy arr or torch tensor. Variable to change dimensions of.

        Returns
        -------
        var2: Broadcast to the same number of dimensions as var1.
    """

    # Handle Python float/int or NumPy scalar
    if isinstance(var2, (float, int, np.floating, np.integer)):
        if isinstance(var1, torch.Tensor):
            var2 = torch.tensor(var2, dtype=var1.dtype, device=var1.device)
        elif isinstance(var1, np.ndarray):
            var2 = np.array(var2, dtype=var1.dtype)
    # For Numpy arrays
    elif isinstance(var2, np.ndarray):
        var2 = np.reshape(var2, (1,) * (var1.ndim - var2.ndim) + var2.shape)
        # Convert to torch tensor if var1 is a torch tensor
        if isinstance(var1, torch.Tensor):
            var2 = torch.from_numpy(var2).to(var1.device, dtype=var1.dtype)
    # For Torch tensors
    elif isinstance(var2, torch.Tensor):
        var2 = var2.view((1,) * (var1.ndim - var2.ndim) + var2.shape)
        # Convert to numpy if var1 is a numpy array
        if isinstance(var1, np.ndarray):
            var2 = var2.cpu().numpy()
        else:
            var2 = var2.to(var1.device, dtype=var1.dtype)
    return var2


def multiplication(data: Union[np.ndarray, torch.Tensor], factor: Union[np.ndarray, torch.Tensor],
                   inverse_transform: bool = False) \
        -> Union[np.ndarray, torch.Tensor]:
    """ Multiply dataset by a factor.

        Parameters
        ----------
        data: arr or tensor. Contains data to transform.
        factor: arr or tensor. Statistics of the data.
        inverse_transform: bool. False for dividing, True for multiplying.

        Returns
        -------
        data_transform: arr or tensor. Scaled dataset.
    """

    # Broadcast to data dimensions
    scaling_factor = broadcast(data, factor)

    # Unstandardization or standardization
    if inverse_transform:
        # Divide
        eps = np.finfo(scaling_factor.dtype).eps if isinstance(scaling_factor, np.ndarray) \
            else torch.finfo(scaling_factor.dtype).eps
        data_transform = data / (scaling_factor + eps)  # Avoid division by zero
    else:
        # Multiply
        data_transform = data * scaling_factor

    return data_transform


def translation(data: Union[np.ndarray, torch.Tensor], value: Union[np.ndarray, torch.Tensor],
                inverse_transform: bool = False) \
        -> Union[np.ndarray, torch.Tensor]:
    """ Shift dataset by adding a value.

        Parameters
        ----------
        data: arr or tensor. Contains data to transform.
        value: arr or tensor. Shift to be added to the data.
        inverse_transform: bool. False for subtracting, True for adding.

        Returns
        -------
        data_transform: arr or tensor. Translated dataset.
    """

    # Broadcast to data dimensions
    shift_value = broadcast(data, value)

    # Unstandardization or standardization
    if inverse_transform:
        # Subtract
        data_transform = data - shift_value
    else:
        # Add
        data_transform = data + shift_value

    return data_transform


def affine(data: Union[np.ndarray, torch.Tensor], factor: Union[np.ndarray, torch.Tensor],
           value: Union[np.ndarray, torch.Tensor], inverse_transform: bool = False) \
        -> Union[np.ndarray, torch.Tensor]:
    """ Affine transformation (linear scaling): data_transform = factor*data + value.

        Parameters
        ----------
        data: arr or tensor. Contains data to transform.
        factor: arr or tensor. Multiply the data by a factor.
        value: arr or tensor. Shift to be added to the data.
        inverse_transform: bool. False for subtracting, True for adding.

        Returns
        -------
        data_transform: arr or tensor. Translated dataset.
    """

    # Affine transformation
    if inverse_transform:
        # data_transform = (data - value)/factor
        data_transform = multiplication(translation(data, value, inverse_transform=inverse_transform),
                                        factor, inverse_transform=inverse_transform)
    else:
        # data_transform = factor*data + value
        data_transform = translation(multiplication(data, factor), value)

    return data_transform


def stdev(data: Union[np.ndarray, torch.Tensor], stats: Dict, inverse_transform: bool = False) \
        -> Union[np.ndarray, torch.Tensor]:
    """ Divide/multiply dataset by stddev.

        Parameters
        ----------
        data: arr or tensor. Contains data to transform.
        stats: arr or tensor. Statistics of the data.
        inverse_transform: bool. False for standardization, True for unstandardization.

        Returns
        -------
        data_transform: arr or tensor. Standardized/unstandardized dataset.
    """

    return multiplication(data, stats['stdev'], inverse_transform=not inverse_transform)


def mean_stdev(data: Union[np.ndarray, torch.Tensor], stats: Dict, inverse_transform: bool = False) \
        -> Union[np.ndarray, torch.Tensor]:
    """ Standardize dataset.

        Parameters
        ----------
        data: arr or tensor. Contains data to transform.
        stats: arr or tensor. Statistics of the data.
        inverse_transform: bool. False for standardization, True for unstandardization.

        Returns
        -------
        data_transform: arr or tensor. Standardized/unstandardized dataset.
    """

    return affine(data, stats['stdev'], stats['mean'], inverse_transform=not inverse_transform)


def min_max(data: Union[np.ndarray, torch.Tensor], stats: Dict, inverse_transform: bool = False) \
        -> Union[np.ndarray, torch.Tensor]:
    """ Normalize dataset.

        Parameters
        ----------
        data: arr or tensor. Contains data to transform.
        stats: arr or tensor. Statistics of the data.
        inverse_transform: bool. False for normalization, True for unnormalization.

        Returns
        -------
        data_transform: arr or tensor. Normalized/unnormalized dataset.
    """

    return affine(data, stats['max']-stats['min'], stats['min'], inverse_transform=not inverse_transform)


def median(data: Union[np.ndarray, torch.Tensor], stats: Dict, inverse_transform: bool = False) \
           -> Union[np.ndarray, torch.Tensor]:
    """ Divide dataset by its median.

        Parameters
        ----------
        data: arr or tensor. Contains data to transform.
        stats: arr or tensor. Statistics of the data.
        inverse_transform: bool. False for normalization, True for unnormalization

        Returns
        -------
        data_transform: arr or tensor. Transformed dataset.
    """

    return multiplication(data, stats['median'], inverse_transform=not inverse_transform)


def identity(data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """ Identity transformation.

        Parameters
        ----------
        data: arr or tensor. Contains data to transform.

        Returns
        -------
        data_transform: arr or tensor. Unchanged dataset.
    """

    return data


def rot90(data: Union[np.ndarray, torch.Tensor], n: int = 1, axes: Union[int, tuple[int, int]] = 1,
          inverse_transform: bool = False) -> Union[np.ndarray, torch.Tensor]:
    """ Rotate dataset by multiples of 90 degrees.

        Parameters
        ----------
        data: arr or tensor. Contains data to transform.
        n: int. Number of 90 degree rotations.
        axes: int or tuple. Axes to rotate.
            - int: Rotate along the first two axes.
            - tuple: Rotate along the specified axes.
        inverse_transform: bool. False for rotation, True for unrotation.

        Returns
        -------
        data_transform: arr or tensor. Rotated dataset.
    """

    # For Cartesian: positive n_rot90 means CW
    k = -n if not inverse_transform else n

    # Apply rotation
    if isinstance(data, np.ndarray):
        # Numpy
        data_transform = np.rot90(data, k=k, axes=axes)
    elif isinstance(data, torch.Tensor):
        # Torch
        data_transform = torch.rot90(data, k=k, dims=axes)
    else:
        raise TypeError("Unsupported data type. Expected numpy array or torch tensor.")

    return data_transform


def flip(data: Union[np.ndarray, torch.Tensor], n: Union[int, tuple[int]] = None,
         inverse_transform: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """ Flip dataset along an axis.

            Parameters
            ----------
            data: arr or tensor. Contains data to transform.
            n: int. Axis to flip along.
            inverse_transform: bool. False for flipping, True for unflipping.

            Returns
            -------
            data_transform: arr or tensor. Flipped dataset.
        """

        # No flip
        if n is None:
            return data
        # Apply flip
        elif isinstance(data, np.ndarray):
            # Numpy
            data_transform = np.flip(data, n)
        elif isinstance(data, torch.Tensor):
            # Torch
            data_transform = torch.tensor(np.flip(data.cpu().numpy(), n), dtype=data.dtype)
        else:
            raise TypeError("Unsupported data type. Expected numpy array or torch tensor.")

        return data_transform


def augment_scalar(data: Union[np.ndarray, torch.Tensor], n_rot90: int = 0, axes_rot90: Union[int, tuple[int, int]] = 1,
                   n_flip: int = None, inverse_transform: bool = False) -> Union[np.ndarray, torch.Tensor]:
    """ Apply random rotation and flip to the dataset.

        Parameters
        ----------
        data: arr or tensor. Contains data to transform.
        n_rot90: int. Number of 90 degree rotations.
        axes_rot90: int or tuple. Axes to rotate.
        n_flip: int. Axis to flip along.
        inverse_transform: bool. False for augmentation, True for unaugmentation.

        Returns
        -------
        data_transform: arr or tensor. Transformed dataset.
    """

    # Apply flip
    data_transform = flip(data, n=n_flip, inverse_transform=inverse_transform)
    # Apply rotation
    data_transform = rot90(data_transform, n=n_rot90, axes=axes_rot90, inverse_transform=inverse_transform)

    return data_transform


def augment_vector(data_x: Union[np.ndarray, torch.Tensor], data_y: Union[np.ndarray, torch.Tensor],
                   n_rot90: int = 0, n_flip: int = None, axes_rot90: Union[int, tuple[int, int]] = 1,
                   inverse_transform: bool = False) \
        -> tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """ Perform all possible augmentations of a vector, accounting for directionality
        (positive and negative signs based on the direction).

        Parameters
        ----------
        data_x: arr or tensor. Contains x-component of the vector.
        data_y: arr or tensor. Contains y-component of the vector.
        n_rot90: int. Number of 90 degree rotations.
        axes_rot90: int or tuple. Axes to rotate.
        n_flip: int. Axis to flip along.
        inverse_transform: bool. False for augmentation, True for unaugmentation.

        Returns
        -------
        data_transform: arr or tensor. Transformed dataset.
    """

    # Apply transformations for both x and y components.
    # Account for the changes in sign, and the fact that the x and y components are swapped
    # when rotating by 90 degrees.
    if n_flip is not None:
        data_x = flip(data_x, n=n_flip, inverse_transform=inverse_transform)
        data_y = flip(data_y, n=n_flip, inverse_transform=inverse_transform)
        if n_flip == 0:
            data_y = -data_y
        elif n_flip == 1:
            data_x = -data_x

    # Apply rotation and account for the swapping of x and y components
    if n_rot90 == 0:
        return data_x, data_y
    elif n_rot90 == 1:
        return -rot90(data_y, n=n_rot90, axes=axes_rot90), rot90(data_x, n=n_rot90, axes=axes_rot90)
    elif n_rot90 == 2:
        return -rot90(data_x, n=n_rot90, axes=axes_rot90), -rot90(data_y, n=n_rot90, axes=axes_rot90)
    elif n_rot90 == 3:
        return rot90(data_y, n=n_rot90, axes=axes_rot90), -rot90(data_x, n=n_rot90, axes=axes_rot90)
    else:
        raise ValueError(f"Unsupported number of rotations: {n_rot90}. Must be in [0, 3].")

def geometric_augmentation(data: Union[np.ndarray, torch.Tensor, tuple[np.ndarray, np.ndarray], tuple[torch.Tensor, torch.Tensor]],
                           vars: list[str], n_rot90: int = 0, axes_rot90: Union[int, tuple[int, int]] = 1, n_flip: int = None) \
        -> Union[np.ndarray, torch.Tensor, tuple[np.ndarray, np.ndarray], tuple[torch.Tensor, torch.Tensor]]:
    """ Apply geometric augmentation to the dataset.

        Parameters
        ----------
        data: arr or tensor. Contains data to transform.
        vars: list of str. Variables to apply augmentation to.
        n_rot90: int. Number of 90 degree rotations.
        axes_rot90: int or tuple. Axes to rotate.
        n_flip: int. Axis to flip along.

        Returns
        -------
        data_transform: arr or tensor. Transformed dataset.
    """

    # Vector augmentation
    for vpair in [("vx", "vy"), ("Bx", "By")]:
        if all(v in vars for v in vpair):
            idx_x, idx_y = vars.index(vpair[0]), vars.index(vpair[1])
            data[..., idx_x], data[..., idx_y] = augment_vector(data[..., idx_x], data[..., idx_y],
                                                                n_flip=n_flip, n_rot90=n_rot90, axes_rot90=axes_rot90)
    # Scalar augmentation
    for scalar in ["I500", "vz", "Bz"]:
        if scalar in vars:
            idx = vars.index(scalar)
            data[..., idx] = augment_scalar(data[..., idx], n_flip=n_flip, n_rot90=n_rot90, axes_rot90=axes_rot90)
    return data

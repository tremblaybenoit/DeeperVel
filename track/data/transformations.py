from typing import Union, Dict
import numpy as np
import torch


def broadcast(var1: Union[np.ndarray, torch.Tensor], var2: Union[np.ndarray, torch.Tensor, float, int]) \
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
    if isinstance(var2, (float, int, np.generic)):
        if isinstance(var1, torch.Tensor):
            return torch.tensor(var2, dtype=var1.dtype, device=var1.device)
        else:
            return np.array(var2, dtype=var1.dtype)
    # If var1 is a numpy array
    elif isinstance(var1, np.ndarray):
        # If var2 is a tensor
        if isinstance(var2, torch.Tensor):
            if var2.device.type != 'cpu':
                var2 = var2.detach().cpu().numpy()
        # Else var 2 is a numpy array
        # Ensure same dtype
        dtype = np.result_type(var1.dtype, var2.dtype)
        if var2.dtype != dtype:
            var2 = var2.astype(dtype, copy=False)
        # Broadcast dimensions
        return np.broadcast_to(var2, var1.shape)
    # If var1 is a torch tensor
    elif isinstance(var1, torch.Tensor):
        # If var2 is a numpy array
        if isinstance(var2, np.ndarray):
            var2 = torch.as_tensor(var2, device=var1.device)
        else:
            var2 = var2.to(var1.device)
        # Else var2 is a torch tensor
        # Ensure same dtype and device
        dtype = torch.promote_types(var1.dtype, var2.dtype)
        if var2.dtype != dtype:
            var2 = var2.to(dtype=dtype)
        # Broadcast dimensions
        if var2.ndim < var1.ndim:
            view_shape = [1] * (var1.ndim - var2.ndim) + list(var2.shape)
            var2 = var2.view(view_shape)
        return var2.expand(var1.shape)
    else:
        raise TypeError("Input variables must be numpy arrays or torch tensors.")


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
        # Avoid division by zero
        eps = np.finfo(scaling_factor.dtype).eps if isinstance(scaling_factor, np.ndarray) \
            else torch.finfo(scaling_factor.dtype).eps
        denom = scaling_factor + eps
        # Divide
        return data.div_(denom) if isinstance(data, torch.Tensor) else np.divide(data, denom, out=data)
    else:
        # Multiply
        return data.mul_(scaling_factor) if isinstance(data, torch.Tensor) \
            else np.multiply(data, scaling_factor, out=data)


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
        return data.sub_(shift_value) if isinstance(data, torch.Tensor) else np.subtract(data, shift_value, out=data)
    else:
        return data.add_(shift_value) if isinstance(data, torch.Tensor) else np.add(data, shift_value, out=data)


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
        data = translation(data, value, inverse_transform=inverse_transform)
        return multiplication(data, factor, inverse_transform=inverse_transform)
    else:
        # data_transform = factor*data + value
        data = multiplication(data, factor, inverse_transform=inverse_transform)
        return translation(data, value, inverse_transform=inverse_transform)


def stdev(data: Union[np.ndarray, torch.Tensor], stats: Dict, inverse_transform: bool = False, axis: int=None) \
        -> Union[np.ndarray, torch.Tensor]:
    """ Divide/multiply dataset by stddev.

        Parameters
        ----------
        data: arr or tensor. Contains data to transform.
        stats: arr or tensor. Statistics of the data.
        inverse_transform: bool. False for standardization, True for unstandardization.
        axis: int or None. Axis along which to standardize. If None, standardize across all dimensions.

        Returns
        -------
        data_transform: arr or tensor. Standardized/unstandardized dataset.
    """
    if axis is None:
        return multiplication(data, stats['stdev'], inverse_transform=not inverse_transform)
    else:
        acc_mean = stats['mean'].mean(axis=axis, keepdims=True)
        acc_var = (stats['stdev'] ** 2 + (stats['mean'] - acc_mean) ** 2).mean(axis=axis, keepdims=True)
        acc_stdev = np.sqrt(acc_var)
        return multiplication(data, acc_stdev, inverse_transform=not inverse_transform)


def mean_stdev(data: Union[np.ndarray, torch.Tensor], stats: Dict, inverse_transform: bool = False, axis: int=None) \
        -> Union[np.ndarray, torch.Tensor]:
    """ Standardize dataset.

        Parameters
        ----------
        data: arr or tensor. Contains data to transform.
        stats: arr or tensor. Statistics of the data.
        inverse_transform: bool. False for standardization, True for unstandardization.
        axis: int or None. Axis along which to standardize. If None, standardize across all dimensions.

        Returns
        -------
        data_transform: arr or tensor. Standardized/unstandardized dataset.
    """
    if axis is None:
        return affine(data, stats['stdev'], stats['mean'], inverse_transform=not inverse_transform)
    else:
        # acc_stats = [{'mean': stats['mean'], 'stdev': stats['stdev'], 'n_samples': 1}]
        acc_mean = stats['mean'].mean(axis=axis, keepdims=True)
        acc_var = (stats['stdev']**2 + (stats['mean'] - acc_mean)**2).mean(axis=axis, keepdims=True)
        acc_stdev = np.sqrt(acc_var)
        return affine(data, acc_stdev, acc_mean, inverse_transform=not inverse_transform)


def min_max(data: Union[np.ndarray, torch.Tensor], stats: Dict, inverse_transform: bool = False, axis=None) \
        -> Union[np.ndarray, torch.Tensor]:
    """ Normalize dataset.

        Parameters
        ----------
        data: arr or tensor. Contains data to transform.
        stats: arr or tensor. Statistics of the data.
        inverse_transform: bool. False for normalization, True for unnormalization.
        axis: int or None. Axis along which to normalize. If None, normalize across all dimensions.

        Returns
        -------
        data_transform: arr or tensor. Normalized/unnormalized dataset.
    """
    if axis is None:
        return affine(data, stats['max']-stats['min'], stats['min'], inverse_transform=not inverse_transform)
    else:
        return affine(data, stats['max'].max(axis=axis, keepdims=True) - stats['min'].min(axis=axis, keepdims=True),
                      stats['min'].min(axis=axis, keepdims=True), inverse_transform=not inverse_transform)


def max(data: Union[np.ndarray, torch.Tensor], stats: Dict, inverse_transform: bool = False, axis=None)\
        -> Union[np.ndarray, torch.Tensor]:
    """ Divide dataset by its maximum value.

        Parameters
        ----------
        data: arr or tensor. Contains data to transform.
        stats: arr or tensor. Statistics of the data.
        inverse_transform: bool. False for normalization, True for unnormalization.
        axis: int or None. Axis along which to normalize. If None, normalize across all dimensions.

        Returns
        -------
        data_transform: arr or tensor. Transformed dataset.
    """
    if axis is None:
        return multiplication(data, stats['max'], inverse_transform=not inverse_transform)
    else:
        return multiplication(data, stats['max'].max(axis=axis, keepdims=True), inverse_transform=not inverse_transform)


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


def identity(data: Union[np.ndarray, torch.Tensor], **kwargs) \
           -> Union[np.ndarray, torch.Tensor]:
    """ Identity transformation.

        Parameters
        ----------
        data: arr or tensor. Contains data to transform.

        Returns
        -------
        data_transform: arr or tensor. Transformed dataset.
    """

    return data


def sin_cos(data: Union[np.ndarray, torch.Tensor], **kwargs) \
           -> Union[tuple[np.ndarray, ...], tuple[torch.Tensor, ...]]:
    """ Apply sine and cosine transformation to the data.

        Parameters
        ----------
        data: arr or tensor. Contains data to transform.

        Returns
        -------
        data_transform: arr or tensor. Transformed dataset.
    """

    # If data is a numpy array
    if isinstance(data, np.ndarray):
        # Preallocate output arrays
        sin_out = np.empty_like(data)
        cos_out = np.empty_like(data)
        # Compute sine and cosine
        np.sin(data, out=sin_out)
        np.cos(data, out=cos_out)
        return sin_out, cos_out
    # If data is a torch tensor
    else:
        # Preallocate output tensors
        sin_out = torch.empty_like(data)
        cos_out = torch.empty_like(data)
        # Compute sine and cosine
        torch.sin(data, out=sin_out)
        torch.cos(data, out=cos_out)
        return sin_out, cos_out


def clip(data: Union[np.ndarray, torch.Tensor], stats: Dict) \
           -> Union[np.ndarray, torch.Tensor]:
    """ Clip the data to the specified range.

        Parameters
        ----------
        data: arr or tensor. Contains data to transform.
        stats: dict. Contains 'min' and 'max' values for clipping.

        Returns
        -------
        data_transform: arr or tensor. Clipped dataset.
    """

    # Extract min and max values from stats
    min_value = mean_stdev(stats['min'], stats)
    max_value = mean_stdev(stats['max'], stats)

    # Apply clipping based on the type of data
    if isinstance(data, np.ndarray):
        return np.clip(data, min_value, max_value)
    elif isinstance(data, torch.Tensor):
        min_value = torch.as_tensor(min_value, device=data.device, dtype=data.dtype)
        max_value = torch.as_tensor(max_value, device=data.device, dtype=data.dtype)
        # Diffuser si besoin
        min_value = min_value.expand_as(data)
        max_value = max_value.expand_as(data)
        return torch.clamp(data, min=min_value, max=max_value)
    else:
        raise TypeError("Input data must be a numpy array or a torch tensor.")


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
        return np.rot90(data, k=k, axes=axes)
    elif isinstance(data, torch.Tensor):
        # Torch
        return torch.rot90(data, k=k, dims=axes)
    else:
        raise TypeError("Unsupported data type. Expected numpy array or torch tensor.")


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

    # Inverse transform is a rotation in the opposite direction
    if inverse_transform:
        n = -n

    # No flip
    if n is None:
        return data
    # Apply flip
    elif isinstance(data, np.ndarray):
        # Numpy
        return np.flip(data, n)
    elif isinstance(data, torch.Tensor):
        # Torch
        dims = [n] if isinstance(n, int) else n
        return torch.flip(data, dims=dims)
    else:
        raise TypeError("Unsupported data type. Expected numpy array or torch tensor.")


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


def geometric_augmentation(data: list[np.ndarray], vars: list[str], n_rot90: int = 0,
                           axes_rot90: Union[int, tuple[int, int]] = 1, n_flip: int = None) \
        -> list[np.ndarray]:
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
            data[idx_x], data[idx_y] = augment_vector(data[idx_x], data[idx_y],
                                                      n_flip=n_flip, n_rot90=n_rot90, axes_rot90=axes_rot90)
    # Scalar augmentation
    for scalar in ["I500", "vz", "Bz"]:
        if scalar in vars:
            idx = vars.index(scalar)
            data[idx] = augment_scalar(data[idx], n_flip=n_flip, n_rot90=n_rot90, axes_rot90=axes_rot90)
    return data

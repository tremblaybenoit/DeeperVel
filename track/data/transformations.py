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

    # For Numpy arrays
    if isinstance(var2, np.ndarray):
        var2 = np.reshape(var2, (1,) * (var1.ndim - var2.ndim) + var2.shape)

        if isinstance(var1, torch.Tensor):
            # Convert to a Torch tensor if var1 is a Torch tensor
            var2 = torch.tensor(var2, dtype=var1.dtype, device=var1.device)

    # For Torch tensors
    elif isinstance(var2, torch.Tensor):
        var2 = var2.view((1,) * (var1.ndim - var2.ndim) + var2.shape)

        if isinstance(var1, np.ndarray):
            # Convert to a Numpy array if var1 is a Numpy array
            var2 = var2.cpu().numpy()

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
        data_transform = data / scaling_factor
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

    # Change direction of rotation
    if inverse_transform:
        n = -n

    # Apply rotation
    if isinstance(data, np.ndarray):
        # Numpy
        data_transform = np.rot90(data, k=n, axes=axes)
    elif isinstance(data, torch.Tensor):
        # Torch
        data_transform = torch.rot90(data, k=n, dims=axes)
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


def augment(data: Union[np.ndarray, torch.Tensor], n_rot90: int = 0, axes_rot90: Union[int, tuple[int, int]] = 1,
            n_flip: int = None, inverse_transform: bool = False) \
        -> Union[np.ndarray, torch.Tensor]:
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
    """ Perform all possible augmentations of a vector, accounting for directionality (positive and negative signs based on direction.

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
    data_transform_x = flip(data_x, n=n_flip, inverse_transform=inverse_transform)
    data_transform_y = flip(data_y, n=n_flip, inverse_transform=inverse_transform)
    # Account for sign changes
    if n_flip is not None:
        if n_flip == 0:
            data_transform_x = -data_transform_x
        elif n_flip == 1:
            data_transform_y = -data_transform_y
    # Apply rotation and account for the swapping of x and y components
    if n_rot90 == 1:
        data_transform_x, data_transform_y = (-rot90(data_transform_y, n=n_rot90, axes=axes_rot90),
                                              rot90(data_transform_x, n=n_rot90, axes=axes_rot90))
    elif n_rot90 == 2:
        data_transform_x, data_transform_y = (-rot90(data_transform_x, n=n_rot90, axes=axes_rot90),
                                              -rot90(data_transform_y, n=n_rot90, axes=axes_rot90))
    elif n_rot90 == 3:
        data_transform_x, data_transform_y = (rot90(data_transform_y, n=n_rot90, axes=axes_rot90),
                                              -rot90(data_transform_x, n=n_rot90, axes=axes_rot90))

    return data_transform_x, data_transform_y


def apply_vector_augmentation(sample, aug_type, vector_keys=[('vx', 'vy')]):
    """
    Apply augmentation to a sample containing vector fields.
    sample: dict of np.ndarray, e.g., {'vx': ..., 'vy': ..., ...}
    aug_type: str, e.g., 'rot90+flip_x'
    vector_keys: list of tuples, each tuple contains the keys for a vector field
    """
    def _rotate90(vx, vy):
        # 90 deg CCW: (vx, vy) -> (-vy, vx)
        return -np.rot90(vy, k=1), np.rot90(vx, k=1)
    def _rotate180(vx, vy):
        # 180 deg: (vx, vy) -> (-vx, -vy)
        return -np.rot90(vx, k=2), -np.rot90(vy, k=2)
    def _rotate270(vx, vy):
        # 270 deg CCW: (vx, vy) -> (vy, -vx)
        return np.rot90(vy, k=3), -np.rot90(vx, k=3)
    def _flip_x(vx, vy):
        # Flip x axis: invert vx
        return -np.flip(vx, axis=1), np.flip(vy, axis=1)
    def _flip_y(vx, vy):
        # Flip y axis: invert vy
        return np.flip(vx, axis=0), -np.flip(vy, axis=0)

    if '+' in aug_type:
        for op in aug_type.split('+'):
            sample = apply_vector_augmentation(sample, op, vector_keys)
        return sample

    for keys in vector_keys:
        vx, vy = sample[keys[0]], sample[keys[1]]
        if aug_type == 'rot90':
            vx, vy = _rotate90(vx, vy)
        elif aug_type == 'rot180':
            vx, vy = _rotate180(vx, vy)
        elif aug_type == 'rot270':
            vx, vy = _rotate270(vx, vy)
        elif aug_type == 'flip_x':
            vx, vy = _flip_x(vx, vy)
        elif aug_type == 'flip_y':
            vx, vy = _flip_y(vx, vy)
        sample[keys[0]], sample[keys[1]] = vx, vy

    # For scalar fields, just apply the geometric transform
    for k, v in sample.items():
        if not any(k in pair for pair in vector_keys):
            if aug_type.startswith('rot'):
                k_rot = int(aug_type.replace('rot', ''))
                v = np.rot90(v, k=k_rot // 90)
            elif aug_type == 'flip_x':
                v = np.flip(v, axis=1)
            elif aug_type == 'flip_y':
                v = np.flip(v, axis=0)
            sample[k] = v
    return sample

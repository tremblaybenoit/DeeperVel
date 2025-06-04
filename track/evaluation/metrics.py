import numpy as np
from typing import Union, Any
from numpy import floating
import scipy as sc


def norm(vx, vy, vz) -> np.ndarray:
    """ Compute the norm of the vector field.

        Parameters:
        ----------
        vx, vy, vz: np.ndarray. Components of the vector field.

        Returns:
        --------
        norm: np.ndarray. Norm of the vector.
    """
    return np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)


def spearman_correlation_coefficient(v1: np.ndarray, v2: np.ndarray, axis: Union[int, tuple]) -> np.ndarray:
    """
        Compute the Spearman correlation coefficient between two images.

        Parameters:
        ----------
        v1, v2: np.ndarray. Input images.
        axis: Union[int, tuple]. Axis along which to compute the metric (i.e., spatial axes).

        Returns:
        --------
        spearman_corr: np.ndarray. Spearman correlation coefficient between v1 and v2.
    """

    # Flatten the images along the specified axis
    if axis is not None:
        v1 = np.moveaxis(v1, axis, 0).reshape(v1.shape[0], -1)
        v2 = np.moveaxis(v2, axis, 0).reshape(v2.shape[0], -1)
    else:
        v1 = v1.flatten()
        v2 = v2.flatten()

    return sc.spearmanr(v1, v2)[0]


class Metrics:
    """ Class for computing various metrics between two vector fields."""

    def __init__(self, v1x: np.ndarray, v2x: np.ndarray,
                 v1y: np.ndarray = None, v2y: np.ndarray = None,
                 v1z: np.ndarray = None, v2z: np.ndarray = None) -> None:
        """ Initialize the Metrics class with two vector fields.
            Note that v1 is the reference field.

            Parameters:
            ----------
            v1x, v2x: np.ndarray. x component of the vector fields.
            v1y, v2y: np.ndarray, optional. y component of the vector fields. Defaults to None.
            v1z, v2z: np.ndarray, optional. z component of the vector fields. Defaults to None.
        """

        # Assign null values to y and z components if not provided
        self.v1x, self.v2x = v1x, v2x
        self.v1y, self.v2y = np.zeros_like(v1x) if v1y is None else v1y, np.zeros_like(v2x) if v2y is None else v2y
        self.v1z, self.v2z = np.zeros_like(v1x) if v1z is None else v1z, np.zeros_like(v2x) if v2z is None else v2z

    def squared_error(self) -> np.ndarray:
        """ Compute the squared error between two vector fields.

            Returns:
            --------
            metric: np.ndarray. Squared error between v1 and v2.
        """
        return (self.v1x - self.v2x) ** 2 + (self.v1y - self.v2y) ** 2 + (self.v1z - self.v2z) ** 2

    def absolute_error(self) -> np.ndarray:
        """ Compute the absolute error between two vector fields.

            Returns:
            --------
            metric: np.ndarray. Absolute error between v1 and v2.
        """
        return np.sqrt(self.squared_error())

    def absolute_relative_error(self) -> np.ndarray:
        """ Compute the absolute relative error between two vector fields.
            Note that v1 is the reference field.

            Returns:
            --------
            metric: np.ndarray. Absolute relative error between v1 and v2.
        """
        return self.absolute_error() / norm(self.v1x, self.v1y, self.v1z)

    def cosine_similarity(self) -> np.ndarray:
        """ Compute the cosine similarity between two vector fields.

            Returns:
            --------
            metric: np.ndarray. Cosine similarity between v1 and v2.
        """
        # Dot product of the two vectors
        dot_prod = self.v1x * self.v2x + self.v1y * self.v2y + self.v1z * self.v2z
        # Compute norms of the vectors
        norm1 = norm(self.v1x, self.v1y, self.v1z)
        norm2 = norm(self.v2x, self.v2y, self.v2z)

        # Compute cosine similarity index
        return dot_prod / (norm1 * norm2)

    def cosine_similarity_index(self, axis: Union[int, tuple] = None) -> floating[Any]:
        """ Compute the cosine similarity index between two vector fields.

            Parameters:
            ----------
            axis: tuple, optional. Defaults to None. Axis along which to compute the metric (i.e., spatial axes).

            Returns:
            --------
            metric: np.ndarray. Cosine similarity index between v1 and v2.
        """
        return np.mean(self.cosine_similarity(), axis=axis)

    def mean_squared_error(self, axis: Union[int, tuple] = None) -> floating[Any]:
        """ Compute the mean squared error between two vector fields.

            Parameters:
            ----------
            axis: tuple, optional. Defaults to None. Axis along which to compute the metric (i.e., spatial axes).

            Returns:
            --------
            metric: np.ndarray. Mean squared error between v1 and v2.
        """
        return np.mean(self.squared_error(), axis=axis)

    def mean_absolute_error(self, axis: Union[int, tuple] = None) -> floating[Any]:
        """ Compute the mean absolute error between two vector fields.

            Parameters:
            ----------
            axis: tuple, optional. Defaults to None. Axis along which to compute the metric (i.e., spatial axes).

            Returns:
            --------
            metric: np.ndarray. Mean absolute error between v1 and v2.
        """
        return np.mean(self.absolute_error(), axis=axis)

    def mean_absolute_relative_error(self, axis: Union[int, tuple] = None) -> floating[Any]:
        """ Compute the mean absolute relative error between two vector fields.

            Parameters:
            ----------
            axis: tuple, optional. Defaults to None. Axis along which to compute the metric (i.e., spatial axes).

            Returns:
            --------
            metric: np.ndarray. Mean absolute relative error between v1 and v2.
        """
        return np.mean(self.absolute_relative_error(), axis=axis)

    def median_absolute_relative_error(self, axis: Union[int, tuple] = None) -> floating[Any]:
        """ Compute the median absolute relative error between two vector fields.

            Parameters:
            ----------
            axis: tuple, optional. Defaults to None. Axis along which to compute the metric (i.e., spatial axes).

            Returns:
            --------
            metric: np.ndarray. Median absolute relative error between v1 and v2.
        """
        return np.median(self.absolute_relative_error(), axis=axis)

    def pearson_correlation_coefficient(self, axis: Union[int, tuple] = None) -> floating[Any]:
        """ Compute the Pearson correlation coefficient between two vector fields.

            Parameters:
            ----------
            axis: tuple, optional. Defaults to None. Axis along which to compute the metric (i.e., spatial axes).

            Returns:
            --------
            metric: np.ndarray. Pearson correlation coefficient between v1 and v2.
        """

        # Compute based on the definition from Schijver et al. (2005)
        coef = np.sum((self.v1x * self.v2x + self.v1y * self.v2y + self.v1z * self.v2z), axis=axis) / \
               np.sqrt(np.sum(self.v1x ** 2 + self.v1y ** 2 + self.v1z ** 2, axis=axis) *
                       np.sum(self.v2x ** 2 + self.v2y ** 2 + self.v2z ** 2, axis=axis))

        return coef

    def vertical_poynting_flux(self, b1x: np.ndarray, b1y: np.ndarray, b1z: np.ndarray) -> floating[Any]:
        """ Compute the vertical Poynting flux between two vector fields.

            Parameters:
            ----------
            axis: tuple, optional. Defaults to None. Axis along which to compute the metric (i.e., spatial axes).

            Returns:
            --------
            metric: np.ndarray. Vertical Poynting flux between v1 and v2.
        """
        return np.mean(self.v1x * self.v2y - self.v1y * self.v2x, axis=axis)

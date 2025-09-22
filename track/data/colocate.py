import numpy as np
import pickle
import hydra
from omegaconf import DictConfig
from track.utilities.instantiators import instantiate
from track.utilities.logic import get_config_path


# Generate random indices to extract random patches
def generate_indices(x_min: int, x_max: int, y_min: int, y_max: int, t_min: int, t_max: int, dx: int, dy: int,
                     size: int, dt=0) -> tuple[list, list, list]:
    """ Generate random indices for x, y, and t coordinates.

        Parameters
        ----------
        x_min: int. Minimum value for x coordinate.
        x_max: int. Maximum value for x coordinate.
        y_min: int. Minimum value for y coordinate.
        y_max: int. Maximum value for y coordinate.
        t_min: int. Minimum value for t coordinate.
        t_max: int. Maximum value for t coordinate.
        dx: int. Minimum distance between x coordinates.
        dy: int. Minimum distance between y coordinates.
        size: int. Number of indices to generate.
        dt: int, optional. Minimum time difference, by default 0.

        Returns
        -------
        Tuple of numpy arrays containing x, y, and t coordinates.
    """

    # Generate random indices for x, y, and t coordinates
    xs = np.arange(x_min, x_max, dx, dtype=int)
    ys = np.arange(y_min, y_max, dy, dtype=int)
    ts = np.arange(t_min, t_max + 1, dtype=int)
    all_cells = []

    # Generate all possible shifted grid cells for each timestep
    for t in ts:
        x_shift = np.random.randint(0, dx, dtype=int)
        y_shift = np.random.randint(0, dy, dtype=int)
        grid_xs = xs + x_shift
        grid_ys = ys + y_shift
        grid_xs = grid_xs[(grid_xs >= x_min) & (grid_xs < x_max)]
        grid_ys = grid_ys[(grid_ys >= y_min) & (grid_ys < y_max)]
        grid = np.array(np.meshgrid(grid_xs, grid_ys)).reshape(2, -1).T
        for x, y in grid:
            all_cells.append((int(x), int(y), int(t)))

    # Shuffle and select cells with constraints
    np.random.shuffle(all_cells)
    selected = []
    with tqdm(total=size, desc="Selecting cells") as pbar:
        for cell in all_cells:
            x, y, t = cell
            # Check for spatial and temporal constraints
            conflict = False
            for x0, y0, t0 in selected:
                if abs(t0 - t) <= dt and x0 <= dx and y0 <= dy:
                    conflict = True
                    break
            if not conflict:
                selected.append(cell)
                pbar.update(1)
            if len(selected) == size:
                break

    if len(selected) < size:
        raise RuntimeError("Could not find enough valid samples with given constraints.")

    x_out, y_out, t_out = zip(*selected)
    return list(x_out), list(y_out), list(t_out)


def colocate(config: DictConfig, save=True) -> dict:
    """ Assign patches for input and output variables.

        Parameters
        ----------
        config : DictConfig. Configuration file containing metadata.
        save : bool, optional. If True, saves the colocated data to a file, by default True.

        Returns
        -------
        dict. Dictionary containing colocated data with indices for x, y, and t coordinates.
    """

    # Initialize input and output configs
    data_config = config.input.data
    patches_config = config.input.patches

    # Initialize data loader
    io = instantiate(data_config, _partial_=False)

    # Constraints for colocated data
    x_min, y_min = 0, 0
    x_max, y_max = io.nx - patches_config.size[1], io.ny - patches_config.size[0]
    t_min, t_max = (np.abs(patches_config.t_range[0] + np.amin(patches_config.dt)),
                    patches_config.t_range[1] - 1 - np.abs(np.amax(patches_config.dt)))
    x, y, t = generate_indices(x_min, x_max, y_min, y_max, t_min, t_max, patches_config.size[1], patches_config.size[0],
                               patches_config.n_samples)

    # Store colocated data
    patches = {"nx": patches_config.size[1], "x_min": x, "x_max": [x_i + patches_config.size[1] for x_i in x],
               "ny": patches_config.size[0], "y_min": y, "y_max": [y_i + patches_config.size[0] for y_i in y],
               "nt": patches_config.n_samples, "t": t, "t_min": t_min, "t_max": t_max}

    # Save colocated data to a file
    if save and getattr(getattr(config.output, "patches", None), "path", None):
        with open(config.output.patches.path, 'wb') as file:
            # noinspection PyTypeChecker
            pickle.dump(patches, file)

    return patches


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """ Compute statistics of a given dataset.

        Parameters
        ----------
        config: DictConfig. Configuration file containing data and statistics paths.

        Returns
        -------
        None.
    """

    # If statistics is part of the preparation steps:
    if hasattr(config.data.preparation, "colocate"):
        # Assign patches for input and output variables
        colocation = colocate(config.data.preparation.colocate)

    return


if __name__ == '__main__':
    """ Compute corresponding indices for tracking inputs and outputs.

        Parameters
        ----------
        --config_path: str. Directory containing configuration file.
        --config_name: str. Configuration filename.
        +experiment: str. Experiment configuration filename to override default configuration.

        Returns
        -------
        pickle file containing indices.
    """

    main()
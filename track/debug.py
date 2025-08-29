import hydra
from omegaconf import DictConfig, OmegaConf
from track.utilities.logic import get_config_path
from track.utilities.instantiators import instantiate
from track.data.colocate import colocate
import pickle
import numpy as np
from track.data.transformations import geometric_augmentation, min_max
from track.utilities.plot import flexible_gridspec, plot_map, save_plot
from track.data.io import MURaMQSDataset


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """ Train neural network based on a set of configurations.

        Parameters
        ----------
        config: str. Main hydra configuration file containing all model hyperparameters.

        Returns
        -------
        None.
    """

    # TODO: Test colocation
    colocations = colocate(config.data.preparation.colocate)

    # Read colocated data
    if hasattr(config.data.preparation, "colocate"):
        with open(config.data.preparation.colocate.output.patches.path, 'rb') as file:
            colocated_data = pickle.load(file)

    # TODO: Read statistics
    if hasattr(config.data.preparation, "statistics"):
        # Read statistics from the pickle file
        with open(config.data.dataset.yz.statistics.path, 'rb') as file:
            statistics = pickle.load(file)


    breakpoint()

    ex_path = "E:\\Data\\ISSI_Team_Flows\\Matthias\\SSD_25x8Mm_16_pdmp_1_ISSI_Flows\\2D"  # os.path.abspath("../E/Data/ISSI_Team_Flows/Matthias/SSD_25x8Mm_16_pdmp_1_ISSI_Flows/2D/")
    ex_slice_type = 'yz'
    ex_slice = [0]  # [192, 400]
    ex_iter = [1]  # [0, 3900, 4200]
    ex_vars = ['I500', 'vx', 'vy', 'vz', 'Bx', 'By', 'Bz']
    ex_x_min, nx, ex_y_min, ny = 256, 96, 256, 96

    # Initialize the dataset
    dataset = MURaMQSDataset(ex_path, dataset=ex_slice_type)
    data = dataset.read(ex_iter, ex_slice, ex_vars)
    vx =data[:, :, 0, 0, 1]
    vx_norm = min_max(vx, statistics[ex_slice[0]]['vx'])

    t = 2560
    data2 = dataset.read(colocated_data['t'][t], ex_slice, ex_vars, x_min=colocated_data['x_min'][t], y_min=colocated_data['y_min'][t],
                         nx=colocated_data['nx'], ny=colocated_data['ny'])
    vx2 = data2[:, :, 0, 0, 1]
    vx2_norm = min_max(vx2, statistics[ex_slice[0]]['vx'])

    breakpoint()


    # Combinations for geometric augmentation
    combinations = [
        (0, None),  # identity
        (1, None),  # rot90
        (2, None),  # rot180
        (3, None),  # rot270
        (0, 0),  # flip x
        (0, 1),  # flip y
        (1, 0),  # rot90 + flip x
        (1, 1),  # rot90 + flip y
    ]
    n_flip, n_rot90 = combinations[np.random.randint(0, 8)]
    axes_rot90 = (0, 1)

    # TODO: Test data loader (train)
    data_loader = instantiate(config.data.loader)
    data_loader.setup(stage='train')

    # Read one patch from the data loader
    patch = data_loader.ds_train[4]

    # Plot input and output patches
    figure, get_axes = flexible_gridspec([4, 4, 4])
    ax0 = get_axes(0, 0)
    plot_map(ax0, patch[0][0], img_pixel=(0.016, 0.016))
    ax1 = get_axes(0, 1)
    plot_map(ax1, patch[0][1], img_pixel=(0.016, 0.016))
    ax2 = get_axes(0, 2)
    plot_map(ax2, patch[0][2], img_pixel=(0.016, 0.016))
    ax3 = get_axes(0, 3)
    plot_map(ax3, patch[0][3], img_pixel=(0.016, 0.016))
    ax4 = get_axes(1, 0)
    plot_map(ax4, patch[0][4], img_pixel=(0.016, 0.016))
    ax5 = get_axes(1, 1)
    plot_map(ax5, patch[0][5], img_pixel=(0.016, 0.016))
    ax6 = get_axes(1, 2)
    plot_map(ax6, patch[0][6], img_pixel=(0.016, 0.016))
    ax7 = get_axes(1, 3)
    plot_map(ax7, patch[0][7], img_pixel=(0.016, 0.016))
    ax8 = get_axes(2, 0)
    plot_map(ax8, patch[0][8], img_pixel=(0.016, 0.016))
    ax9 = get_axes(2, 1)
    plot_map(ax9, patch[0][9], img_pixel=(0.016, 0.016))
    ax10 = get_axes(2, 2)
    plot_map(ax10, patch[1][0], img_pixel=(0.016, 0.016))
    ax11 = get_axes(2, 3)
    plot_map(ax11, patch[1][1], img_pixel=(0.016, 0.016))

    save_plot(figure, filename='patch_example4.png')




    breakpoint()


    # TODO: Test data loader (test)
    data_loader2 = instantiate(config.data)
    data_loader2.setup(stage='test')

    # TODO: Test statistics (with transverse invariance)

    # TODO: Test augmentation

    # TODO: Test plotting routines

    return


if __name__ == '__main__':
    """ Train neural network to track plasma motions (or other physical quantities).

        Parameters
        ----------
        --config_path: str. Directory containing configuration file.
        --config_name: str. Configuration filename.
        +experiment: str. Experiment configuration filename to override default configuration.

        Returns
        -------
        checkpoint: Training weights & biases.
    """

    main()

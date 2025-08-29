from config.setup import read_hydra_as_dict, setup_directories_from_hydra, check_key_in_dict


#########################################################################################################
# CONFIGURATION
#########################################################################################################

# Hydra/Snakemake config
hydra_config_path = config.get("hydra-config-path", "../config")
hydra_config_name = config.get("hydra-config-name", "default")
hydra_experiment = config.get("hydra-experiment", None)

# Create the necessary directories
setup_directories_from_hydra(config_path=hydra_config_path, config_name=hydra_config_name,
                             overrides=f'+experiment={hydra_experiment}')
# Hydra configuration file
hydra_config = read_hydra_as_dict(config_path=hydra_config_path, config_name=hydra_config_name,
                                  overrides=f'+experiment={hydra_experiment}')
# Data configuration file (from Snakemake config file)
data_config = hydra_config["data"]
# Loader configuration file (from the Snakemake config file)
loader_config = hydra_config["loader"]
# Paths configuration file (from the Snakemake config file)
paths_config = hydra_config["paths"]
# Callback configuration file (from Snakemake config file)
checkpoint_config = hydra_config["callbacks"]["model_checkpoint"]

def get_path_from_config(config, key='path', flag_directories=False):
    """
    Extracts paths from a configuration dictionary.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    key : str, optional
        Key to extract from the configuration dictionary. The default is 'path'.
    flag_directories : bool, optional
        Flag to check if the configuration is a directory. The default is False.

    Returns
    -------
    dict
        Dictionary containing the paths.
    """

    # If the flag is set to True
    if flag_directories:

        # Check if the key is a directory
        return [directory(config[subkey][key]) if config[subkey]['type'] == 'directory'
                else config[subkey][key] for subkey in config]

    else:
        # Extract the paths from the configuration
        return [config[subkey][key] for subkey in config]

#########################################################################################################
# TARGET
#########################################################################################################

# Target: The one rule to rule them all
rule trained_tracker:
    input:
        checkpoint = checkpoint_config['filename']

#########################################################################################################
# JOBS
#########################################################################################################

# Loop over preparation steps
for step_nb, step_name in enumerate(data_config['preparation']):
    rule:
        name: f"{step_name}"
        input:
            get_path_from_config(data_config['preparation'][step_name]['input'], key='path')
        output:
            get_path_from_config(data_config['preparation'][step_name]['output'], key='path', flag_directories=True)
        params:
            config_name = hydra_config_name,
            experiment = hydra_experiment,
            target = step_name,
            condition = f"~data.dataset.state.preparation.{step_name}" if check_key_in_dict(step_name, data_config['preparation']) else ""
        shell:
            """
            python -m track.data.{params.target} \
            --config-name={params.config_name} \
            +experiment={params.experiment} \
            {params.condition}
            """

# Location of the data
#if check_key_in_dict('colocation', data_config):
#    rule colocation:
#        input:
#            radiance_data = data_config['colocation']['input']['radiance']['path'],
#            state_data = data_config['colocation']['input']['state']['path']
#        params:
#            config_name = hydra_config_name,
#            experiment = hydra_experiment,
#            log_path= paths_config['log_dir'],
#            run_path=paths_config['run_dir']
#        output:
#            dataframe = data_config['colocation']['output']['data']['path']
#        shell:
#            """
#            python -m retrieval.data.colocation \
#            --config-name={params.config_name} \
#            +experiment={params.experiment}
#        """

# Training
rule training:
    input:
        radiance_data = radiance_config['files']['data']['path'],
        state_data = state_config['files']['data']['path'],
        radiance_stats = radiance_config['preparation']['statistics']['output']['data']['path'],
        state_stats = state_config['preparation']['statistics']['output']['data']['path'],
        data_frame = data_config['colocation']['output']['data']['path']
    params:
        config_name = hydra_config_name,
        experiment = hydra_experiment,
        checkpoint_path = paths_config['checkpoint_dir'],
        log_path = paths_config['log_dir'],
        run_path = paths_config['run_dir'],
    output:
        checkpoint = checkpoint_config['filename']
    # resources:
    #     nvidia_gpu = 1
    shell:
        """
        python -c "import os; os.makedirs('{params.log_path}', exist_ok=True)" && \
        python -c "import os; os.makedirs('{params.run_path}', exist_ok=True)" && \
        python -c "import os; os.makedirs('{params.checkpoint_path}', exist_ok=True)" && \
        python -m retrieval.train \
        --config-name={params.config_name} \
        +experiment={params.experiment}
        """

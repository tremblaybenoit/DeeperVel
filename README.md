# DeeperVel 
A unified, automated, and reproducible framework for deep learning-based velocity (or other physical quantities of interest) reconstructions from observations.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Experiment configuration](#experiment-configuration)
  - [Manual execution of individual steps](#manual-execution-of-individual-steps)
  - [Automated workflow (recommended)](#automated-workflow-recommended)
- [Documentation](#documentation)
- [References and Acknowledgements](#references-and-acknowledgements)

# Installation

Clone the repository:

```bash
git clone https://github.com/tremblaybenoit/DeeperVel.git
```
DeeperVel is built with [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) and [Hydra](https://hydra.cc/docs/intro/). 

Create a new conda environment and install pre-requisites from [`environment.yaml`](environment.yaml):
```bash
conda create -n deepervel python=3.10
conda activate deepvervel
conda env update -f environment.yaml
```

# Usage
## Experiment configuration
Create or edit a configuration file in the [`config/experiment`](config/experiment) folder to set your experiment parameters.

1. Start with the `defaults` section to set the default configurations:
    - `paths` (from folder [`config/paths`](config/paths)): Directories for data and outputs.
    - `hydra` (from folder [`config/hydra`](config/hydra)): Hydra settings.
    - `data` (from folder [`config/data`](config/data)): Dataset parameters.
    - `preparation` (from folder [`config/preparation`](config/preparation)): Data preparation steps.
    - `loader` (from folder [`config/loader`](config/loader)): Data loading parameters.
    - `model` (from folder [`config/model`](config/model)): Model architecture and parameters.
    - `trainer` (from folder [`config/trainer`](config/trainer)): Training parameters.
    - `callbacks` (from folder [`config/callbacks`](config/callbacks)): Callbacks during training.
    - `logger` (from folder [`config/logger`](config/logger)): Logging parameters.
2. Add `overrides` below the `defaults` to change specific default parameters as needed. 

**Note**: The order of the `defaults` matters, as later entries can override earlier ones.

**Example**: The following diagram illustrates the structure of [`config/experiment/MURaMQS_DeepVel.yaml`](config/experiment/MURaMQS_DeepVel.yaml). 
It sets the `defaults` and then performs parameter `overrides`.
```mermaid
---
title: Structure of the experiment configuration file "config/experiment/MURaMQS_DeepVel.yaml"
---
flowchart LR
  A["/experiment: MURaMQS_DeepVel"]
  A --> B["defaults"]
  B --> B1["/paths: default"]
  B --> B2["/hydra: default"]
  B --> B3["/data: MURaMQS"]
  B --> B4["/preparation: default"]
  B --> B5["/loader: default"]
  B --> B6["/model: DeepVel"]
  B --> B7["/trainer: gpu"]
  B --> B8["/callbacks: default"]
  B --> B9["/logger: default"]

  %% All configs point to overrides
  B1 --> C
  B2 --> C
  B3 --> C
  B4 --> C
  B5 --> C
  B6 --> C
  B7 --> C
  B8 --> C
  B9 --> C

  C["Overrides"]
  C --> C1["task_name"]
  C --> C2["/paths"]
  C2 --> C21["task_dir"]
  C2 --> C22["run_id"]
  C2 --> C23["data_dir"]
  C --> C3["/trainer"]
  C3 --> C31["min_epochs"]
  C3 --> C32["max_epochs"]
  C --> C4["/data"]
  C4 --> C41["dtype"]

  %% Color table
  classDef experiment fill:#22313F,stroke:#888,stroke-width:1px,color:#fff;
  classDef final_overrides fill:#22313F,stroke:#888,stroke-width:1px,color:#fff;
  classDef paths fill:#FFD580,stroke:#888,stroke-width:1px,color:#000;
  classDef hydra fill:#A97FFF,stroke:#888,stroke-width:1px,color:#000;
  classDef data fill:#B0E57C,stroke:#888,stroke-width:1px,color:#000;
  classDef preparation fill:#FFB347,stroke:#888,stroke-width:1px,color:#000;
  classDef loader fill:#FF7F7F,stroke:#888,stroke-width:1px,color:#000;
  classDef model fill:#FFB3B3,stroke:#888,stroke-width:1px,color:#000;
  classDef trainer fill:#80B3FF,stroke:#888,stroke-width:1px,color:#000;
  classDef callbacks fill:#57D9AD,stroke:#888,stroke-width:1px,color:#000;
  classDef logger fill:#D99157,stroke:#888,stroke-width:1px,color:#000;

  %% Assignation des classes
  class A,B,C1 experiment;
  class C final_overrides;
  class B1,C2,C21,C22,C23 paths;
  class B2 hydra;
  class B3,C4,C41 data;
  class B4 preparation;
  class B5 loader;
  class B6,C5,C51 model;
  class B7,C3,C31,C32 trainer;
  class B8 callbacks;
  class B9 logger;
```

## Manual execution of individual steps
Each step of the workflow can be run manually using the corresponding Python script and 
experiment configuration.

1. Configure directories:
```bash
python -m config.setup -overrides "+experiment=MURaMQS_DeepVel"
```
2. Prepare data for the tracking model training:
```bash
python -m track.data.colocate +experiment=MURaMQS_DeepVel
python -m track.data.statistics +experiment=MURaMQS_DeepVel
```
3. Train the tracking model:
```bash
python -m track.train +experiment=MURaMQS_DeepVel
```
4. Test and evaluate the tracking model:
```bash
python -m track.test +experiment=MURaMQS_DeepVel
python -m track.evaluation.validation +experiment=MURaMQS_DeepVel
```
5. Predict using the tracking model:
```bash
python -m track.predict +experiment=MURaMQS_DeepVel
```

## Automated workflow (recommended)
DeeperVel uses the [Snakemake workflow management system](https://snakemake.readthedocs.io/en/stable/) for reproducibility.

To perform a dry-run (i.e., to check the workflow prior to execution) of the Snakefile rule [`test`](Snakefile) with the [`MURaMQS_DeepVel`](config/experiment/MURaMQS_DeepVel.yaml) experiment configuration:

```bash
snakemake --dry-run --verbose test --config hydra-experiment=MURaMQS_DeepVel
```

Remove `--dry-run` to actually run the workflow. 

To account for missing dependencies, add the `--rerun-incomplete` flag:

```bash
snakemake --dry-run --rerun-incomplete --verbose test --config hydra-experiment=MURaMQS_DeepVel
```

To draw a [directed acyclic graph (DAG)](https://en.wikipedia.org/wiki/Directed_acyclic_graph) of the training workflow (e.g., [`track/train.mmd`](track/train.mmd) for Snakefile rule [`test`](Snakefile)):

```bash
snakemake test --rulegraph mermaid-js --config hydra-experiment=MURaMQS_DeepVel > train.mmd
```
Replace `--rulegraph` with `--dag` to highlight completed rules with dashed boxes.

**Example**: The following graph shows the workflow for the Snakefile rule [`test`](Snakefile) for experiment [`MURaMQS_DeepVel`](config/experiment/MURaMQS_DeepVel.yaml). 

```mermaid
---
title: DeeperVel training workflow - Tracking model
---
flowchart TB
	id0[test]
	id1[data]
	id2[statistics]
	id3[split]
	id4[train]
	style id0 fill:#57CAD9,stroke-width:2px,color:#333333
	style id1 fill:#D9CA57,stroke-width:2px,color:#333333
	style id2 fill:#57D9AD,stroke-width:2px,color:#333333
	style id3 fill:#D99157,stroke-width:2px,color:#333333
	style id4 fill:#D95757,stroke-width:2px,color:#333333
	id4 --> id0
	id1 --> id0
	id2 --> id0
	id3 --> id0
	id1 --> id2
	id1 --> id3
	id1 --> id4
	id2 --> id4
	id3 --> id4
```

# Documentation
The DeeperVel project documentation is available at https://deepervel.readthedocs.io/.

# References and Acknowledgements
- Model architectures:
  - ``DeepVel`` was adapted from the original implementation by Asensio Ramos et al. (2017):
    - Paper: https://www.aanda.org/articles/aa/pdf/2017/08/aa30783-17.pdf.
    - Repository: https://github.com/aasensio/deepvel.
  - ``DeepVelU`` was adapted from the original implementation by Tremblay & Attié (2020):
    - Paper: https://www.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2020.00025/full.
    - Repository: https://github.com/tremblaybenoit/DeepVelU_Frontiers.
  - ``DeeperVel`` was adapted from the original implementation by Tremblay & Cossette (2021):
    - Paper: https://www.swsc-journal.org/articles/swsc/pdf/2021/01/swsc200059.pdf.
  - ``MultiScaleDL`` was adapted from the original implementation by Ishikawa et al. (2022):
    - Paper: https://www.aanda.org/articles/aa/pdf/2022/02/aa41743-21.pdf.
    - Repository: https://github.com/RT-Ishikawa/MultiScaleDL.
- Workflow:
  - Inspiration for the [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) and [Hydra](https://hydra.cc/docs/intro/) framework comes from the following repositories: 
    - Lightning-Hydra-Template: https://github.com/ashleve/lightning-hydra-template.
    - ``Anemoi`` framework by ECMWF: https://github.com/ecmwf/anemoi-core.

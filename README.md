# DeeperVel
Deep learning flow tracking tool for the solar photopshere, chromosphere, and upper convection zone; heavily-inspired by Andrés Asensio Ramos' DeepVel neural network.

# Installation

Clone the repository and install the requirements:

```bash
git clone 
```

Pytorch Lightning + Hydra + Snakemake framework:
```bash
pip install pytorch-lightning hydra-core snakemake
```

# Usage
Example:
```bash
snakemake -np
```
DAG:
```bash
snakemake --rulegraph flowtracking | dot -Tsvg > dag.svg
```

# Documentation
For more information, please refer to the [documentation](https://deepervel.readthedocs.io/en/latest/).

# References
- Asensio Ramos, A. (2019). DeepVel: Deep learning for the estimation of horizontal velocities of solar granules. Astronomy & Astrophysics, 623, A101.
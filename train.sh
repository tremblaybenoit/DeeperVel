#!/bin/bash
# Loop through all matching YAML files
for config in config/experiments/MURaMQS_DeepVel_*.yaml; do
    echo "Training with config: $config"
    python -m track.train +experiment="$config"
done
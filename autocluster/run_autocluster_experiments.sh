#!/bin/bash

# Define the conda environment name
conda_env_name="autocluster_experiments"

# Check if the conda environment exists
if conda env list | grep -q "$conda_env_name"; then
    echo "Conda environment '$conda_env_name' already exists."
else
    # Create conda environment from environment.yml
    conda env create --name "$conda_env_name" --file environment.yml
fi

# Activate the conda environment
source activate "$conda_env_name"

python3 autocluster_experiments.py

conda deactivate
#!/bin/bash

# Define the conda environment name
conda_env_name="ml2dac_experiments"

# Check if the conda environment exists
if conda env list | grep -q "$conda_env_name"; then
    echo "Conda environment '$conda_env_name' already exists."
else
    # Create conda environment from environment.yml
    conda env create --name "$conda_env_name" --file environment.yml
fi

# Activate the conda environment
source activate "$conda_env_name"

# Get the directory of the script
script_dir=$(dirname "$(readlink -f "$0")")

# Set PYTHONPATH relative to the script's directory
export PYTHONPATH=$PYTHONPATH:$script_dir/src

# Execute the Python script
python3 $script_dir/ml2dac_experiments.py

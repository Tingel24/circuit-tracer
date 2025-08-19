#!/bin/bash -l
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks=1                         # number of tasks
#SBATCH --ntasks-per-node=1                # number of tasks per node
#SBATCH --gpus-per-task=1                  # number of gpu per task
#SBATCH --cpus-per-task=1                  # number of cores per task
#SBATCH --time=00:15:00                    # time (HH:MM:SS)
#SBATCH --partition=gpu                    # partition
#SBATCH --account=p200918                  # project account
#SBATCH --qos=default                      # SLURM qos

# Set working directory and Conda base path
export WDIR="/home/users/u103092/circuit-tracer/local-circuit-tracer/cluster"
export CONDA_HOME="$WDIR/conda_base_path/miniconda3"
export PATH="$CONDA_HOME/bin:$PATH"

# Activate Conda environment
source activate circuit-tracer-env

echo "starting python script"
# Run the attribution script
python model_attribution.py

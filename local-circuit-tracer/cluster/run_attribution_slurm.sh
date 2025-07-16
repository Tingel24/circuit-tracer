#!/bin/bash -l
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks=1                         # number of tasks
#SBATCH --ntasks-per-node=1                # number of tasks per node
#SBATCH --gpus-per-task=1                  # number of gpu per task
#SBATCH --cpus-per-task=1                  # number of cores per task
#SBATCH --time=00:15:00                    # time (HH:MM:SS)
#SBATCH --partition=gpu                    # partition
#SBATCH --account=u103092                  # project account
#SBATCH --qos=default                      # SLURM qos

# Set working directory and Conda base path
export WDIR="/project/home/p200xxx"
export CONDA_HOME="$WDIR/conda_base_path/miniconda3"
export PATH="$CONDA_HOME/bin:$PATH"

#Load Python module
module load Python

#Check Python version
python -c  'import sys; print(sys.version)'

# Activate Conda environment
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate circuit-tracer-env

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the attribution script
srun python run_model_attribution.py

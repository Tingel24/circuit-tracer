#!/bin/bash

# One-time setup script for Miniconda and project environment

# Define working directory (change to your project path)
export WDIR="/home/users/u103092/circuit-tracer/local-circuit-tracer/cluster"
export CONDA_DIR="$WDIR/conda_base_path"
export CONDA_HOME="$CONDA_DIR/miniconda3"

# Create base directory
mkdir -p "$CONDA_DIR"
cd "$CONDA_DIR"

# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh

# Run installer
./Miniconda3-latest-Linux-x86_64.sh -b -p "$CONDA_HOME"

# Initialize Conda for bash
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda init bash

# Activate base environment
conda activate

# Create new environment for your project
conda create -y --name circuit-tracer-env python=3.11.3
conda activate circuit-tracer-env

# Install required Python packages
conda install -y numpy pandas torch torchvision
pip install transformers git+https://github.com/Tingel24/circuit-tracer.git

# Confirm installation
which python
python --version
conda info --envs

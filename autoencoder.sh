#!/bin/bash

#SBATCH --job-name=autoencoder
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

module load miniconda
conda activate mfcn

python autoencoder.py
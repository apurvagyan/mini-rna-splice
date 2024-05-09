#!/bin/bash

#SBATCH --job-name=scatter_rna
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=pi_krishnaswamy
#SBATCH --mem=64G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err
module load miniconda
conda activate env_3_8

python scatter_graphs.py
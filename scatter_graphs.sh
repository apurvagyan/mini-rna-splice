#!/bin/bash

#SBATCH --job-name=scatter_rna
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=6
#SBATCH --partition=gpu
#SBATCH --mem=256G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err
module load miniconda
conda activate mfcn

python scatter_graphs.py
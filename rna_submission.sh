#!/bin/bash
  
#SBATCH --job-name=process_rna
#SBATCH --output=process_rna.txt
#SBATCH --cpus-per-task=4       # Request 4 CPUs
#SBATCH --mem-per-cpu=8G        # Request 4GB of memory per CPU
#SBATCH --time=23:00:00
#SBATCH --partition=pi_krishnaswamy

module load miniconda
conda activate splicenn

python process_rna_seq.py


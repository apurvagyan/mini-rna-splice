#!/bin/bash
  
#SBATCH --job-name=process_rna_sequences
#SBATCH --output=folding_job_output.txt
#SBATCH --cpus-per-task=24    
#SBATCH --mem-per-cpu=8G     
#SBATCH --time=23:59:00
#SBATCH --mail-user=charles.xu@yale.edu   
#SBATCH --mail-type=ALL

module load miniconda
conda activate splicenn

python fold_rna_seq.py


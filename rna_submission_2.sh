#!/bin/bash
  
#SBATCH --job-name=process_rna_2
#SBATCH --output=process_rna_2.txt
#SBATCH --cpus-per-task=16    
#SBATCH --mem-per-cpu=8G     
#SBATCH --time=23:00:00
#SBATCH --mail-user=charles.xu@yale.edu   
#SBATCH --mail-type=BEGIN,END  

module load miniconda
conda activate splicenn

python process_rna_seq_2.py


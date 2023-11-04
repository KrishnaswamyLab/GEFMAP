#!/bin/bash

#SBATCH --job-name=process_EB
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err
# SECONDS = 0.00
cd ~/project/metabolic_GNN/src
module load miniconda
conda activate metabolomics

python  preprocess_EB.py
# echo “Time elapsed: $SECONDS seconds” >> fcn_ecoli_test.txt
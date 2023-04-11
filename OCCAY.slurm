#!/bin/bash
#SBATCH -J OCCAY
#SBATCH -t 72:00:00 
#SBATCH -N 1
#SBATCH --nodelist=node1
#SBATCH --ntasks-per-node=5
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8

source /usr/anaconda3/etc/profile.d/conda.sh
conda activate swimming
env | grep SLURM
#srun python3 train.py
srun python3 train.py

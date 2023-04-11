#!/bin/bash
#SBATCH -J OCCAY
#SBATCH -t 72:00:00 
#SBATCH -N 1
#SBATCH --nodelist=node1
#SBATCH --ntasks-per-node=2 
#SBATCH --gpus-per-task=1 
#SBATCH --cpus-per-task=4


source /usr/anaconda3/etc/profile.d/conda.sh
conda activate swimming
env | grep SLURM
srun python test.py

#!/bin/bash
#SBATCH --job-name=x
#SBATCH -N 1
##SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --mail-user=zhicao@umich.edu
#SBATCH --mail-type=END
SBATCH --output="result.txt"

hostname
nvidia-smi

sleep 15
python tutorial_train.py
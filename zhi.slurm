#!/bin/bash
#SBATCH --job-name="finetune"
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
SBATCH --output=result.txt
#SBATCH --mail-user=zhicao@umich.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=lgns
#SBATCH --nodelist=lgn4

hostname
nvidia-smi

sleep 15
python tutorial_train.py

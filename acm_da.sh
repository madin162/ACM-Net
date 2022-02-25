#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -o stdout.%j
#SBATCH -e stderr.%j
#SBATCH --gres=gpu:1

source activate man_torch1.10
python main_act.py --batch_size 4 --num_workers 0 --without_wandb --train_mode source
source deactivate
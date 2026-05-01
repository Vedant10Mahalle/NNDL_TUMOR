#!/bin/bash
#SBATCH --job-name=tumor_cnn
#SBATCH --partition=h200
#SBATCH --gres=gpu:1g.18gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/output.txt

source ~/env/bin/activate
cd ~/tumor_cnn

python train.py --model resnet

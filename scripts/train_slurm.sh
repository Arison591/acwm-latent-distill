#!/usr/bin/env bash
# Example SLURM training script for ACWM-DiT.
# Adapt partition/account/paths for your cluster.
#SBATCH --job-name=acwm_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.log

ENV=${1:-"push_cube"}          # e.g. bash train_slurm.sh push_cube
CONFIG="configs/envs/${ENV}.yaml"

export ACWM_DATA_ROOT="/path/to/acwm-phys-dataset"
export WAN_VAE_PATH="/path/to/Wan2.1_VAE.pth"

torchrun \
  --nproc_per_node=4 \
  train.py \
  --config "$CONFIG"

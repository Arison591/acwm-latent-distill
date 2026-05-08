"""
Entry point for training ACWM-DiT.

Single GPU:
    python train.py --config configs/envs/push_cube.yaml

Multi-GPU (torchrun):
    torchrun --nproc_per_node=4 train.py --config configs/envs/push_cube.yaml

Environment variables:
    ACWM_DATA_ROOT   Root directory of the ACWM-Phys dataset  (default: ./data)
    WAN_VAE_PATH     Path to Wan2.1_VAE.pth                   (default: Wan2.1_VAE.pth)
    WANDB_PROJECT    W&B project name                          (default: from config)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from acwm.trainer.train_dynamics import main

if __name__ == "__main__":
    main()

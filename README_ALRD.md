# ACWM Latent Distillation Scaffold

This repository update adds a small research scaffold for action-latent response distillation on top of ACWM-Phys.

The first target is not a large new world model. The target is a controlled Push Cube/Reacher loop that can answer one narrow question:

> Can a compact student preserve specialist teachers' action response under counterfactual actions?

## What Changed

- `acwm/action_latent/`
  - continuous action encoders: identity, MLP, Conv1D
  - magnitude bucket assignment for `small` / `large` action regimes
  - counterfactual actions: `zero`, `reverse`, `scale_0_5`, plus optional `scale_2` and `shuffle`
- `acwm/distill/`
  - prediction KD and response KD losses
  - bucket teacher checkpoint loading helpers
- `acwm/dynamics/diffusion_forcing_wm.py`
  - optional `model_config.action_encoder`
  - distillation-friendly `training_outputs(...)`
- `acwm/trainer/train_dynamics.py`
  - fixed model config path loading
  - optional action bucket filtering for specialist teacher training
  - optional teacher-student KD and response KD during training
- `scripts/alrd/`
  - action inspection, bucket cache generation, action ablation evaluation, and smoke checks

## Quick Start

Static smoke test without ACWM-Phys data:

```bash
cd /root/acwm-latent-distill
bash scripts/alrd/run_push_cube_smoke.sh
```

Inspect actions after downloading data:

```bash
export ACWM_DATA_ROOT=/path/to/ACWM-Phys
python scripts/alrd/inspect_actions.py --env push_cube --split ind_train
python scripts/alrd/make_action_buckets.py --env push_cube --split ind_train
```

Train the first latent-action baseline:

```bash
export WAN_VAE_PATH=/path/to/Wan2.1_VAE.pth
WANDB_MODE=disabled python train.py --config configs/alrd/push_cube_latent.yaml
```

Train specialist teachers:

```bash
WANDB_MODE=disabled python train.py --config configs/alrd/push_cube_teacher_small.yaml
WANDB_MODE=disabled python train.py --config configs/alrd/push_cube_teacher_large.yaml
```

Train the response-KD student after teacher checkpoints exist:

```bash
WANDB_MODE=disabled python train.py --config configs/alrd/push_cube_student_resp_kd.yaml
```

Run action ablation evaluation:

```bash
python scripts/alrd/eval_action_ablation.py \
  --env push_cube \
  --cfg configs/alrd/push_cube_student_resp_kd.yaml \
  --ckpt checkpoints/alrd_push_cube_student_resp_kd_240x240/latest.pt \
  --steps 20 \
  --save_videos
```

## Notes

- Default ALRD configs use intentionally small `train_size` and `total_steps`. Raise them after the smoke path is stable.
- Response KD is computed in flow-matching velocity space, not pixel space.
- Teacher splits must use a verified action statistic: Push Cube has fixed action
  magnitude, so its useful split is signed action direction rather than magnitude.

## Current Status

See [docs/current_status.md](docs/current_status.md) for the current idea assessment, implemented scaffold, validation results, and next experiment steps.

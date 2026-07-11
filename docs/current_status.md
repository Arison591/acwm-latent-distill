# ACWM Latent Distillation Current Status

Date: 2026-07-07

## Idea Status

The idea is still worth pursuing, but the useful version is narrower than the original wording "distill from latent".

The sharp research question is:

> Can a student ACWM preserve specialist teachers' action response under counterfactual actions?

This is stronger than ordinary world-model distillation because it asks whether the model remains controllable after distillation, not just whether predicted videos look plausible. The most important evaluation target is therefore action response: `pred(a)` should differ meaningfully from `pred(0)`, `pred(-a)`, and `pred(0.5a)` in physically plausible ways.

The idea has three defensible pieces:

- Continuous action latent can make the action interface more expressive than raw low-dimensional actions, especially for action chunks.
- Small/large action-regime teachers may learn local action-to-dynamics mappings better than one generalist.
- Response KD can discourage the student from becoming visually smooth but action-blind.

The key risks are also clear:

- Raw action may already be strong on low-dimensional environments such as Push Cube and Reacher.
- Small/large magnitude splits may not create stronger teachers; this must be tested before trusting distillation.
- Response KD can copy teacher errors if the teachers are weak or unreliable on counterfactual actions.

Current recommendation: continue with the small Push Cube/Reacher protocol first. Do not jump to VQ/RVQ, high-dimensional robot tasks, or large-scale training until oracle specialists show a real advantage.

## What Was Implemented

The project was initialized at:

```bash
/root/acwm-latent-distill
```

It is based on:

```bash
https://github.com/xavihart/ACWM-Phys-dev
```

Current branch:

```bash
main
```

Implemented components:

- `acwm/action_latent/`
  - `IdentityActionEncoder`
  - `MLPActionEncoder`
  - `ConvActionEncoder`
  - action magnitude bucket assignment
  - counterfactual actions: `zero`, `reverse`, `scale_0_5`, optional `scale_2`, `shuffle`
  - dataset filtering for small/large specialist teacher training
- `acwm/distill/`
  - prediction loss
  - prediction KD loss
  - response KD loss
  - teacher checkpoint state-dict loading helpers
- `acwm/dynamics/diffusion_forcing_wm.py`
  - optional `model_config.action_encoder`
  - `encode_action(...)`
  - `training_outputs(...)` so student and teacher can share noise/timestep for KD
- `acwm/trainer/train_dynamics.py`
  - fixed model-config path loading
  - action-bucket train filtering
  - optional teacher loading
  - optional `L_pred + lambda_kd * L_kd + mu_resp * L_resp` training path
- `configs/alrd/`
  - `push_cube_latent.yaml`
  - `push_cube_teacher_small.yaml`
  - `push_cube_teacher_large.yaml`
  - `push_cube_student_resp_kd.yaml`
- `scripts/alrd/`
  - `inspect_actions.py`
  - `make_action_buckets.py`
  - `eval_action_ablation.py`
  - `run_push_cube_smoke.sh`
- `docs/`
  - `experiment_protocol.md`
  - `go_no_go.md`
  - this status document

Large files are ignored through `.gitignore`.

## Validation Already Run

These checks passed:

```bash
python -m compileall acwm scripts/alrd tests/alrd
python tests/alrd/test_smoke.py
bash scripts/alrd/run_push_cube_smoke.sh
```

Update on 2026-07-07: the local workspace now has the Push Cube smoke subset under `data/`,
`checkpoints/Wan2.1_VAE.pth`, and offline wandb configured through `.env.local`.

The real smoke path has been run for:

```bash
python train.py --config configs/alrd/push_cube_latent.yaml
python train.py --config configs/alrd/push_cube_teacher_small.yaml
python train.py --config configs/alrd/push_cube_teacher_large.yaml
python train.py --config configs/alrd/push_cube_student_resp_kd.yaml
```

Generated checkpoints:

```bash
checkpoints/alrd_push_cube_latent_smoke_240x240/latest.pt
checkpoints/alrd_push_cube_teacher_small_240x240/latest.pt
checkpoints/alrd_push_cube_teacher_large_240x240/latest.pt
checkpoints/alrd_push_cube_student_resp_kd_240x240/latest.pt
```

Two important fixes landed during setup:

- sampled datasets now preserve original trajectory ids after `max_trajs` sampling;
- Push Cube action magnitude is constant, so bucket splitting falls back to `signed_action_0`.

## Validated Update (2026-07-11)

See [ALRD Validation Report](alrd_validation_2026-07-11.md) for the full protocol and
numbers. The key result is a narrowed but positive feasibility signal: on Push Cube,
signed-target-coordinate-specialist response KD improved paired `pred(a)` versus `pred(-a)` sensitivity
by over two orders of magnitude relative to a matched latent-only baseline on both ID
and OOD subsets, while keeping true-action rollout quality essentially unchanged.
The current paired eval uses the usable-train signed-coordinate threshold `signed_action_0 =
-0.018154`; ID/OOD eval videos are complete, while local `ind_train` is partially
repaired to 1129/1500 available videos on the data disk.

This replaces the old small/large-magnitude wording for Push Cube. Its action magnitude
is constant, so the feasibility split is an absolute-target signed-coordinate regime,
not demonstrated motion direction. The opposite signed-coordinate teacher was
action-blind and must not be used for distillation until it passes the oracle gate.

## Next Steps

1. Set paths:

```bash
export ACWM_DATA_ROOT=/path/to/ACWM-Phys
export WAN_VAE_PATH=/path/to/Wan2.1_VAE.pth
```

For this machine, use:

```bash
source .env.local
```

2. Inspect Push Cube action statistics:

```bash
python scripts/alrd/inspect_actions.py --env push_cube --split ind_train
python scripts/alrd/make_action_buckets.py --env push_cube --split ind_train
```

3. Run the latent-action smoke training:

```bash
WANDB_MODE=disabled python train.py --config configs/alrd/push_cube_latent.yaml
```

4. Train specialist teachers:

```bash
WANDB_MODE=disabled python train.py --config configs/alrd/push_cube_teacher_small.yaml
WANDB_MODE=disabled python train.py --config configs/alrd/push_cube_teacher_large.yaml
```

5. Only if specialists beat the generalist inside their buckets, train the response-KD student:

```bash
WANDB_MODE=disabled python train.py --config configs/alrd/push_cube_student_resp_kd.yaml
```

6. Evaluate action response:

```bash
python scripts/alrd/eval_action_ablation.py \
  --env push_cube \
  --cfg configs/alrd/push_cube_student_resp_kd.yaml \
  --ckpt checkpoints/alrd_push_cube_student_resp_kd_240x240/latest.pt \
  --steps 20 \
  --save_videos
```

## Go / No-Go Reminder

Continue if:

- latent-action baseline is competitive with raw action;
- small/large teachers beat the generalist in their own buckets;
- response KD improves `zero`, `reverse`, and `scale_0_5` action ablations;
- visual metrics do not collapse.

Stop or redesign if:

- specialists are not stronger than the generalist;
- `pred(a)`, `pred(0)`, and `pred(-a)` remain nearly identical;
- response KD only creates larger differences without better real-action rollout.

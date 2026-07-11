# ALRD Experiment Protocol

## 历史协议（Push Cube；非当前默认路径）

下列 Push Cube 流程作为既有可复现实验保留，不得覆盖 Reacher 的负结论。当前
Robot Arm 的 response-first 协议见 `robot_arm_response_probe.md` 与
`robot_arm_oracle_gate.md`。

## 2026-07-11 protocol outcome

The ACWM-Phys specialist/KD path is now a no-go under the response-first gates. Push
Cube remains a narrow signed-target-coordinate feasibility result, Reacher rejects the
magnitude partition, and Robot Arm fails Gate A. Do not run specialist training or KD
from this protocol unless a future high-DoF dataset first passes Gate A.

## Order

1. **Raw ACWM-DiT baseline**
   - Run the original `configs/envs/push_cube.yaml`.
   - Record MSE, masked MSE, PSNR, SSIM, and action ablation behavior.

2. **Continuous action-latent baseline**
   - Run `configs/alrd/push_cube_latent.yaml`.
   - Compare against raw action before adding teachers.

3. **Oracle specialist teachers**
   - Run `push_cube_teacher_small.yaml` and `push_cube_teacher_large.yaml`.
   - A specialist must beat the generalist inside its own bucket before distillation is worth running.
   - First inspect action magnitudes. If they are constant (as in Push Cube), define
     buckets by an action-semantics-aware axis. For Push Cube, `signed_action_0` is a
     signed absolute-target-coordinate regime, not motion direction; do not describe
     it as small/large magnitude or direction.

4. **Prediction KD student**
   - Use the student config with `mu_resp: 0.0`.
   - This isolates ordinary teacher-student matching.

5. **Response KD student**
   - Use `mu_resp: 0.5` with `zero`, `reverse`, and `scale_0_5`.
   - Compare action ablation and action-inference metrics against prediction KD.

## Required Metrics

- ACWM-Phys metrics: MSE, masked MSE, PSNR, SSIM.
- Action sensitivity: `MSE(pred(a), pred(0))`.
- Negation response: compare `pred(a)` and `pred(-a)` and interpret it according to
  environment semantics (negated target for Push Cube, opposite torque for Reacher).
- Scale behavior: compare `pred(0.5a)`, `pred(a)`, and later `pred(2a)`.
- Visual grid: input, GT, `pred(a)`, `pred(0)`, `pred(-a)`, `pred(0.5a)`.
- Pair every action variant with the exact same initial diffusion noise. Otherwise
  action-sensitivity is confounded by sampling variance.
- Verify that every metadata-referenced video exists before reporting metrics. Missing
  videos must be repaired or excluded, never converted to all-zero observations.

## Default Counterfactuals

- `zero`: tests whether the model ignores action.
- `reverse`: tests directional response.
- `scale_0_5`: tests monotonicity without pushing far out of distribution.

Avoid enabling `scale_2` and `shuffle` until the first three are stable.

# ALRD Experiment Protocol

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

4. **Prediction KD student**
   - Use the student config with `mu_resp: 0.0`.
   - This isolates ordinary teacher-student matching.

5. **Response KD student**
   - Use `mu_resp: 0.5` with `zero`, `reverse`, and `scale_0_5`.
   - Compare action ablation and action-inference metrics against prediction KD.

## Required Metrics

- ACWM-Phys metrics: MSE, masked MSE, PSNR, SSIM.
- Action sensitivity: `MSE(pred(a), pred(0))`.
- Direction consistency: compare `pred(a)` and `pred(-a)`.
- Scale behavior: compare `pred(0.5a)`, `pred(a)`, and later `pred(2a)`.
- Visual grid: input, GT, `pred(a)`, `pred(0)`, `pred(-a)`, `pred(0.5a)`.

## Default Counterfactuals

- `zero`: tests whether the model ignores action.
- `reverse`: tests directional response.
- `scale_0_5`: tests monotonicity without pushing far out of distribution.

Avoid enabling `scale_2` and `shuffle` until the first three are stable.

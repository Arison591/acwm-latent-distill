# ALRD Go / No-Go Criteria

## Continue If

- The latent-action baseline matches or beats raw action on Push Cube/Reacher without breaking visual metrics.
- `small` and `large` teachers outperform the generalist in their own action buckets.
- Response KD improves action ablation metrics over prediction-only KD.
- `pred(a)` and `pred(-a)` produce visibly different and directionally plausible futures.
- Action-sensitivity gains do not come with a large PSNR/SSIM collapse.

## Current 2026-07-11 Gate

Push Cube has constant action magnitude, so `small`/`large` is actually a signed
target-coordinate split there. The signed-target-coordinate response-KD run is a **go**:
under paired-noise ablation it improved `MSE(pred(a), pred(-a))` by over two orders of
magnitude on both ID and repaired OOD signed-coordinate subsets without a meaningful
true-action rollout loss. Continue only with the responsive specialist; do not use the
action-blind opposite-direction teacher as a distillation target.
Before scaling, finish the local Push Cube train-video repair: the current machine has
1129/1500 `ind_train` videos available after linking data-disk downloads, so bucket
thresholds and new training runs must be rechecked once the split is complete.

## Stop Or Redesign If

- Specialist teachers are weaker than the generalist after comparable compute.
- Response KD only increases prediction differences but makes real-action rollout worse.
- `pred(a)`, `pred(0)`, and `pred(-a)` remain almost identical.
- The action encoder helps only by increasing model capacity, with no improvement in counterfactual response.
- Push Cube works but Reacher fails completely under the same protocol.

## Reacher 2026-07-11 Gate

Reacher action magnitude is non-degenerate and balanced, but it is not a useful
specialist axis: on 1,000 complete train episodes, high/low visual-motion proxy ratio
is only `1.0705` and torque/motion correlation is `r=0.0546`. The official baseline is
already strongly action-responsive across three seeds. Outcome: **C, redesign
partition**. No specialist or KD run is allowed on this rejected partition. Move to
Robot Arm and derive one action-semantics-aware failure axis before training teachers.

## First Redesigns

- Use partitions derived from verified environment action semantics; reserve magnitude
  splits for datasets with verified magnitude variation and verified motion effect.
- Keep continuous MLP action latent, but add an auxiliary transition-prediction head.
- Raise `mu_resp` from `0.5` to `1.0` only after teacher quality is established.
- Move response KD from velocity space to VAE latent rollout space if velocity matching is too noisy.

# ALRD Go / No-Go Criteria

## Continue If

- The latent-action baseline matches or beats raw action on Push Cube/Reacher without breaking visual metrics.
- `small` and `large` teachers outperform the generalist in their own action buckets.
- Response KD improves action ablation metrics over prediction-only KD.
- `pred(a)` and `pred(-a)` produce visibly different and directionally plausible futures.
- Action-sensitivity gains do not come with a large PSNR/SSIM collapse.

## Stop Or Redesign If

- Specialist teachers are weaker than the generalist after comparable compute.
- Response KD only increases prediction differences but makes real-action rollout worse.
- `pred(a)`, `pred(0)`, and `pred(-a)` remain almost identical.
- The action encoder helps only by increasing model capacity, with no improvement in counterfactual response.
- Push Cube works but Reacher fails completely under the same protocol.

## First Redesigns

- Replace median magnitude split with direction split for 2D action environments.
- Keep continuous MLP action latent, but add an auxiliary transition-prediction head.
- Raise `mu_resp` from `0.5` to `1.0` only after teacher quality is established.
- Move response KD from velocity space to VAE latent rollout space if velocity matching is too noisy.

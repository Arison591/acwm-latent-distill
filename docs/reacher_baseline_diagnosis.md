# Reacher Paired-Noise Baseline Diagnosis

## Verdict: highly action-responsive baseline / practical ceiling effect

The official 100k-step Reacher DiT-S checkpoint was evaluated with the same observation
and identical initial diffusion noise for `a`, `0`, `-a`, `0.25a`, `0.5a`, `0.75a`,
`1.5a`, and `2a`. Each split uses eight deterministic windows, 10 denoising steps and
three diffusion seeds. ID/OOD videos are complete and decoded; seed 0 also persists
four video grids per split.

The model already has absolute paired action responses around `1e-3`, several orders
above the Push Cube baseline response around `2e-7`. There is no credible room to use
near-zero baseline sensitivity as the Reacher improvement story. This is a practical
ceiling effect for the proposed response-amplification objective, not a mathematical
upper bound on response MSE.

## True-action quality (three-seed mean ± std)

| Split | MSE | masked MSE | PSNR | SSIM |
| --- | ---: | ---: | ---: | ---: |
| ID | `3.787e-4 ± 1.73e-5` | `8.167e-3 ± 4.66e-4` | `34.303 ± 0.196` | `0.99098 ± 0.00017` |
| OOD | `1.3090e-3 ± 7.53e-7` | `2.2940e-2 ± 3.64e-5` | `28.8310 ± 0.0024` | `0.978894 ± 0.000008` |

These are factual-action rollout metrics. Metrics computed against factual GT for a
counterfactual action are not physical-correctness metrics and are not interpreted as
such.

## Absolute paired response (three-seed mean ± std)

`MSE(pred(a), pred(a_cf))`:

| Transformation | Reacher semantics | ID | OOD |
| --- | --- | ---: | ---: |
| `0` | zero torque | `1.37146e-3 ± 2.89e-7` | `1.42452e-3 ± 9.06e-7` |
| `-a` | opposite torques | `1.12608e-3 ± 1.04e-6` | `1.15131e-3 ± 1.33e-6` |
| `0.25a` | quarter torque | `1.30544e-3 ± 3.34e-7` | `1.31785e-3 ± 1.51e-6` |
| `0.5a` | half torque | `1.20894e-3 ± 7.38e-7` | `1.25122e-3 ± 1.53e-6` |
| `0.75a` | three-quarter torque | `1.08232e-3 ± 2.31e-6` | `1.15242e-3 ± 1.47e-6` |
| `1.5a` | one-and-a-half torque | `8.61648e-4 ± 2.23e-6` | `9.33648e-4 ± 4.85e-6` |
| `2a` | doubled torque | `6.38167e-4 ± 1.91e-6` | `3.89697e-3 ± 2.26e-5` |

Every response direction is consistent across all three diffusion seeds with very
small seed variance. This establishes strong model output sensitivity, not matched
counterfactual physical correctness.

## Scale structure

The declared monotonicity criterion is non-decreasing response with `|alpha-1|`,
excluding equal-distance ties. ID satisfies 3/4 comparable pairs for every seed
(`0.75 ± 0.00` fraction). Its extrapolation branch is non-monotonic: response at `2a`
is lower than at `1.5a`. OOD satisfies 4/4 for every seed (`1.00 ± 0.00`) and has a
large `2a` response, plausibly reflecting stronger out-of-range behavior.

Thus the baseline is responsive, but sensitivity alone does not guarantee a simple
or physically correct response curve. No metric was altered to manufacture headroom.

## Artifacts

- Per-seed JSON summaries:
  `results/alrd_action_ablation/reacher/{ind_test,ood_test}/VideoDiT_S_reacher_240x240/seed_*/summary.json`
- Three-seed aggregates and response plots:
  `aggregate_3seeds.json` and `action_response_curve_3seeds.png` in each split directory
- Seed-0 video grids: `seed_0/sample_*.mp4`

Together with the failed magnitude-regime gate, this baseline result supports moving
away from Reacher specialist response amplification rather than tuning the loss to
force an improvement.

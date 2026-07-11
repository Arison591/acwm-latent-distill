# ALRD Validation Report — 2026-07-11

## Verdict

**Go, with a narrower hypothesis.** On Push Cube, a signed-target-coordinate specialist
teacher plus response KD preserves a measurable action response in that regime. The
original small/large-magnitude ensemble formulation does **not** work: Push Cube has
unit-norm actions, so its magnitude split is degenerate and the fallback is actually a
signed first target-coordinate split.

The supported claim is therefore:

> With paired diffusion noise, response KD from an action-responsive signed-target-coordinate
> specialist can increase a student's counterfactual action response without a
> material rollout-quality loss in the specialist's regime.

It is not yet evidence for a general multi-specialist or all-action solution.

## Protocol and resource guardrails

- Environment: ACWM-Phys Push Cube, DiT-S, 100 optimizer steps, 256 sampled train
  trajectories.
- Evaluation: fixed input, fixed initial diffusion noise across `a`, `0`, `-a`, and
  `0.5a`; 10 denoising steps; seed 123; 8 batches on the indicated action bucket.
- Bucket: the current usable train-set median of the mean first action coordinate
  (`-0.018154` after skipping still-missing train videos), called `small` for
  backward compatibility. It is a **signed target-coordinate regime** bucket, not an
  action-magnitude or demonstrated motion-direction bucket.
- No full checkpoints were written during controls or ablations. Weight-only exports
  were temporary files on `/dev/shm`; all persistent OOD video data was placed on the
  separate `/root/autodl-tmp` data disk.
- The originally incomplete OOD video split was repaired before reporting OOD metrics:
  34 missing Push Cube OOD videos were downloaded from `t1an/ACWM-Phys` to the data
  disk and linked into the local split. Missing videos now cause a dataset skip/fail
  rather than silently becoming zero frames.
- The local `ind_train` video split was also partially repaired without growing the
  system disk: 545 missing train videos were downloaded to `/root/autodl-tmp` and
  linked into `data/`, raising availability from 584/1500 to 1129/1500 trajectories.
  The remaining 371 train videos were not downloaded because unauthenticated HF
  requests repeatedly stalled or hit TLS EOFs; larger runs must complete this split
  first and then rerun bucket estimation.

## Main results

For Push Cube, generic API mode `reverse` means a **negated-target action**, because
the action encodes an absolute target position; it is not an established reverse
motion. The response is `MSE(pred(a), pred(-a))` under the same initial diffusion
noise. Larger is only useful when true-action rollout quality is retained.

| Split / signed-target-coordinate bucket | Model | True-action MSE | PSNR | Absolute paired response |
| --- | --- | ---: | ---: | ---: |
| ID | continuous latent baseline | 0.016894 | 17.726 | 1.975e-7 |
| ID | single-specialist response KD | 0.016965 | 17.709 | 2.690e-5 |
| OOD | continuous latent baseline | 0.013030 | 18.891 | 2.068e-7 |
| OOD | single-specialist response KD | 0.012920 | 18.926 | 3.159e-5 |

The absolute paired response changes from `1.975e-7` to `2.690e-5` ID and from
`2.068e-7` to `3.159e-5` OOD: about **136× ID** and **153× OOD** over the matched
latent-only baseline. Its ID MSE changes by less than 0.5%; its OOD MSE is slightly
lower. Paired noise makes this evidence of model output sensitivity to the negated
target rather than diffusion sampling variance. It does **not** prove action
controllability or physical correctness.

The persistent evaluation artifacts did not retain the responsive teacher's absolute
paired response, so teacher response and student/teacher response retention are
unavailable rather than inferred. Reacher evaluations must persist both explicitly.

The continuous action-latent interface itself also showed a modest visual benefit in
the same 100-step control: ID rollout MSE `0.01594` versus `0.01742` for raw actions
(about 8.5% lower), with PSNR `17.97` versus `17.59`.

## Failed variants and diagnosis

1. The original two-teacher ensemble is not viable yet. Its `large` direction teacher
   had near-zero action response (`~1e-6` in paired ablation), so it is not an oracle
   worth distilling.
2. Raising `mu_resp` from 0.5 to 2.0 while retaining both teachers did not solve this:
   negated-target response remained `5.61e-6` over the full test subset. More loss weight
   cannot repair an action-blind teacher target.
3. Restricting training and KD to the responsive specialist raised sensitivity to
   `2.69e-5` while retaining rollout quality. This identifies teacher quality and
   regime gating—not response-loss weight—as the critical design variables.

## Current limitations

- Only one environment, one signed target-coordinate regime, one seed, and 100 training steps
  have been tested. This is a feasibility result, not a paper-scale claim.
- The current local train split is still incomplete (1129/1500 videos available).
  The reported ID/OOD eval subsets are complete, but any new training or teacher
  comparison should first finish the train-video repair and re-estimate buckets.
- There is no ground-truth counterfactual rollout for `-a` or `0.5a`; the metric proves
  response preservation, while physical correctness still needs an environment with
  counterfactual labels or simulator rollouts.
- The teacher still has a larger response than the student, so distillation is partial.
- Push Cube's fixed action magnitude makes it unsuitable for testing magnitude experts.
  Reacher or an environment with genuine magnitude variation is required before
  claiming a magnitude-specialist method.

## Next scale-up gate

Keep the single-specialist formulation only if it reproduces across at least three
seeds and then on Reacher (or another environment with a non-degenerate regime split).
For each setting, require paired action ablations, true-action metrics, an oracle
teacher check, and complete video files before any larger run.

## Reacher migration update

The Reacher metadata audit is now implemented and recorded in
`docs/reacher_action_semantics.md`. From 40,000 real train metadata entries, per-chunk
torque norm has mean `2.0721`, std `0.5756`, and coefficient of variation `0.2778`.
The median split is exactly balanced (20,000/20,000), so action magnitude itself is
non-degenerate. Train/ID/OOD marginal norm distributions are close by standardized
mean difference.

Authenticated downloads completed a version-consistent feasibility inventory of
1,000 train, 100 ID and 100 OOD videos. On the aggregate metadata, the magnitude split
is balanced but does not define a meaningful visual-motion regime: high/low motion
proxy ratio is only `1.0705`, with episode-level correlation `r=0.0546`. The proposed
low/high specialist partition therefore fails before training. Neither teacher is
eligible for distillation, and all KD runs remain deliberately unexecuted. The
independent official-baseline diagnosis awaits completion of its 3.1 GB checkpoint.

The checkpoint was subsequently completed and evaluated with paired noise on ID/OOD,
eight windows, 10 denoising steps and three seeds. The baseline is already strongly
action-responsive: zero/opposite-torque response is approximately `1.37e-3/1.13e-3`
ID and `1.42e-3/1.15e-3` OOD, with very small seed variance. True-action MSE is
`3.79e-4` ID and `1.309e-3` OOD. ID scale monotonicity is only 3/4 because the `2a`
response falls below `1.5a`; OOD is 4/4 and reacts sharply at `2a`.

## Final Reacher decision

**C. specialist hypothesis fails; redesign partition.** This is not a response-KD
failure result: response KD was correctly prohibited because neither hypothetical
magnitude teacher passed the partition/oracle precondition. Reacher also exhibits a
practical baseline-response ceiling, so forcing additional sensitivity would not
answer the physical-structure question.

The only recommended next research route is Robot Arm with a partition derived from
its verified joint-angle-delta semantics and measured visual-motion/failure axis.
Repeat diagnosis and oracle gating there; do not carry over Reacher's median magnitude
split or tune response loss on Reacher.

# When Should Action-Conditioned World Models Use Specialists? A Negative Study on Response-Structure Gates in ACWM-Phys

## Abstract

This report records a negative result for the ACWM latent-distillation branch. The
original idea was to use action-magnitude partitions to train specialist teachers and
then transfer their behavior through response KD. That idea is no longer supported by
the evidence in this repository.

The revised protocol became response-first: before defining specialists, the dense
baseline must show a stable local action-response failure or anisotropy under paired
diffusion noise. Push Cube showed only a narrow signed-target-coordinate feasibility
signal. Reacher rejected the magnitude partition before teacher training and then
showed a dense baseline that was already strongly action-responsive. Robot Arm, the
instrumentation MVP, completed Gate A with the official dense checkpoint and failed:
full temporal action shuffle changes the model output, but per-dimension local
response is below the preregistered noise-floor threshold and random grouping controls
do not support stable anisotropy.

The protocol therefore prohibits specialist training and KD on the current
ACWM-Phys evidence. The useful output of this branch is a response-aware evaluation
and structured action-diagnostic toolkit, not a validated specialist-distillation
method. Future specialist or response-KD work should restart only after a true
high-DoF heterogeneous action dataset first passes Gate A.

## Motivation

Good factual-action video prediction does not prove that an action-conditioned world
model has learned the correct local action-response field. A model can predict the
observed action trajectory well while remaining insensitive, over-smoothed, or
miscalibrated under counterfactual actions. This matters for distillation: a student
that matches a teacher only at factual actions may preserve image quality while losing
controllability-relevant response structure.

Specialists are only justified if the action space contains a measured response
structure that a dense model under-models. A statistical partition, such as low versus
high action magnitude, is not enough. It must correspond to a response or failure axis
that is visible under controlled paired evaluation. This is why the active protocol
requires response probing before teacher training.

## Protocol

The revised pipeline is response-first, not specialist-first:

```text
Baseline response probe
  -> Gate A: baseline response measurability and local anisotropy
  -> teacher admission against a comparable dense baseline
  -> prediction KD / response KD student comparison
```

The implemented instrumentation is intentionally reusable rather than Robot Arm
specific:

- `ActionSchema` records action dimensions, groups, representation type, empirical
  group statistics, masks, and semantic provenance.
- `ActionPerturbationSampler` produces repeatable perturbations from the schema:
  full temporal shuffle, within-group temporal shuffle, group masking, local additive
  perturbations, and signed local directions. For `unknown` action semantics it
  refuses physical interpretations such as zero-action controls.
- `ResponseProbe` runs paired model evaluations and saves per-window records.
- `CounterfactualEvaluator` computes paired response, noise-floor ratios,
  finite-difference directional responses, and teacher/student paired-response MSE.
- Random dimension grouping controls are generated with
  `scripts/response_structure/make_random_group_schema.py`.
- `scripts/response_structure/gate_a.py` emits a machine-readable Gate A decision.
- Three-seed aggregation, response heatmaps, response distributions, and ID/OOD
  degradation summaries are produced by the response-structure scripts.
- `scripts/response_structure/download_hf_file_ranges.py` was added to download a
  single large Hugging Face checkpoint with resumable byte ranges on the data disk.

Paired model evaluation always uses the same observation/history and identical
initial diffusion noise across factual and perturbed actions. An intentionally
unpaired same-action prediction supplies a sampling-noise floor. This prevents random
diffusion variation from being mistaken for action response.

Gate A asks whether the official dense baseline measurably depends on action, whether
temporal action shuffle or group masking rises above the paired noise floor, whether
local group/direction differences are stable across three seeds, and whether those
differences are larger than random grouping variance. If Gate A fails, specialist
training and KD are prohibited.

## Push Cube

Push Cube killed the original action-magnitude assumption. The action magnitude is
effectively constant, so the `small`/`large` terminology in early configs does not
describe magnitude specialists. The useful split became a signed first target
coordinate, with threshold `signed_action_0 = -0.018154` in the repaired local train
subset.

The narrow positive signal was real but limited. In paired-noise evaluation,
single-specialist response KD increased `MSE(pred(a), pred(-a))` from `1.975e-7` to
`2.690e-5` on ID and from `2.068e-7` to `3.159e-5` on OOD, roughly `136x` and `153x`
over the matched latent-only baseline. True-action rollout quality was not materially
hurt: ID MSE changed from `0.016894` to `0.016965`, and OOD MSE changed from
`0.013030` to `0.012920`.

This result is useful as a feasibility check for paired response evaluation. It is
not evidence for a general specialist claim. It uses one narrow signed-target regime,
one seed, short training, and no matched physical counterfactual ground truth.

Primary source: `docs/alrd_validation_2026-07-11.md`.

## Reacher

Reacher has externally verified two-dimensional torque actions. Here, magnitude was
non-degenerate: the 1,000-episode feasibility subset split exactly `500/500` at the
median, and the aggregate 40,000-entry metadata split was also balanced. But a
non-degenerate magnitude distribution did not define a meaningful motion-effect
regime. The high-torque group had only `1.0705x` the visual-motion proxy of the
low-torque group, and the episode-level action/motion correlation was only
`r = 0.0546`.

The oracle gate therefore rejected the magnitude partition before teacher training.
The official dense Reacher baseline then showed a practical ceiling for the proposed
response-amplification story: paired responses were already around `1e-3`, with
zero/opposite-torque response approximately `1.37146e-3` / `1.12608e-3` on ID and
`1.42452e-3` / `1.15131e-3` on OOD. True-action quality was strong as well:
three-seed MSE was `3.787e-4` ID and `1.3090e-3` OOD.

This is not a response-KD failure, because KD was correctly not run. It is a
partition-precondition failure plus a dense-baseline response ceiling.

Primary sources: `docs/reacher_action_semantics.md`,
`docs/reacher_oracle_gate.md`, and `docs/reacher_baseline_diagnosis.md`.

## Robot Arm Gate A

Robot Arm is the main instrumentation benchmark for the response-first pivot. The
official dense checkpoint was downloaded and verified readable:

```text
/root/autodl-tmp/acwm-response-checkpoints/VideoDiT_S_robot_arm_240x240/latest.pt
sha256: 438303c23acfd153ac42ddb8eb84cc412687432242bafcb1276b85f03934c6f7
```

The action schema records `action_dim = 7`, dimension names `dim_0` through `dim_6`,
and `representation = unknown`. The current workspace did not contain verifiable
per-dimension Robot Arm control semantics, so the schema uses stable dimension-index
groups and does not invent joint names.

Data audit evidence:

- Dataset revision: `4017e5f5900a7a8590d4e944b6a84f55df75a0c1`.
- ID eval split: `105/105` videos present, complete.
- OOD eval split: `105/105` videos present, complete.
- Train inventory in `action_audit.json`: `688/2002` videos present and `1314`
  missing at audit time; background repair continued, but Gate A was not blocked
  because the evaluated ID/OOD splits were complete.
- OOD action norm distribution is shifted relative to train: standardized mean
  difference `0.8459475592843546`, std ratio `1.419562980290758`.

Gate A used three seeds `[0, 1, 2]`, paired initial diffusion noise, ID/OOD probe
aggregates, and random dimension grouping controls. The preregistered threshold was a
minimum response/noise-floor ratio of `2.0`.

The full temporal action shuffle gap was measurable:

| Split | full temporal shuffle gap mean | std |
| --- | ---: | ---: |
| ID | `0.008114452473819256` | `0.0005309778574601698` |
| OOD | `0.007564159808680415` | `0.0004861596680240644` |

This means the model is not globally action-invariant under a severe temporal action
perturbation. However, the local per-dimension response ratios are far below the Gate
A threshold. The largest local ratio is `dim_1`:

| Split | best local group | response/noise-floor ratio mean | std |
| --- | --- | ---: | ---: |
| ID | `dim_1` | `0.31307352675745886` | `0.1727052039500961` |
| OOD | `dim_1` | `0.3975854863723119` | `0.21366368496429508` |

All other dimensions are lower. For example, ID ratios are `0.0335263` for `dim_0`,
`0.0229595` for `dim_2`, `0.0130471` for `dim_3`, `0.0000463` for `dim_4`,
`0.0159808` for `dim_5`, and `0.0014136` for `dim_6`. OOD ratios are similarly
below threshold.

The random grouping control also rejects the anisotropy story. Its mean
response/noise-floor ratio is `0.05714959195879709`, std is
`0.12437334891438451`, and `stable_groups` is empty. The machine decision is:

```text
decision: fail
reason: 未检出稳定且高于噪声地板、并超过随机分组方差的动作响应；必须先诊断条件路径或评估器。
```

The precise conclusion is not "Robot Arm has no action response." The precise
conclusion is: full temporal action shuffle produces measurable output changes, but
the probe did not find stable local response anisotropy above the paired-noise floor
and random grouping variance. Therefore Robot Arm does not admit specialist teachers
or KD under the current protocol.

Primary sources:
`results/response_structure/robot_arm/gate_a/decision.json`,
`results/response_structure/robot_arm/probe/ind_test_aggregate.json`,
`results/response_structure/robot_arm/probe/ood_test_aggregate.json`,
`results/response_structure/robot_arm/probe/id_ood_response_degradation.json`,
`results/response_structure/robot_arm/action_audit/action_audit.json`, and
`results/response_structure/robot_arm/action_audit/action_schema.json`.

## What Failed

- Magnitude partitioning as a default expert axis.
- Training sample-bucket specialists before measuring baseline response structure.
- Running KD before a teacher has passed admission against a comparable dense
  baseline.
- Treating a larger `MSE(pred(a), pred(a_cf))` as physical correctness.
- Treating Robot Arm as sufficient evidence for response-subspace specialists.
- Rescuing a failed gate by threshold tuning or unrestricted partition search.

## What Did Not Fail

- Response-aware evaluation. The branch produced reusable paired-noise probing,
  aggregation, plots, and machine-readable gates.
- Structured action representation as an engineering direction.
- The possibility that true high-DoF heterogeneous action spaces need different action
  encoders or local response diagnostics.
- Future DexJoCo/EgoDex-style investigations, provided they start with official action
  semantics and pass Gate A before teacher training.

## Limitations

- ACWM-Phys action dimensionality is limited. Robot Arm is 7D and is not a true
  dexterous high-DoF arm-hand benchmark.
- Robot Arm action semantics remain unresolved in the current workspace, so the MVP
  uses dimension-index groups rather than verified joint or end-effector groups.
- There is no matched physical counterfactual ground truth in the released ACWM-Phys
  dataset/code. Current response metrics are model-output probes, not proof of
  physical correctness.
- The local-response threshold of `2.0` may be conservative, but changing it after the
  fact would invalidate the gate.
- Robot Arm train videos were still being completed. The ID/OOD evaluation splits were
  complete, and Gate A was not blocked, but future training would require a complete
  train inventory.
- The Robot Arm probe used a bounded MVP evaluation setting. A larger evaluation could
  refine diagnostics, but it cannot retroactively justify specialist/KD without
  passing the preregistered gate.

## Future Work

Freeze the ACWM-Phys specialist/KD direction as a negative study. Do not run more
Reacher KD, do not revive magnitude specialists, and do not train Robot Arm specialists
from the current Gate A failure.

The next legitimate specialist/response-KD attempt should start on a true high-DoF
heterogeneous action dataset. A DexJoCo-style arm-hand benchmark is the preferred
direction: first inspect the official action format, define `ActionSchema` groups from
verified semantics, run paired-noise Gate A, compare against random grouping controls,
and only then consider teacher admission or KD.

The immediate publishable output is therefore a response-aware diagnostic and
evaluation study: it clarifies when structured action encoding or specialist response
transfer is unsupported, and it preserves the tooling needed to test stronger
benchmarks without repeating the same mistake.

## Reproducibility and Security Notes

Validation commands:

```bash
python -m compileall -q acwm/action_latent scripts/response_structure tests/alrd
python tests/alrd/test_smoke.py
```

Large Robot Arm artifacts were stored under `/root/autodl-tmp` through the
`results/response_structure` symlink to avoid filling the system disk. The Hugging
Face token used to download the official Robot Arm checkpoint was removed from
`.env.local`; the final local status check was `HF_TOKEN_ABSENT`.

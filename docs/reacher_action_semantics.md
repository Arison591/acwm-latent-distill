# Reacher Action Semantics and Data Audit

## Gate decision

**Reacher torque magnitude is non-degenerate, but a low/high-magnitude specialist
partition is rejected.** The split is balanced, yet torque magnitude is only weakly
associated with future visual motion in the version-consistent metadata/video subset.
Per the protocol, no low/high specialist is trained.

The paper defines Reacher as a two-link MuJoCo arm whose two action dimensions are
joint torques. Thus `0`, `-a`, `0.5a`, and `2a` mean zero torque, opposite joint
torques, half torque, and doubled torque. This semantics is specific to Reacher.

## Reproducible evidence

```bash
python scripts/alrd/inspect_action_semantics.py \
  --env reacher \
  --max_episodes 1000 \
  --analyze_motion \
  --output_dir results/alrd_action_stats/reacher
```

The machine-readable result and plots are in
`results/alrd_action_stats/reacher/`. The evaluated file inventory is complete:

| Split | evaluated metadata entries | videos present | videos missing |
| --- | ---: | ---: | ---: |
| train | 1,000 (first 1,000 from aggregate metadata) | 1,000 | 0 |
| ID | 100 | 100 | 0 |
| OOD | 100 | 100 | 0 |

All 1,200 MP4s were downloaded to the data disk and decoded. The aggregate train
metadata contains 40,000 entries; only the explicit 1,000-entry feasibility subset is
claimed complete. The full aggregate action-only statistics were also inspected and
lead to the same non-degenerate magnitude finding (40,000 entries, 20,000/20,000
median split, threshold about `2.1023`).

## Action distribution

For the complete 1,000-episode train subset, per-step L2 norm has mean about `2.0834`
and std about `1.0194`. Per-episode mean norm has coefficient of variation `0.2776`.
The median threshold `2.1167` yields exactly 500 low and 500 high episodes. This proves
non-degeneracy and balance.

The full 40,000-entry train metadata has near-zero marginal means in both torque
dimensions, near-balanced quadrants, lag-1 autocorrelation around `0.9983`, and sign
transition rates around `1.39%`. Torque trajectories are smooth rather than IID.

ID and OOD step-norm means are `2.1393` and `2.0316`. Relative to the complete
train-subset distribution, standardized mean shifts are only `+0.055` and `-0.051`;
there is no large marginal action-norm shift.

## Torque magnitude versus future motion

The declared visual-motion proxy is the mean absolute difference between consecutive
64×64 grayscale frames, averaged per episode. It is reproducible and measures rendered
motion, but it is not simulator joint displacement.

| Group | episodes | mean visual-motion proxy |
| --- | ---: | ---: |
| low torque | 500 | `3.9106e-4` |
| high torque | 500 | `4.1863e-4` |

The high group is only `1.0705×` the low group, and episode-level Pearson correlation
between mean torque norm and visual motion is `r=0.0546`. This is too weak to treat
median torque magnitude as a real motion-effect regime axis. Magnitude specialists are
therefore disallowed despite the non-degenerate norm distribution.

## Metadata-version conflict

`metadata_part_0.pt` through `metadata_part_4.pt` contain 1,000 entries with 50-step
actions, while aggregate `metadata.pt` contains 40,000 entries with 37-step actions.
For identical video paths and seeds, the action tensors and scales differ. The parts
produce an apparently stronger action/motion correlation (`r≈0.453`), but mixing them
with the aggregate ID/OOD metadata would be invalid. They are retained as a documented
dataset-version conflict and are not used to justify the specialist partition.

## Answered questions

- Action norm non-degenerate: **yes**.
- Median split balanced: **yes**.
- Magnitude split a justified specialist axis: **no**.
- Low/high torque clearly maps to different future motion: **no; only a 7% proxy gap
  and `r=0.055`**.
- Large train/ID/OOD marginal norm shift: **no for aggregate metadata**.

The research pipeline stops low/high specialist training here. Phase 2 paired-noise
baseline diagnosis remains valid and independent of this partition rejection.

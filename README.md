# Action-Response Structure in ACWM-Phys

**A negative study on when action-conditioned video world models should use specialist models.**

This repository is a research fork of
[ACWM-Phys](https://xavihart.github.io/ACWM-Phys). The upstream project provides
the benchmark, datasets, checkpoints, and ACWM-DiT baseline. Our work asks a
different question:

> When is it actually justified to replace one dense action-conditioned world
> model with action-response specialists and distillation?

Our current answer is negative for the tested ACWM-Phys settings. The dense
baseline can show action response, but we did not find a stable response
structure that justifies specialist training under the current protocol.

## Current Conclusion

Specialist training and response-KD are **not active directions** in this repo
unless a future dataset first passes the response-structure gate.

| Environment | Question tested | Main evidence | Decision |
|---|---|---|---|
| Push Cube | Can a simple action partition support response distillation? | Only a narrow signed-target-coordinate feasibility signal was found. The original magnitude split was degenerate. | Feasibility only, not a general specialist result |
| Reacher | Does low/high action magnitude define useful specialists? | The magnitude split was rejected; the official dense baseline was already strongly action-responsive. | No specialist/KD |
| Robot Arm | Is there stable local action-response anisotropy above noise? | Gate A failed: action-shuffle response was measurable, but stable per-dimension response above paired noise and random-group variance was not found. | No specialist/KD |

The project-level result is therefore a **negative result**: on these tested
ACWM-Phys environments, specialist/KD training would be premature and poorly
justified.

## What This Repository Contributes

- A response-structure gate for deciding whether specialist models are justified
  before training them.
- A Robot Arm action audit and response probe over the official dense checkpoint.
- A Reacher baseline diagnosis showing that the dense baseline already responds
  to torque changes.
- A Push Cube feasibility study showing that the only useful signal was narrower
  than the original magnitude-specialist hypothesis.
- A written negative-result report documenting the failure modes and stopping
  criteria.

## Reports and Status

- Full English report:
  [`docs/negative_result_response_structure.md`](docs/negative_result_response_structure.md)
- 中文总结:
  [`docs/negative_result_summary_zh.md`](docs/negative_result_summary_zh.md)
- Current project status:
  [`docs/current_status.md`](docs/current_status.md)
- Go / no-go decision:
  [`docs/go_no_go.md`](docs/go_no_go.md)
- Experiment protocol:
  [`docs/experiment_protocol.md`](docs/experiment_protocol.md)

Additional environment-specific notes:

- [`docs/robot_arm_response_probe.md`](docs/robot_arm_response_probe.md)
- [`docs/robot_arm_action_semantics.md`](docs/robot_arm_action_semantics.md)
- [`docs/robot_arm_oracle_gate.md`](docs/robot_arm_oracle_gate.md)
- [`docs/reacher_baseline_diagnosis.md`](docs/reacher_baseline_diagnosis.md)
- [`docs/reacher_action_semantics.md`](docs/reacher_action_semantics.md)
- [`docs/reacher_oracle_gate.md`](docs/reacher_oracle_gate.md)
- [`docs/alrd_validation_2026-07-11.md`](docs/alrd_validation_2026-07-11.md)
- [`docs/response_kd_results.md`](docs/response_kd_results.md)

## Repository Map

| Path | Purpose |
|---|---|
| `acwm/action_latent/response.py` | Shared response-measurement utilities |
| `scripts/response_structure/` | Robot Arm audit, response probe, aggregation, random-group control, and Gate A scripts |
| `configs/alrd/` | Legacy Push Cube ALRD configs kept as negative-hypothesis artifacts |
| `docs/` | Reports, status notes, go/no-go decisions, and environment diagnostics |
| `results/` | Local generated artifacts; ignored by Git unless explicitly force-added |

## Reproducing the Robot Arm Gate

The checked-in report records the measured results. To rerun the Robot Arm gate,
prepare the ACWM-Phys Robot Arm data and the official Robot Arm checkpoint, then
use the response-structure scripts.

Example layout:

```bash
export ACWM_DATA_ROOT=/path/to/ACWM-Phys
export ROBOT_ARM_CKPT=/path/to/VideoDiT_S_robot_arm_240x240/latest.pt
```

Audit the available Robot Arm data:

```bash
python scripts/response_structure/audit_robot_arm.py \
  --data-root "$ACWM_DATA_ROOT" \
  --output results/response_structure/robot_arm/action_audit/action_audit.json
```

Run response probes for each split and seed:

```bash
python scripts/response_structure/run_response_probe.py \
  --cfg configs/envs/robot_arm.yaml \
  --ckpt "$ROBOT_ARM_CKPT" \
  --schema results/response_structure/robot_arm/action_audit/action_schema.json \
  --split ind_test \
  --seed 0 \
  --steps 50 \
  --max-batches 8 \
  --output results/response_structure/robot_arm/probe/ind_test/seed_0/summary.json
```

Aggregate probe outputs and evaluate Gate A:

```bash
python scripts/response_structure/aggregate_response_probe.py \
  --probe-root results/response_structure/robot_arm/probe/ind_test \
  --output results/response_structure/robot_arm/probe/ind_test_aggregate.json

python scripts/response_structure/gate_a.py \
  --audit results/response_structure/robot_arm/action_audit/action_audit.json \
  --probe-root results/response_structure/robot_arm/probe/ind_test \
  --random-probe-root results/response_structure/robot_arm/probe/random_group_control \
  --output results/response_structure/robot_arm/gate_a/decision.json
```

The recorded Robot Arm decision in this study was `fail`, so no specialist or KD
training was launched from the gate.

## Development Notes

The project keeps upstream ACWM-Phys training and evaluation code available for
context, but the current research branch is documentation- and diagnosis-focused.

For a quick local sanity check:

```bash
python -m compileall -q acwm/action_latent scripts/response_structure tests/alrd
python tests/alrd/test_smoke.py
```

## Upstream ACWM-Phys

This work depends on the original ACWM-Phys benchmark and released checkpoints:

- Project page: <https://xavihart.github.io/ACWM-Phys>
- Dataset: <https://huggingface.co/datasets/t1an/ACWM-Phys>
- Checkpoints: <https://huggingface.co/t1an/ACWM-Phys-checkpoints>

The upstream benchmark covers eight physical environments:

| Category | Environments |
|---|---|
| Rigid-Body | Push Cube, Stack Cube |
| Deformable | Push Rope, Cloth Move |
| Particle | Push Sand, Pour Water |
| Kinematics | Robot Arm, Reacher |

Please cite the original ACWM-Phys work when using the benchmark, dataset, or
released checkpoints:

```bibtex
@article{xue2026acwm,
  title={ACWM-Phys: Investigating Generalized Physical Interaction in Action-Conditioned Video World Models},
  author={Xue, Haotian and Chen, Yipu and Ma, Liqian and Zhao, Zelin and Moukheiber, Lama and Zhu, Yuchen and Che, Yongxin},
  journal={arXiv preprint arXiv:2605.08567},
  year={2026}
}
```

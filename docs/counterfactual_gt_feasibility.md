# Matched Counterfactual Ground-Truth Feasibility

## Finding

The released ACWM-Phys dataset and code are insufficient to reproduce matched
counterfactual ground truth from the same initial state today. The dataset metadata
does contain an integer `seed`, action tensor, video path, and length. A seed is not a
serialized simulator state, and the public code contains dataset loading, training,
and evaluation only—no MuJoCo XML/assets, environment construction, reset/state-set
API, rendering configuration, or data-generation script.

The [official ACWM-Phys development repository](https://github.com/xavihart/ACWM-Phys-dev)
tree contains model and dataset-consumer code but no simulator generation directory.
The [paper](https://arxiv.org/abs/2605.08567) states that Reacher is a two-link MuJoCo
arm controlled by two joint torques and that simulated OOD shifts are reproducible;
it does not publish the environment assets or a same-state replay interface in the
released repository.

## Questions

### Can the initial state be reproduced?

Not from the current release alone. It may be possible inside the authors' unreleased
generator if `seed` deterministically controls goal, initial joint positions and
velocities, model parameters, and renderer state. That cannot be verified from the
offline metadata. Exact replay would preferably use a saved MuJoCo `qpos`, `qvel`,
goal, model-parameter snapshot, and renderer/camera configuration rather than seed
alone.

### Can arbitrary counterfactual actions be executed?

Conceptually yes in MuJoCo, because Reacher actions are direct torques. Practically no
with the current repository because there is no environment instance to reset and
step. A valid interface must restore the identical pre-action simulator state, apply
the factual or counterfactual torque sequence, and render with identical camera and
timing.

### Required simulator assets

- the exact MuJoCo XML/model and dependent meshes/textures;
- environment and controller/data-generation code;
- initial-state/goal sampling and all randomization ranges;
- reset seed mapping or serialized `qpos`, `qvel`, goal and model parameters;
- action clipping/scaling, control timestep, frame skip, rollout horizon;
- camera, lighting, resolution, renderer version, and any post-processing;
- the dataset commit/version mapping, because current Hub metadata counts/horizons
  differ from the paper table.

### Is the current repository enough?

No. It is enough to measure paired model output sensitivity, but not physical
correctness against matched simulator counterfactuals. Hence all current response
metrics must retain the limitation: they prove that predictions change when actions
change, not that the counterfactual future is correct.

### Next resettable benchmark

The single recommended fallback is a pinned version of the standard
[Gymnasium MuJoCo Reacher environment](https://gymnasium.farama.org/environments/mujoco/reacher/).
It exposes a resettable MuJoCo state and direct two-dimensional torque actions, so
same-state factual/counterfactual rollouts can be generated. It is not pixel-identical
to ACWM-Phys Reacher; it should be treated as a new controlled benchmark, not as ground
truth for existing ACWM-Phys videos.

## Minimal future protocol

Pin MuJoCo/Gymnasium and assets, save the complete initial simulator state, render one
factual rollout plus `0`, `-a`, and scale rollouts from repeated state restoration,
then compare model predictions to those matched videos. This is a separate dataset
construction task and is intentionally not implemented in this validation round.

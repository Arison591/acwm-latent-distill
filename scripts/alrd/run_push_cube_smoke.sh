#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export ACWM_DATA_ROOT="${ACWM_DATA_ROOT:-$PWD/data}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-$PWD/checkpoints/Wan2.1_VAE.pth}"
export WANDB_MODE="${WANDB_MODE:-offline}"

python -m compileall acwm scripts/alrd tests/alrd
python tests/alrd/test_smoke.py
python - <<'PY'
from pathlib import Path
import yaml

for path in sorted(Path("configs/alrd").glob("*.yaml")):
    with path.open() as f:
        yaml.safe_load(f)
    print(f"loaded {path}")
PY

if [[ ! -f "${WAN_VAE_PATH:-Wan2.1_VAE.pth}" ]]; then
  echo "WAN_VAE_PATH is missing; static smoke passed, skipping train.py forward smoke."
  exit 0
fi

if [[ ! -f "${ACWM_DATA_ROOT:-./data}/rigid_dynamics/push_block/ind_train/metadata.pt" ]]; then
  echo "ACWM_DATA_ROOT push_cube data is missing; static smoke passed, skipping train.py forward smoke."
  exit 0
fi

WANDB_MODE=disabled python train.py --config configs/alrd/push_cube_latent.yaml

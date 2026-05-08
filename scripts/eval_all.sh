#!/usr/bin/env bash
# Evaluate all 8 ACWM-Phys environments with the default DiT-S checkpoints.
# Usage: bash scripts/eval_all.sh [--save_videos]

set -e
SAVE_FLAG=${1:-""}

ENVS=(push_cube stack_cube push_rope clothmove push_sand pour_water robot_arm reacher)

for ENV in "${ENVS[@]}"; do
  echo "========================================"
  echo " Evaluating: $ENV"
  echo "========================================"
  python eval.py \
    --env "$ENV" \
    --steps 50 \
    --split both \
    --max_trajs 50 \
    --batch_size 2 \
    --output_root results/ \
    $SAVE_FLAG
done

echo ""
echo "All done. Results written to results/results.md"

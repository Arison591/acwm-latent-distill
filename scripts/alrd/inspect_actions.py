#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from acwm.action_latent.buckets import compute_chunk_magnitude
from acwm.dataset.data_config import get_config_by_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ACWM-Phys action statistics.")
    parser.add_argument("--env", default="push_cube", help="Dataset registry name")
    parser.add_argument("--split", default="ind_train", help="Dataset split directory")
    parser.add_argument("--seq_len", type=int, default=None, help="Override sequence length")
    parser.add_argument("--sampling_rate", type=int, default=None, help="Override sampling rate")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    cfg_kwargs = {}
    if args.seq_len is not None:
        cfg_kwargs["seq_len"] = args.seq_len
    if args.sampling_rate is not None:
        cfg_kwargs["sampling_rate"] = args.sampling_rate
    cfg = get_config_by_name(args.env, **cfg_kwargs)
    split_root = Path(cfg.root_dir) / args.split
    metadata_path = split_root / "metadata.pt"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.pt not found: {metadata_path}. Set ACWM_DATA_ROOT or download the dataset."
        )

    metadata = torch.load(metadata_path, weights_only=False)
    if args.max_episodes is not None:
        metadata = metadata[: args.max_episodes]

    required_span = (cfg.seq_len - 1) * cfg.sampling_rate + 1
    episode_mags = []
    window_mags = []
    lengths = []
    action_dim = None

    for entry in metadata:
        actions = entry["actions"].float()
        action_dim = actions.shape[-1]
        lengths.append(int(entry.get("length", actions.shape[0])))
        episode_mags.append(float(compute_chunk_magnitude(actions).item()))
        max_start = max(0, actions.shape[0] - required_span)
        for start in range(max_start + 1):
            rel = torch.arange(start, start + required_span, cfg.sampling_rate)
            rel = rel[rel < actions.shape[0]]
            if rel.numel() > 0:
                window_mags.append(float(compute_chunk_magnitude(actions[rel]).item()))

    result = {
        "env": args.env,
        "split": args.split,
        "metadata_path": str(metadata_path),
        "episodes": len(metadata),
        "action_dim": action_dim,
        "seq_len": cfg.seq_len,
        "sampling_rate": cfg.sampling_rate,
        "length": _summary(torch.tensor(lengths, dtype=torch.float32)),
        "episode_magnitude": _summary(torch.tensor(episode_mags, dtype=torch.float32)),
        "window_magnitude": _summary(torch.tensor(window_mags, dtype=torch.float32)),
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"env={args.env} split={args.split}")
        print(f"metadata={metadata_path}")
        print(f"episodes={result['episodes']} action_dim={action_dim}")
        for key in ("length", "episode_magnitude", "window_magnitude"):
            print(f"{key}: {result[key]}")


def _summary(values: torch.Tensor) -> dict:
    if values.numel() == 0:
        return {}
    quantiles = torch.quantile(values, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(quantiles[0].item()),
        "q25": float(quantiles[1].item()),
        "median": float(quantiles[2].item()),
        "q75": float(quantiles[3].item()),
        "max": float(quantiles[4].item()),
    }


if __name__ == "__main__":
    main()

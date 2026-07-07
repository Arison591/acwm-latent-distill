#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from acwm.action_latent.buckets import compute_bucket_score, compute_chunk_magnitude
from acwm.dataset.data_config import get_config_by_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Create action-magnitude bucket cache.")
    parser.add_argument("--env", default="push_cube")
    parser.add_argument("--split", default="ind_train")
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--sampling_rate", type=int, default=None)
    parser.add_argument("--quantile", type=float, default=0.5)
    parser.add_argument("--out", default=None, help="Output .pt path")
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
        raise FileNotFoundError(f"metadata.pt not found: {metadata_path}")

    metadata = torch.load(metadata_path, weights_only=False)
    required_span = (cfg.seq_len - 1) * cfg.sampling_rate + 1
    records = []
    magnitudes = []
    signed_scores = []

    for traj_idx, entry in enumerate(metadata):
        actions = entry["actions"].float()
        max_start = max(0, actions.shape[0] - required_span)
        for start_f in range(max_start + 1):
            rel = torch.arange(start_f, start_f + required_span, cfg.sampling_rate)
            rel = rel[rel < actions.shape[0]]
            action_window = actions[rel]
            mag = float(compute_chunk_magnitude(action_window).item()) if rel.numel() else 0.0
            signed_score = float(compute_bucket_score(action_window, score_type="signed_action_0").item()) if rel.numel() else 0.0
            magnitudes.append(mag)
            signed_scores.append(signed_score)
            records.append({"traj_idx": traj_idx, "start_f": start_f, "magnitude": mag})

    mags = torch.tensor(magnitudes, dtype=torch.float32)
    if torch.allclose(mags.min(), mags.max()):
        score_type = "signed_action_0"
        scores = signed_scores
    else:
        score_type = "magnitude"
        scores = magnitudes

    score_tensor = torch.tensor(scores, dtype=torch.float32)
    threshold = float(torch.quantile(score_tensor, args.quantile).item())
    for record, score in zip(records, scores):
        record["score"] = score
        record["score_type"] = score_type
        record["bucket_id"] = 1 if record["score"] > threshold else 0
        record["bucket_name"] = "large" if record["bucket_id"] == 1 else "small"

    out_path = Path(args.out) if args.out else Path("artifacts/action_buckets") / args.env / f"{args.split}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "env": args.env,
            "split": args.split,
            "seq_len": cfg.seq_len,
            "sampling_rate": cfg.sampling_rate,
            "quantile": args.quantile,
            "threshold": threshold,
            "score_type": score_type,
            "records": records,
        },
        out_path,
    )
    print(f"wrote {out_path}")
    print(f"windows={len(records)} threshold={threshold:.6f} score_type={score_type}")


if __name__ == "__main__":
    main()

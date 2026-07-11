#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from acwm.action_latent.action_stats import action_statistics, collect_actions, magnitude_split_diagnostics, motion_effect_diagnostics, split_comparison
from acwm.dataset.data_config import get_config_by_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect action semantics from real ACWM-Phys metadata.")
    parser.add_argument("--env", required=True)
    parser.add_argument("--splits", nargs="+", default=["ind_train", "ind_test", "ood_test"])
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_episodes", type=int)
    parser.add_argument("--scatter_samples", type=int, default=30000)
    parser.add_argument("--train_metadata_parts", action="store_true", help="Use sorted metadata_part_*.pt files for ind_train.")
    parser.add_argument("--analyze_motion", action="store_true", help="Decode videos and compare action magnitude with frame motion.")
    args = parser.parse_args()
    cfg = get_config_by_name(args.env)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries, metadata_by_split, completeness = {}, {}, {}
    for split in args.splits:
        root, metadata_path = Path(cfg.root_dir) / split, Path(cfg.root_dir) / split / "metadata.pt"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"required metadata missing: {metadata_path}")
        part_paths = sorted(root.glob("metadata_part_*.pt")) if split == "ind_train" and args.train_metadata_parts else []
        if part_paths:
            metadata = [entry for part in part_paths for entry in torch.load(part, weights_only=False)]
            metadata_source = [str(path.resolve()) for path in part_paths]
        else:
            metadata = torch.load(metadata_path, weights_only=False, mmap=True)
            metadata_source = [str(metadata_path.resolve())]
        if args.max_episodes is not None:
            metadata = metadata[:args.max_episodes]
        metadata_by_split[split] = metadata
        summaries[split] = action_statistics(metadata)
        paths = [root / entry["video_path"] for entry in metadata]
        present = sum(path.is_file() for path in paths)
        completeness[split] = {"metadata_sources": metadata_source, "metadata_episodes": len(metadata), "referenced_videos_present": present, "referenced_videos_missing": len(paths) - present, "complete": present == len(paths)}
    complete = all(item["complete"] for item in completeness.values())
    summary = {
        "schema_version": 1,
        "environment": args.env,
        "action_semantics": "2D joint torque" if args.env == "reacher" else "not encoded; inspect environment source",
        "splits": summaries,
        "split_comparison": split_comparison(summaries),
        "magnitude_split": magnitude_split_diagnostics(metadata_by_split["ind_train"]),
        "data_completeness": completeness,
        "motion_effect_analysis": _motion_analysis(Path(cfg.root_dir) / "ind_train", metadata_by_split["ind_train"], summary_threshold=magnitude_split_diagnostics(metadata_by_split["ind_train"])["threshold"]) if args.analyze_motion else {"status": "not_evaluated", "reason": "Run with --analyze_motion to decode videos."},
    }
    (args.output_dir / "action_stats.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _plots(args.output_dir, metadata_by_split, args.scatter_samples)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _motion_analysis(root: Path, metadata: list, summary_threshold: float) -> dict:
    action_magnitudes, motion_magnitudes, decode_failures = [], [], []
    for index, entry in enumerate(metadata):
        path = root / entry["video_path"]
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            frames.append(gray.astype("float32") / 255.0)
        cap.release()
        if len(frames) < 2:
            decode_failures.append({"index": index, "video_path": str(path), "decoded_frames": len(frames)})
            continue
        video = torch.from_numpy(np.stack(frames))
        motion_magnitudes.append(float((video[1:] - video[:-1]).abs().mean()))
        action_magnitudes.append(float(torch.as_tensor(entry["actions"]).float().norm(dim=-1).mean()))
    if decode_failures:
        return {"status": "failed_incomplete_decode", "decoded_episodes": len(action_magnitudes), "decode_failures": decode_failures[:20]}
    result = motion_effect_diagnostics(torch.tensor(action_magnitudes), torch.tensor(motion_magnitudes), summary_threshold)
    return {"status": "complete", "motion_proxy": "mean absolute difference between consecutive 64x64 grayscale frames", **result}


def _plots(output_dir: Path, metadata_by_split: dict, scatter_samples: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for split, metadata in metadata_by_split.items():
        episodes, _ = collect_actions(metadata)
        ax.hist(torch.cat(episodes).norm(dim=-1).numpy(), bins=80, density=True, histtype="step", linewidth=1.2, label=split)
    ax.set(xlabel="Per-step action L2 norm", ylabel="Density", title="Action norm distributions")
    ax.legend(); fig.tight_layout(); fig.savefig(output_dir / "norm_histogram.png", dpi=160); plt.close(fig)
    train, _ = collect_actions(metadata_by_split["ind_train"])
    actions = torch.cat(train)
    if actions.shape[1] == 2:
        if actions.shape[0] > scatter_samples:
            actions = actions[torch.randperm(actions.shape[0], generator=torch.Generator().manual_seed(0))[:scatter_samples]]
        fig, ax = plt.subplots(figsize=(5.5, 5.5)); ax.scatter(actions[:, 0], actions[:, 1], s=2, alpha=0.15, rasterized=True)
        ax.axhline(0, color="black", linewidth=0.5); ax.axvline(0, color="black", linewidth=0.5)
        ax.set(xlabel="Torque dimension 0", ylabel="Torque dimension 1", title="Reacher train torque scatter")
        fig.tight_layout(); fig.savefig(output_dir / "torque_scatter.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()

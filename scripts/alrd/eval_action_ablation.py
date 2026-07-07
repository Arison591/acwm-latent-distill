#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from acwm.action_latent.counterfactual import make_counterfactual_actions
from acwm.dataset.dataset import RoboticsDatasetWrapper
from eval import ENV_MAP, compute_metrics, load_checkpoint, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed-input action ablations.")
    parser.add_argument("--env", default="push_cube", choices=list(ENV_MAP.keys()))
    parser.add_argument("--split", default="ind_test", choices=["ind_train", "ind_test", "ood_test"])
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=8)
    parser.add_argument("--modes", nargs="+", default=["zero", "reverse", "scale_0_5"])
    parser.add_argument("--output_root", default="results/alrd_action_ablation")
    parser.add_argument("--save_videos", action="store_true")
    args = parser.parse_args()

    cfg_path, ckpt_path = ENV_MAP[args.env]
    cfg_path = args.cfg or cfg_path
    ckpt_path = args.ckpt or ckpt_path

    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, device)
    step = load_checkpoint(model, ckpt_path, device)
    print(f"loaded checkpoint step={step}")

    dataset_kwargs = dict(config.get("dataset", {}))
    dataset_name = dataset_kwargs.pop("name", args.env)
    for key in ("train_size", "ind_test_size", "ood_test_size"):
        dataset_kwargs.pop(key, None)
    dataset = RoboticsDatasetWrapper.get_dataset(dataset_name, split=args.split, **dataset_kwargs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    out_root = Path(args.output_root) / args.env / args.split
    out_root.mkdir(parents=True, exist_ok=True)
    metrics = {mode: [] for mode in ["original", *args.modes]}
    deltas = {mode: [] for mode in args.modes}

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.max_batches:
            break
        obs = batch["obs"].to(device)
        action = batch["action"].to(device)
        o_0 = obs[:, 0].permute(0, 2, 3, 1).contiguous()
        gt_video = obs.permute(0, 1, 3, 4, 2).contiguous()
        variants = {"original": action}
        variants.update(make_counterfactual_actions(action, args.modes))

        predictions = {}
        with torch.no_grad():
            for name, variant_action in variants.items():
                predictions[name] = model.generate(
                    o_0,
                    variant_action,
                    num_inference_steps=args.steps,
                    noise_level=0.0,
                    mode="parallel",
                )
                metrics[name].append(compute_metrics(predictions[name], gt_video))

        for name in args.modes:
            min_len = min(predictions["original"].shape[1], predictions[name].shape[1])
            delta = (predictions["original"][:, :min_len] - predictions[name][:, :min_len]).pow(2).mean()
            deltas[name].append(float(delta.item()))

        if args.save_videos and batch_idx < 4:
            _save_grid(predictions, out_root / f"sample_{batch_idx}.mp4")

    summary = {
        "env": args.env,
        "split": args.split,
        "checkpoint": ckpt_path,
        "steps": args.steps,
        "metrics": {name: _mean_metric(rows) for name, rows in metrics.items()},
        "action_sensitivity_mse": {
            name: float(np.mean(values)) if values else 0.0 for name, values in deltas.items()
        },
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {summary_path}")


def _mean_metric(rows):
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _save_grid(predictions, path: Path) -> None:
    ordered = list(predictions.keys())
    videos = []
    min_len = min(predictions[name].shape[1] for name in ordered)
    for name in ordered:
        video = predictions[name][0, :min_len].detach().cpu().numpy()
        videos.append(video)
    grid = np.concatenate(videos, axis=2)
    imageio.mimsave(path, (np.clip(grid, 0, 1) * 255).astype(np.uint8), fps=10)


if __name__ == "__main__":
    main()

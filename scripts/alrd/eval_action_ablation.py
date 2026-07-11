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

from acwm.action_latent.counterfactual import counterfactual_scale, counterfactual_semantics, make_counterfactual_actions
from acwm.action_latent.dataset_filter import (
    estimate_action_bucket_threshold,
    filter_dataset_by_action_bucket,
)
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
    parser.add_argument("--modes", nargs="+", default=["zero", "reverse", "scale_0_25", "scale_0_5", "scale_0_75", "scale_1_5", "scale_2"])
    parser.add_argument("--output_root", default="results/alrd_action_ablation")
    parser.add_argument("--save_videos", action="store_true")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for paired initial diffusion noise across action modes.")
    parser.add_argument("--bucket", choices=["small", "large"], default=None,
                        help="Evaluate only one action bucket using a threshold estimated on ind_train.")
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
    bucket_threshold = None
    bucket_score_type = None
    if args.bucket is not None:
        train_kwargs = dict(dataset_kwargs)
        train_dataset = RoboticsDatasetWrapper.get_dataset(
            dataset_name, split="ind_train", **train_kwargs
        )
        bucket_threshold, bucket_score_type = estimate_action_bucket_threshold(
            train_dataset, return_score_type=True
        )
        dataset = filter_dataset_by_action_bucket(
            dataset,
            bucket=args.bucket,
            threshold=bucket_threshold,
            score_type=bucket_score_type,
        )
        print(
            f"filtered action bucket={args.bucket} size={len(dataset)} "
            f"threshold={bucket_threshold:.6f} score_type={bucket_score_type}"
        )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Keep summaries from different checkpoints separate; otherwise running a
    # baseline followed by a student silently overwrites the first result.
    ckpt_file = Path(ckpt_path)
    checkpoint_tag = ckpt_file.parent.name
    if not checkpoint_tag or checkpoint_tag in {"shm", "tmp"}:
        checkpoint_tag = ckpt_file.stem
    out_root = Path(args.output_root) / args.env / args.split / checkpoint_tag / f"seed_{args.seed}"
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
                # Reuse the exact same initial noise for every action variant.
                # Without this pairing, the reported action sensitivity is mostly
                # diffusion sampling variance rather than an action response.
                generator = torch.Generator(device=device).manual_seed(args.seed + batch_idx)
                predictions[name] = model.generate(
                    o_0,
                    variant_action,
                    num_inference_steps=args.steps,
                    noise_level=0.0,
                    mode="parallel",
                    generator=generator,
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
        "seed": args.seed,
        "paired_initial_noise": True,
        "counterfactual_semantics": counterfactual_semantics(args.env, args.modes),
        "bucket": args.bucket,
        "bucket_threshold": bucket_threshold,
        "bucket_score_type": bucket_score_type,
        "metrics": {name: _mean_metric(rows) for name, rows in metrics.items()},
        "action_sensitivity_mse": {
            name: float(np.mean(values)) if values else 0.0 for name, values in deltas.items()
        },
    }
    scale_curve = sorted(
        [(counterfactual_scale(name), summary["action_sensitivity_mse"][name]) for name in args.modes if name.startswith("scale_")]
        + [(1.0, 0.0)]
    )
    summary["action_response_curve"] = [{"alpha": alpha, "response_mse": response} for alpha, response in scale_curve]
    ordered = sorted(scale_curve, key=lambda pair: abs(pair[0] - 1.0))
    comparable = [(left, right) for left, right in zip(ordered, ordered[1:]) if abs(left[0] - 1.0) < abs(right[0] - 1.0)]
    summary["scale_monotonicity"] = {
        "definition": "response should be non-decreasing with |alpha - 1|; equal-distance ties are excluded",
        "satisfied_pairs": sum(left[1] <= right[1] for left, right in comparable),
        "comparable_pairs": len(comparable),
        "fraction": sum(left[1] <= right[1] for left, right in comparable) / len(comparable) if comparable else None,
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

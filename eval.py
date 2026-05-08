"""
Evaluate an ACWM-DiT checkpoint on ACWM-Phys environments.

Usage:
    python eval.py --env push_cube --ckpt checkpoints/push_cube/checkpoint_100000.pt \\
                   --steps 50 --split both --save_videos --output_root results/

Environment variable:
    ACWM_DATA_ROOT  Root directory of the downloaded ACWM-Phys dataset (default: ./data)
"""
import os
import sys
import yaml
import fcntl
import argparse
import numpy as np
import torch
import imageio
from tqdm import tqdm
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from acwm.model.interface import get_dynamics_class
from acwm.dataset.dataset import RoboticsDatasetWrapper
from acwm.dataset.data_config import get_config_by_name

# ── Default config / checkpoint mapping ──────────────────────────────────────
ENV_MAP = {
    "push_cube":  ("configs/envs/push_cube.yaml",  "checkpoints/push_cube/checkpoint_100000.pt"),
    "stack_cube": ("configs/envs/stack_cube.yaml", "checkpoints/stack_cube/checkpoint_100000.pt"),
    "push_rope":  ("configs/envs/push_rope.yaml",  "checkpoints/push_rope/checkpoint_100000.pt"),
    "clothmove":  ("configs/envs/clothmove.yaml",  "checkpoints/clothmove/checkpoint_100000.pt"),
    "push_sand":  ("configs/envs/push_sand.yaml",  "checkpoints/push_sand/checkpoint_100000.pt"),
    "pour_water": ("configs/envs/pour_water.yaml", "checkpoints/pour_water/checkpoint_100000.pt"),
    "robot_arm":  ("configs/envs/robot_arm.yaml",  "checkpoints/robot_arm/checkpoint_100000.pt"),
    "reacher":    ("configs/envs/reacher.yaml",    "checkpoints/reacher/checkpoint_100000.pt"),
}

SCHEDULER_BUFFERS = {"scheduler.sigmas", "scheduler.timesteps", "scheduler.linear_timesteps_weights"}


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(config, device):
    ds_cfg = get_config_by_name(config["dataset"]["name"])
    config["model_config"]["action_dim"] = ds_cfg.action_dim

    model_type = config.get("model_type")
    if model_type:
        model_cfg_path = os.path.join("configs/model", f"{model_type}.yaml")
        with open(model_cfg_path) as f:
            model_spec = yaml.safe_load(f)
        base = config.get("model_config", {})
        base.update(model_spec.get("model_config", {}))
        config["model_config"] = base

    dynamics_class = get_dynamics_class(config["dynamics_class"])
    model = dynamics_class(config["model_name"], config["model_config"])
    model.to(device).eval()
    return model


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in ckpt["model_state_dict"].items()
        if (k[7:] if k.startswith("module.") else k) not in SCHEDULER_BUFFERS
    }
    model.load_state_dict(sd, strict=False)
    return ckpt.get("step", 0)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(pred_video, gt_video):
    min_len = min(pred_video.shape[1], gt_video.shape[1])
    pred = pred_video[:, :min_len]
    gt   = gt_video[:, :min_len]

    mse_per = ((pred - gt) ** 2).mean(dim=(1, 2, 3, 4))
    mse = mse_per.mean().item()

    # Masked-MSE: weight by per-pixel motion (floor=0.01 to avoid ignoring static regions)
    motion_diff = torch.abs(gt - gt[:, :1])
    motion_mask = motion_diff.max(dim=4, keepdim=True)[0].max(dim=1, keepdim=True)[0]
    weight = (0.01 + motion_mask).expand_as(gt)
    masked_mse = ((weight * (pred - gt) ** 2).sum() / (weight.sum() + 1e-8)).item()

    # PSNR
    psnr = 10 * np.log10(1.0 / (mse + 1e-8))

    # SSIM (per frame, averaged)
    pred_np = pred.cpu().numpy()
    gt_np   = gt.cpu().numpy()
    ssim_vals = []
    for b in range(pred_np.shape[0]):
        for t in range(pred_np.shape[1]):
            s = structural_similarity(
                pred_np[b, t], gt_np[b, t],
                data_range=1.0, channel_axis=-1,
            )
            ssim_vals.append(s)
    ssim = float(np.mean(ssim_vals))

    return {"mse": mse, "masked_mse": masked_mse, "psnr": psnr, "ssim": ssim}


# ── Video saving ──────────────────────────────────────────────────────────────

def save_sample_video(pred_video, gt_video, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    min_len = min(pred_video.shape[1], gt_video.shape[1])
    pred = pred_video[0, :min_len].cpu().numpy()
    gt   = gt_video[0,  :min_len].cpu().numpy()
    combined = np.concatenate([gt, pred], axis=2)           # [T, H, 2W, 3]
    video_uint8 = (np.clip(combined, 0, 1) * 255).astype(np.uint8)
    imageio.mimsave(os.path.join(out_dir, "video.mp4"), video_uint8, fps=10)


# ── Results logging ───────────────────────────────────────────────────────────

def append_results_md(results_by_split, env, steps, output_root):
    md_path = os.path.join(output_root, "results.md")
    header = (
        "# Evaluation Results\n\n"
        "| Env          | Split     | Steps | MSE      | Masked-MSE | PSNR   | SSIM   |\n"
        "|:-------------|:----------|------:|:---------|:-----------|:-------|:-------|\n"
    )
    rows = []
    for split, metrics in results_by_split.items():
        rows.append(
            f"| {env:<12} | {split:<9} | {steps:>5} | "
            f"{np.mean(metrics['mse']):.6f} | "
            f"{np.mean(metrics['masked_mse']):.6f}   | "
            f"{np.mean(metrics['psnr']):.2f}  | "
            f"{np.mean(metrics['ssim']):.4f} |"
        )
    with open(md_path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        if os.path.getsize(md_path) == 0:
            f.write(header)
        for row in rows:
            f.write(row + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


# ── Evaluation loop ───────────────────────────────────────────────────────────

def eval_split(model, split_name, dataset_name, args, device, dataset_kwargs=None):
    print(f"\n=== Evaluating split: {split_name} ===")
    kwargs = dict(dataset_kwargs) if dataset_kwargs else {}
    for k in ("name", "test_cuts", "train_size", "ind_test_size", "ood_test_size"):
        kwargs.pop(k, None)

    dataset = RoboticsDatasetWrapper.get_dataset(
        dataset_name, split=split_name,
        max_trajs=args.max_trajs, test_cuts=args.test_cuts, **kwargs,
    )
    print(f"Dataset size: {len(dataset)} windows")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    accum = {"mse": [], "masked_mse": [], "psnr": [], "ssim": []}
    saved_vids = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc=split_name)):
        obs    = batch["obs"].to(device)
        action = batch["action"].to(device)
        o_0      = obs[:, 0].permute(0, 2, 3, 1).contiguous()
        gt_video = obs.permute(0, 1, 3, 4, 2).contiguous()

        with torch.no_grad():
            pred_video = model.generate(
                o_0, action, num_inference_steps=args.steps,
                noise_level=0.0, mode="parallel",
            )

        m = compute_metrics(pred_video, gt_video)
        for k, v in m.items():
            accum[k].append(v)

        if args.save_videos and saved_vids < args.max_saved_vids:
            out_dir = os.path.join(
                args.output_root, args.env,
                f"steps_{args.steps}", split_name, f"sample_{saved_vids}",
            )
            save_sample_video(pred_video, gt_video, out_dir)
            saved_vids += 1

    return accum


def main():
    parser = argparse.ArgumentParser(description="Evaluate ACWM-DiT on ACWM-Phys.")
    parser.add_argument("--env",          type=str, required=True,
                        choices=list(ENV_MAP.keys()), help="Environment name")
    parser.add_argument("--steps",        type=int, default=50,
                        help="Number of denoising steps (default: 50)")
    parser.add_argument("--split",        type=str, default="both",
                        choices=["ind_test", "ood_test", "both"])
    parser.add_argument("--ckpt",         type=str, default=None,
                        help="Path to checkpoint (overrides default from ENV_MAP)")
    parser.add_argument("--cfg",          type=str, default=None,
                        help="Path to config yaml (overrides default from ENV_MAP)")
    parser.add_argument("--max_trajs",    type=int, default=50)
    parser.add_argument("--test_cuts",    type=int, default=10)
    parser.add_argument("--batch_size",   type=int, default=4)
    parser.add_argument("--num_workers",  type=int, default=4)
    parser.add_argument("--output_root",  type=str, default="results")
    parser.add_argument("--save_videos",  action="store_true")
    parser.add_argument("--max_saved_vids", type=int, default=10)
    args = parser.parse_args()

    cfg_path, ckpt_path = ENV_MAP[args.env]
    if args.cfg:
        cfg_path = args.cfg
    if args.ckpt:
        ckpt_path = args.ckpt

    print(f"Config:     {cfg_path}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Steps:      {args.steps}")

    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(config, device)
    step  = load_checkpoint(model, ckpt_path, device)
    print(f"Loaded checkpoint at training step {step}")

    dataset_kwargs = config.get("dataset", {})
    actual_env = dataset_kwargs.get("name", args.env)

    splits = ["ind_test", "ood_test"] if args.split == "both" else [args.split]
    results_by_split = {}
    for split_name in splits:
        results_by_split[split_name] = eval_split(
            model, split_name, actual_env, args, device, dataset_kwargs,
        )

    print("\n=== Final Results ===")
    for split_name, metrics in results_by_split.items():
        print(
            f"  {split_name}: MSE={np.mean(metrics['mse']):.4f} | "
            f"M-MSE={np.mean(metrics['masked_mse']):.4f} | "
            f"PSNR={np.mean(metrics['psnr']):.2f} | "
            f"SSIM={np.mean(metrics['ssim']):.4f}"
        )

    os.makedirs(args.output_root, exist_ok=True)
    append_results_md(results_by_split, args.env, args.steps, args.output_root)
    print(f"\nResults saved to {args.output_root}/results.md")


if __name__ == "__main__":
    main()

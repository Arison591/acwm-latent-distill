#!/usr/bin/env python
"""使用同一初始扩散噪声执行逐窗口动作响应探针。"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from acwm.action_latent.response import ActionGroup, ActionPerturbationSampler, ActionSchema, CounterfactualEvaluator
from acwm.dataset.dataset import RoboticsDatasetWrapper
from eval import compute_metrics, load_checkpoint, load_model

def _predict(model, first, action, steps, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        return model.generate(first, action, num_inference_steps=steps, noise_level=0.0, mode="parallel", generator=generator)

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def _robot_arm_root(data_root: Path | None, cfg_root: str | None) -> Path:
    if data_root is not None:
        return data_root if data_root.name == "robot_arm_64" else data_root / "kinematics" / "robot_arm_64"
    if cfg_root:
        return Path(cfg_root)
    env_root = os.environ.get("ACWM_DATA_ROOT")
    if env_root:
        return Path(env_root) / "kinematics" / "robot_arm_64"
    return Path("data/kinematics/robot_arm_64")

def _assert_split_complete(root: Path, split: str) -> dict:
    metadata_path = root / split / "metadata.pt"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"缺少 {split} 元数据: {metadata_path}")
    data = torch.load(metadata_path, map_location="cpu", weights_only=False)
    missing = [row.get("video_path", "<missing video_path>") for row in data if not (metadata_path.parent / row.get("video_path", "")).is_file()]
    result = {"split": split, "metadata": str(metadata_path), "episodes": len(data), "videos_missing": len(missing), "complete": not missing, "missing_examples": missing[:20]}
    if missing:
        raise FileNotFoundError(json.dumps(result, ensure_ascii=False))
    return result

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True); p.add_argument("--ckpt", required=True); p.add_argument("--schema", type=Path, required=True)
    p.add_argument("--split", default="ind_test", choices=["ind_test", "ood_test"]); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=20); p.add_argument("--max-batches", type=int, default=8); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--random-group-control", action="store_true", help="明确标记该 schema 为预注册随机维度组对照。")
    p.add_argument("--data-root", type=Path, help="ACWM_DATA_ROOT 或 kinematics/robot_arm_64 的直接路径。")
    p.add_argument("--dataset-revision", default="unknown")
    p.add_argument("--finite-difference-epsilon", type=float, default=0.1)
    args = p.parse_args(); schema_json = json.loads(args.schema.read_text())
    schema = ActionSchema(action_dim=schema_json["action_dim"], representation=schema_json["representation"], dimension_names=schema_json.get("dimension_names"), provenance=schema_json["provenance"], groups=[ActionGroup(**g) for g in schema_json["groups"]], group_statistics=schema_json.get("group_statistics", {}))
    if not schema.group_statistics: raise RuntimeError("schema 缺少训练集归一化统计；拒绝以测试集统计采样扰动")
    with open(args.cfg) as f: cfg = yaml.safe_load(f)
    robot_root = _robot_arm_root(args.data_root, cfg.get("dataset", {}).get("root_dir"))
    split_inventory = _assert_split_complete(robot_root, args.split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model = load_model(cfg, device); step = load_checkpoint(model, args.ckpt, device)
    ds_cfg = dict(cfg["dataset"]); name = ds_cfg.pop("name"); [ds_cfg.pop(k, None) for k in ("train_size", "ind_test_size", "ood_test_size")]
    ds_cfg["root_dir"] = str(robot_root)
    loader = DataLoader(RoboticsDatasetWrapper.get_dataset(name, split=args.split, **ds_cfg), batch_size=1, shuffle=False)
    sampler, evaluator, rows = ActionPerturbationSampler(schema), CounterfactualEvaluator(), []
    started = time.time()
    for i, batch in enumerate(loader):
        if i >= args.max_batches: break
        obs, action = batch["obs"].to(device), batch["action"].to(device); first = obs[:, 0].permute(0,2,3,1).contiguous(); gt = obs.permute(0,1,3,4,2).contiguous()
        factual = _predict(model, first, action, args.steps, args.seed+i, device)
        # 相同动作、不同噪声是评估器的噪声地板对照。
        floor = _predict(model, first, action, args.steps, args.seed+i+1_000_000, device)
        # unknown 表示不允许 zero/scale；其余四种无物理语义前提的扰动必须全部报告。
        full, record = sampler.sample(action, schema.groups[0].name, "full_temporal_shuffle", seed=args.seed+i)
        pred = _predict(model, first, full, args.steps, args.seed+i, device)
        rows.append({"window": i, **record, **evaluator.response_summary(factual, pred, noise_floor=floor), "factual_metrics": compute_metrics(factual, gt)})
        for group in schema.groups:
            local_response = None
            for kind in ("group_mask", "within_group_shuffle", "local_additive", "signed_direction"):
                perturbed, record = sampler.sample(action, group.name, kind, seed=args.seed+i)
                response = _predict(model, first, perturbed, args.steps, args.seed+i, device)
                summary = evaluator.response_summary(factual, response, noise_floor=floor)
                if kind == "local_additive": local_response = evaluator.finite_difference_response(factual, response, args.finite_difference_epsilon)
                if kind == "signed_direction" and local_response is not None:
                    signed_response = evaluator.finite_difference_response(factual, response, args.finite_difference_epsilon)
                    summary["finite_difference_epsilon"] = args.finite_difference_epsilon
                    summary["finite_difference_directional_correlation"] = evaluator.directional_correlation(local_response, signed_response)
                rows.append({"window": i, **record, **summary, "factual_metrics": compute_metrics(factual, gt)})
    grouped = {}
    for group in schema.groups:
        values = [r for r in rows if r["group"] == group.name and r["kind"] == "local_additive"]
        grouped[group.name] = {key: float(np.mean([v[key] for v in values])) for key in ("paired_response_mse", "paired_response_l1", "paired_noise_floor_mse", "response_to_noise_floor_ratio")} if values else {}
    shuffle_rows = [row for row in rows if row["kind"] == "full_temporal_shuffle"]
    ckpt_path = Path(args.ckpt)
    summary = {"environment": name, "split": args.split, "seed": args.seed, "paired_initial_noise": True, "noise_floor_is_intentionally_unpaired": True, "checkpoint": str(ckpt_path), "checkpoint_sha256": _sha256(ckpt_path), "checkpoint_step": step, "dataset_root": str(robot_root), "dataset_revision": args.dataset_revision, "split_inventory": split_inventory, "random_group_control": args.random_group_control, "groups": grouped, "action_shuffle_gap": float(np.mean([row["paired_response_mse"] for row in shuffle_rows])) if shuffle_rows else None, "windows": rows}
    summary["run_manifest"] = {"cfg": args.cfg, "schema": str(args.schema), "steps": args.steps, "max_batches": args.max_batches, "device": str(device), "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "torch": torch.__version__, "python": platform.python_version(), "elapsed_sec": time.time() - started}
    args.output.mkdir(parents=True, exist_ok=True); (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k != "windows"}, ensure_ascii=False, indent=2))
if __name__ == "__main__": main()

#!/usr/bin/env python
"""审计 Robot Arm 元数据、视频清单与动作统计；不读取或补零视频。"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from acwm.action_latent.action_stats import action_statistics, split_comparison
from acwm.action_latent.response import ActionSchema
from acwm.dataset.data_config import get_config_by_name

def _inventory(root: Path, metadata: list) -> dict:
    missing = [entry.get("video_path", "<missing video_path>") for entry in metadata if not (root / entry.get("video_path", "")).is_file()]
    metadata_path = root / "metadata.pt"
    return {"metadata_path": str(metadata_path.resolve()), "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(), "episodes": len(metadata), "videos_present": len(metadata)-len(missing), "videos_missing": len(missing), "complete": not missing, "missing_examples": missing[:20]}

def _extended_stats(metadata: list) -> dict:
    stats = action_statistics(metadata)
    actions = torch.cat([torch.as_tensor(row["actions"]).float() for row in metadata])
    diff = torch.cat([torch.diff(torch.as_tensor(row["actions"]).float(), dim=0) for row in metadata if len(row["actions"]) > 1])
    second = torch.cat([torch.diff(torch.as_tensor(row["actions"]).float(), n=2, dim=0) for row in metadata if len(row["actions"]) > 2])
    covariance = torch.cov(actions.T)
    eig = torch.linalg.eigvalsh(covariance).clamp_min(0)
    windows = []
    for row in metadata:
        episode = torch.as_tensor(row["actions"]).float()
        for start in range(max(1, episode.shape[0] - 36)):
            window = episode[start:start + 37]
            windows.append([float(window.norm(dim=-1).mean()), float(torch.diff(window, dim=0).norm(dim=-1).mean()), float((window.abs() > 1e-8).sum(-1).float().mean())])
    complexity = torch.tensor(windows)
    stats.update({"covariance": covariance.tolist(), "correlation": torch.corrcoef(actions.T).tolist(), "action_velocity": {"mean": diff.mean(0).tolist(), "std": diff.std(0, unbiased=False).tolist()}, "action_acceleration": {"mean": second.mean(0).tolist(), "std": second.std(0, unbiased=False).tolist()}, "effective_covariance_rank": int((eig > max(float(eig.max()) * 1e-6, 1e-12)).sum()), "simultaneously_active_dimensions": {"definition": "|a| > 1e-8", "mean": float((actions.abs() > 1e-8).sum(-1).float().mean()), "histogram": torch.bincount((actions.abs() > 1e-8).sum(-1), minlength=actions.shape[-1]+1).tolist()}, "per_window_action_complexity": {"window_length": 37, "mean_l2": {"mean": float(complexity[:,0].mean()), "std": float(complexity[:,0].std(unbiased=False))}, "mean_velocity_l2": {"mean": float(complexity[:,1].mean()), "std": float(complexity[:,1].std(unbiased=False))}, "mean_active_dimensions": {"mean": float(complexity[:,2].mean()), "std": float(complexity[:,2].std(unbiased=False))}}})
    return stats

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, required=True); p.add_argument("--data-root", type=Path); p.add_argument("--dataset-revision", default="unknown"); args = p.parse_args()
    cfg = get_config_by_name("robot_arm")
    base = args.data_root or Path(cfg.root_dir)
    report = {"environment": "robot_arm", "root": str(base), "dataset_revision": args.dataset_revision, "schema_version": 1, "decision": "insufficient_evidence"}
    splits = {}
    for split in ("ind_train", "ind_test", "ood_test"):
        path = base / split / "metadata.pt"
        if not path.is_file(): splits[split] = {"complete": False, "failure": f"缺少元数据: {path}"}; continue
        data = torch.load(path, map_location="cpu", weights_only=False)
        splits[split] = {"inventory": _inventory(path.parent, data), "statistics": _extended_stats(data)}
    report["splits"] = splits
    split_stats = {name: item["statistics"] for name, item in splits.items() if "statistics" in item}
    report["split_distribution_shift"] = split_comparison(split_stats) if "ind_train" in split_stats else {}
    available = [x for x in splits.values() if "statistics" in x]
    if available:
        dim = available[0]["statistics"]["action_dim"]
        schema = ActionSchema.unresolved(dim, provenance="当前工作区未找到可核验的 Robot Arm 官方动作定义；使用稳定维度索引，未臆造关节名称。")
        train = splits.get("ind_train", {}).get("statistics")
        if train: schema.fit(torch.cat([torch.as_tensor(x["actions"]).float() for x in torch.load(base / "ind_train" / "metadata.pt", weights_only=False)]))
        report["action_schema"] = schema.to_dict()
    report["inventory_complete"] = all(x.get("inventory", {}).get("complete", False) for x in splits.values())
    report["eval_split_ready"] = {
        split: bool(splits.get(split, {}).get("inventory", {}).get("complete", False))
        for split in ("ind_test", "ood_test")
    }
    report["modeling_stop"] = not all(report["eval_split_ready"].values())
    if report["modeling_stop"]:
        report["stop_reason"] = "ID/OOD 评估 split 元数据/视频不完整；禁止以缺失样本或补零视频继续模型响应评估。"
    elif not report["inventory_complete"]:
        report["stop_reason"] = "全量 train inventory 仍不完整；继续补齐数据，但允许已完整的 ID/OOD split 执行 Gate A baseline response。"
    else:
        report["stop_reason"] = None
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "action_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    if "action_schema" in report:
        (args.output / "action_schema.json").write_text(json.dumps(report["action_schema"], ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == "__main__": main()

#!/usr/bin/env python
"""汇总同一 checkpoint/split 的三种子逐窗口 ResponseProbe 结果。"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def _write_plots(result: dict, rows: list[dict], output: Path) -> dict:
    plot_status = {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    groups = list(result["groups"])
    metrics = ["paired_response_mse", "paired_response_l1", "response_to_noise_floor_ratio"]
    heat = np.array([[result["groups"][group][metric]["mean"] for metric in metrics] for group in groups], dtype=float)
    fig, ax = plt.subplots(figsize=(max(6, len(metrics) * 1.6), max(3, len(groups) * 0.35)))
    image = ax.imshow(heat, aspect="auto")
    ax.set_xticks(range(len(metrics)), metrics, rotation=30, ha="right")
    ax.set_yticks(range(len(groups)), groups)
    ax.set_title("Per-group response sensitivity")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    heatmap_path = output.with_name(output.stem + "_sensitivity_heatmap.png")
    fig.savefig(heatmap_path, dpi=160)
    plt.close(fig)
    plot_status["sensitivity_heatmap"] = str(heatmap_path)

    values = [row["paired_response_mse"] for row in rows if "paired_response_mse" in row]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=min(40, max(5, int(np.sqrt(len(values))))), color="#4c78a8", alpha=0.85)
    ax.set_title("Response distribution")
    ax.set_xlabel("paired_response_mse")
    ax.set_ylabel("count")
    fig.tight_layout()
    dist_path = output.with_name(output.stem + "_response_distribution.png")
    fig.savefig(dist_path, dpi=160)
    plt.close(fig)
    plot_status["response_distribution"] = str(dist_path)
    plot_status["available"] = True
    return plot_status

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--probe-root", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    summaries = [json.loads(path.read_text()) for path in sorted(args.probe_root.glob("seed_*/summary.json"))]
    if not summaries: raise FileNotFoundError("未找到 seed_*/summary.json")
    group_names = sorted(set.intersection(*(set(item["groups"]) for item in summaries)))
    groups = {name: {metric: {"mean": float(np.mean([item["groups"][name][metric] for item in summaries])), "std": float(np.std([item["groups"][name][metric] for item in summaries], ddof=0))} for metric in ("paired_response_mse", "paired_response_l1", "paired_noise_floor_mse", "response_to_noise_floor_ratio")} for name in group_names}
    rows = [row for item in summaries for row in item["windows"]]
    result = {"seed_count": len(summaries), "seeds": [item["seed"] for item in summaries], "random_group_control": all(item.get("random_group_control", False) for item in summaries), "groups": groups, "action_shuffle_gap": {"mean": float(np.mean([item["action_shuffle_gap"] for item in summaries])), "std": float(np.std([item["action_shuffle_gap"] for item in summaries], ddof=0))}, "response_distribution": {"count": len(rows), "paired_response_mse_quantiles": {str(q): float(np.quantile([row["paired_response_mse"] for row in rows], q)) for q in (0, .05, .5, .95, 1)}}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result["plots"] = _write_plots(result, rows, args.output)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n"); print(json.dumps(result, ensure_ascii=False, indent=2))
if __name__ == "__main__": main()

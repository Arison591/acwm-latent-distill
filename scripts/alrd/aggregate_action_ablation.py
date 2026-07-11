#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-seed paired action-ablation summaries.")
    parser.add_argument("summary", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = [json.loads(path.read_text()) for path in args.summary]
    if len({(row["env"], row["split"], row["checkpoint"], row["steps"]) for row in rows}) != 1:
        raise ValueError("summaries do not describe the same evaluation setting")
    if not all(row.get("paired_initial_noise") for row in rows):
        raise ValueError("all summaries must use paired initial noise")
    result = {
        "environment": rows[0]["env"],
        "split": rows[0]["split"],
        "checkpoint": rows[0]["checkpoint"],
        "steps": rows[0]["steps"],
        "seeds": [row["seed"] for row in rows],
        "num_seeds": len(rows),
        "paired_initial_noise": True,
        "metrics": _nested_mean_std([row["metrics"] for row in rows]),
        "action_sensitivity_mse": _nested_mean_std([row["action_sensitivity_mse"] for row in rows]),
        "scale_monotonicity_fraction": _mean_std([row["scale_monotonicity"]["fraction"] for row in rows]),
        "per_seed_summary_paths": [str(path) for path in args.summary],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _plot_response_curve(rows, args.output.with_name("action_response_curve_3seeds.png"))
    print(json.dumps(result, indent=2, sort_keys=True))


def _nested_mean_std(rows: list[dict]) -> dict:
    result = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        result[key] = _nested_mean_std(values) if isinstance(values[0], dict) else _mean_std(values)
    return result


def _mean_std(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0)), "values": array.tolist()}


def _plot_response_curve(rows: list[dict], path: Path) -> None:
    curves = {row["seed"]: {point["alpha"]: point["response_mse"] for point in row["action_response_curve"]} for row in rows}
    alphas = sorted(set.intersection(*(set(curve) for curve in curves.values())))
    values = np.asarray([[curve[alpha] for alpha in alphas] for curve in curves.values()])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(alphas, values.mean(axis=0), marker="o", label="mean")
    ax.fill_between(alphas, values.mean(axis=0) - values.std(axis=0), values.mean(axis=0) + values.std(axis=0), alpha=0.2, label="±1 std")
    ax.axvline(1.0, color="black", linewidth=0.7, linestyle="--")
    ax.set(xlabel="Torque scale alpha", ylabel="MSE(pred(a), pred(alpha a))", title="Paired-noise action response")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""依据预先定义的规则汇总三种随机种子的 Gate A 结果。"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True, help="各 seed_x/summary.json 的父目录")
    parser.add_argument("--random-probe-root", type=Path, help="随机分组对照的各 seed_x/summary.json 父目录")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-response-noise-ratio", type=float, default=2.0)
    parser.add_argument("--min-stable-groups", type=int, default=1)
    parser.add_argument("--min-random-margin-std", type=float, default=1.0)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    eval_ready = audit.get("eval_split_ready") or {}
    summaries = [json.loads(path.read_text()) for path in sorted(args.probe_root.glob("seed_*/summary.json"))]
    random_summaries = [json.loads(path.read_text()) for path in sorted(args.random_probe_root.glob("seed_*/summary.json"))] if args.random_probe_root else []
    result = {"gate": "A", "predeclared_thresholds": {"min_response_noise_ratio": args.min_response_noise_ratio, "min_stable_groups": args.min_stable_groups, "min_random_margin_std": args.min_random_margin_std}, "audit": str(args.audit), "probe_summaries": [str(p) for p in sorted(args.probe_root.glob("seed_*/summary.json"))], "random_probe_summaries": [str(p) for p in sorted(args.random_probe_root.glob("seed_*/summary.json"))] if args.random_probe_root else []}
    if not all(eval_ready.get(split, False) for split in ("ind_test", "ood_test")):
        result.update(decision="insufficient_evidence", reason="ID/OOD 评估 split 不完整；按协议禁止以缺失样本替代后继续评估。")
    elif len(summaries) < 3:
        result.update(decision="insufficient_evidence", reason=f"需要三个种子，当前只有 {len(summaries)} 个完整探针汇总。")
    elif len(random_summaries) < 3 or not all(summary.get("random_group_control", False) for summary in random_summaries):
        result.update(decision="insufficient_evidence", reason="缺少跨三种子的随机分组方差对照；不能把维度组差异解释为真实各向异性。")
    else:
        # 每个 summary 的 groups: {name: {response_to_noise_floor_ratio: ...}}
        groups = sorted(set.intersection(*(set(s.get("groups", {})) for s in summaries)))
        random_values = [
            value.get("response_to_noise_floor_ratio")
            for summary in random_summaries
            for value in summary.get("groups", {}).values()
            if value.get("response_to_noise_floor_ratio") is not None
        ]
        random_mean = float(np.mean(random_values)) if random_values else None
        random_std = float(np.std(random_values, ddof=0)) if random_values else None
        stable = []
        for group in groups:
            ratios = [s["groups"][group].get("response_to_noise_floor_ratio") for s in summaries]
            mean_ratio = float(np.mean(ratios)) if all(x is not None for x in ratios) else None
            random_margin_ok = (
                mean_ratio is not None
                and random_mean is not None
                and random_std is not None
                and mean_ratio >= random_mean + args.min_random_margin_std * random_std
            )
            if all(x is not None and x >= args.min_response_noise_ratio for x in ratios) and random_margin_ok:
                stable.append({"group": group, "ratios": ratios, "mean_ratio": mean_ratio, "random_mean": random_mean, "random_std": random_std})
        result["random_group_control"] = {"mean_response_to_noise_floor_ratio": random_mean, "std_response_to_noise_floor_ratio": random_std, "count": len(random_values)}
        result["stable_groups"] = stable
        if len(stable) >= args.min_stable_groups:
            result.update(decision="pass", reason="存在跨三种子、超过成对噪声地板且大于随机分组方差的稳定组响应；可进入预注册分组候选阶段。")
        else:
            result.update(decision="fail", reason="未检出稳定且高于噪声地板、并超过随机分组方差的动作响应；必须先诊断条件路径或评估器。")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
if __name__ == "__main__": main()

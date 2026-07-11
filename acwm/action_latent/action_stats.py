from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch


DEFAULT_QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)


def summarize(values: torch.Tensor, quantiles: Sequence[float] = DEFAULT_QUANTILES) -> dict:
    values = values.detach().float().reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {"count": 0}
    levels = torch.tensor(quantiles, dtype=values.dtype, device=values.device)
    q_values = torch.quantile(values, levels)
    return {
        "count": int(values.numel()),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "quantiles": {f"{float(level):g}": float(value) for level, value in zip(levels, q_values)},
    }


def collect_actions(metadata: Iterable[Mapping]) -> tuple[list[torch.Tensor], int]:
    episodes, action_dim = [], None
    for index, entry in enumerate(metadata):
        if "actions" not in entry:
            raise KeyError(f"metadata entry {index} has no 'actions' field")
        actions = torch.as_tensor(entry["actions"]).detach().float()
        if actions.ndim != 2 or actions.shape[0] == 0:
            raise ValueError(f"metadata entry {index} actions must be non-empty [T,D], got {tuple(actions.shape)}")
        if action_dim is None:
            action_dim = int(actions.shape[1])
        elif actions.shape[1] != action_dim:
            raise ValueError(f"metadata entry {index} action dim {actions.shape[1]} != {action_dim}")
        if not torch.isfinite(actions).all():
            raise ValueError(f"metadata entry {index} contains non-finite actions")
        episodes.append(actions)
    return episodes, int(action_dim or 0)


def action_statistics(metadata: Iterable[Mapping], max_lag: int = 5) -> dict:
    episodes, action_dim = collect_actions(metadata)
    if not episodes:
        raise ValueError("metadata is empty")
    flat = torch.cat(episodes)
    result = {
        "episode_count": len(episodes),
        "step_count": int(flat.shape[0]),
        "action_shape": [int(episodes[0].shape[0]), action_dim],
        "action_dim": action_dim,
        "dimension_statistics": [summarize(flat[:, dim]) for dim in range(action_dim)],
        "step_l2_norm": summarize(flat.norm(dim=-1)),
        "chunk_mean_l2_norm": summarize(torch.tensor([episode.norm(dim=-1).mean() for episode in episodes])),
        "temporal_autocorrelation_by_lag": {
            str(lag): [_autocorrelation(episodes, dim, lag) for dim in range(action_dim)]
            for lag in range(1, max_lag + 1)
        },
        "sign_transitions_by_dimension": [_sign_transitions(episodes, dim) for dim in range(action_dim)],
    }
    if action_dim == 2:
        result["quadrant_distribution"] = _quadrants(flat)
    return result


def magnitude_split_diagnostics(metadata: Iterable[Mapping]) -> dict:
    episodes, _ = collect_actions(metadata)
    magnitudes = torch.tensor([episode.norm(dim=-1).mean() for episode in episodes])
    median = float(torch.median(magnitudes))
    low = magnitudes <= median
    low_count, high_count = int(low.sum()), int((~low).sum())
    mean, std = float(magnitudes.mean()), float(magnitudes.std(unbiased=False))
    return {
        "score": "per_episode_mean_action_l2_norm",
        "threshold": median,
        "low_count": low_count,
        "high_count": high_count,
        "low_fraction": low_count / len(episodes),
        "high_fraction": high_count / len(episodes),
        "coefficient_of_variation": std / abs(mean) if mean else None,
        "non_degenerate": bool(std > max(1e-8, abs(mean) * 1e-3) and low_count and high_count),
    }


def split_comparison(split_stats: Mapping[str, Mapping]) -> dict:
    train = split_stats.get("ind_train")
    if train is None:
        return {}
    result = {}
    for split, stats in split_stats.items():
        if split == "ind_train":
            continue
        train_norm, norm = train["step_l2_norm"], stats["step_l2_norm"]
        pooled = max(train_norm["std"], 1e-12)
        result[split] = {
            "step_norm_mean_difference": norm["mean"] - train_norm["mean"],
            "step_norm_standardized_mean_difference": (norm["mean"] - train_norm["mean"]) / pooled,
            "step_norm_std_ratio": norm["std"] / pooled,
        }
    return result


def motion_effect_diagnostics(action_magnitudes: torch.Tensor, motion_magnitudes: torch.Tensor, threshold: float) -> dict:
    action_magnitudes = action_magnitudes.float().reshape(-1)
    motion_magnitudes = motion_magnitudes.float().reshape(-1)
    if action_magnitudes.numel() != motion_magnitudes.numel() or action_magnitudes.numel() < 2:
        raise ValueError("action and motion magnitudes must have the same length >= 2")
    low = action_magnitudes <= threshold
    high = ~low
    if not low.any() or not high.any():
        raise ValueError("threshold must produce non-empty low and high groups")
    x = action_magnitudes - action_magnitudes.mean()
    y = motion_magnitudes - motion_magnitudes.mean()
    denominator = torch.sqrt(x.square().sum() * y.square().sum())
    low_mean = float(motion_magnitudes[low].mean())
    high_mean = float(motion_magnitudes[high].mean())
    return {
        "count": int(action_magnitudes.numel()),
        "low_count": int(low.sum()),
        "high_count": int(high.sum()),
        "low_motion_mean": low_mean,
        "high_motion_mean": high_mean,
        "high_minus_low": high_mean - low_mean,
        "high_to_low_ratio": high_mean / low_mean if low_mean else None,
        "pearson_action_motion": float((x * y).sum() / denominator) if denominator > 0 else None,
    }


def _autocorrelation(episodes: Sequence[torch.Tensor], dim: int, lag: int) -> float | None:
    pairs = [(episode[:-lag, dim], episode[lag:, dim]) for episode in episodes if episode.shape[0] > lag]
    if not pairs:
        return None
    x, y = torch.cat([pair[0] for pair in pairs]), torch.cat([pair[1] for pair in pairs])
    x, y = x - x.mean(), y - y.mean()
    denominator = torch.sqrt(x.square().sum() * y.square().sum())
    return float((x * y).sum() / denominator) if denominator > 0 else None


def _sign_transitions(episodes: Sequence[torch.Tensor], dim: int) -> dict:
    changes = pairs = zeros = 0
    for episode in episodes:
        values = episode[:, dim]
        zeros += int((values == 0).sum())
        left, right = torch.sign(values[:-1]), torch.sign(values[1:])
        valid = (left != 0) & (right != 0)
        pairs += int(valid.sum())
        changes += int(((left != right) & valid).sum())
    return {"eligible_pairs": pairs, "transition_count": changes, "transition_rate": changes / pairs if pairs else None, "zero_count": zeros}


def _quadrants(actions: torch.Tensor) -> dict:
    labels = np.full(actions.shape[0], "axis", dtype=object)
    x, y = actions[:, 0].numpy(), actions[:, 1].numpy()
    labels[(x > 0) & (y > 0)] = "++"
    labels[(x < 0) & (y > 0)] = "-+"
    labels[(x < 0) & (y < 0)] = "--"
    labels[(x > 0) & (y < 0)] = "+-"
    counts, total = Counter(labels.tolist()), actions.shape[0]
    return {label: {"count": counts[label], "fraction": counts[label] / total} for label in ("++", "-+", "--", "+-", "axis")}

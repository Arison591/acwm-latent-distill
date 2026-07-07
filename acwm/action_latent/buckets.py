from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class ActionBucketConfig:
    strategy: str = "magnitude_median"
    threshold: Optional[float] = None
    small_id: int = 0
    large_id: int = 1


def compute_chunk_magnitude(actions: torch.Tensor) -> torch.Tensor:
    """Return mean L2 action magnitude per chunk.

    Accepts [T, A] or [B, T, A]. Returns [] for a single chunk or [B] for a batch.
    """
    if actions.ndim not in (2, 3):
        raise ValueError(f"actions must be [T, A] or [B, T, A], got {tuple(actions.shape)}")
    norms = torch.linalg.vector_norm(actions.float(), ord=2, dim=-1)
    return norms.mean(dim=-1)


def compute_bucket_score(actions: torch.Tensor, score_type: str = "auto") -> torch.Tensor:
    """Return a scalar score for action bucket splits.

    Magnitude is the preferred score. Some ACWM-Phys environments, including
    Push Cube, use unit-length directional actions; in that case magnitude is
    constant, so use the signed first action dimension as a deterministic
    fallback.
    """
    mag = compute_chunk_magnitude(actions)
    if score_type == "magnitude":
        return mag
    if score_type in {"signed_action_0", "signed_first_dim"}:
        return actions.float()[..., 0].mean(dim=-1)
    if score_type != "auto":
        raise ValueError(f"unknown action bucket score_type: {score_type}")

    mag_flat = mag.reshape(-1)
    if mag_flat.numel() <= 1 or not torch.allclose(mag_flat.min(), mag_flat.max()):
        return mag
    return actions.float()[..., 0].mean(dim=-1)


def assign_magnitude_buckets(
    actions: torch.Tensor,
    threshold: Optional[float] = None,
    score_type: str = "auto",
) -> Tuple[torch.Tensor, float]:
    """Assign small/large buckets by mean action magnitude.

    If threshold is omitted, use the median of the provided batch/chunks.
    Returns bucket ids where 0=small and 1=large, plus the threshold used.
    """
    score = compute_bucket_score(actions, score_type=score_type)
    score_flat = score.reshape(-1)
    if score_flat.numel() == 0:
        raise ValueError("cannot bucket empty action tensor")
    if threshold is None:
        threshold_tensor = torch.quantile(score_flat, 0.5)
        threshold_value = float(threshold_tensor.item())
    else:
        threshold_value = float(threshold)
        threshold_tensor = score.new_tensor(threshold_value)
    buckets = (score > threshold_tensor).long()
    return buckets, threshold_value

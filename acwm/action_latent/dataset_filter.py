from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch.utils.data import Subset

from .buckets import compute_bucket_score


BUCKET_NAME_TO_ID = {"small": 0, "large": 1}


def filter_dataset_by_action_bucket(
    dataset,
    bucket: str | int,
    threshold: Optional[float] = None,
    quantile: float = 0.5,
    score_type: str = "auto",
) -> Subset:
    """Return a Subset containing windows from one action-magnitude bucket.

    The function inspects dataset metadata and window indices only. It does not
    call __getitem__ and therefore does not decode videos.
    """
    base_dataset, candidate_positions = _unwrap_dataset(dataset)
    if not hasattr(base_dataset, "indices") or not hasattr(base_dataset, "full_metadata"):
        raise TypeError("dataset must be BaseRoboticsDataset or a Subset of it")

    target = BUCKET_NAME_TO_ID.get(str(bucket).lower(), bucket)
    target = int(target)
    if target not in (0, 1):
        raise ValueError("bucket must be 'small'/0 or 'large'/1")

    if score_type == "auto":
        threshold, score_type = estimate_action_bucket_threshold(
            dataset, quantile=quantile, return_score_type=True
        )
    scores = estimate_action_bucket_scores(dataset, score_type=score_type)
    if not scores:
        raise ValueError("cannot filter an empty dataset")

    score_tensor = torch.tensor(scores, dtype=torch.float32)
    if threshold is None:
        threshold = float(torch.quantile(score_tensor, quantile).item())

    keep_positions = []
    for local_pos, score in enumerate(score_tensor):
        bucket_id = 1 if float(score.item()) > threshold else 0
        if bucket_id == target:
            keep_positions.append(local_pos)

    return Subset(dataset, keep_positions)


def estimate_action_bucket_threshold(dataset, quantile: float = 0.5, return_score_type: bool = False):
    score_type = _infer_dataset_score_type(dataset)
    scores = estimate_action_bucket_scores(dataset, score_type=score_type)
    if not scores:
        raise ValueError("cannot estimate threshold for an empty dataset")
    threshold = float(torch.quantile(torch.tensor(scores, dtype=torch.float32), quantile).item())
    if return_score_type:
        return threshold, score_type
    return threshold


def estimate_action_bucket_scores(dataset, score_type: str = "magnitude") -> list[float]:
    base_dataset, candidate_positions = _unwrap_dataset(dataset)
    if not hasattr(base_dataset, "indices") or not hasattr(base_dataset, "full_metadata"):
        raise TypeError("dataset must be BaseRoboticsDataset or a Subset of it")
    return [_window_bucket_score(base_dataset, base_pos, score_type=score_type) for base_pos in candidate_positions]


def _infer_dataset_score_type(dataset) -> str:
    magnitudes = estimate_action_bucket_scores(dataset, score_type="magnitude")
    if not magnitudes:
        raise ValueError("cannot infer score type for an empty dataset")
    mags = torch.tensor(magnitudes, dtype=torch.float32)
    if torch.allclose(mags.min(), mags.max()):
        return "signed_action_0"
    return "magnitude"


def _unwrap_dataset(dataset) -> tuple[object, Sequence[int]]:
    if isinstance(dataset, Subset):
        base, base_positions = _unwrap_dataset(dataset.dataset)
        return base, [base_positions[i] for i in dataset.indices]
    return dataset, list(range(len(dataset.indices)))


def _window_bucket_score(dataset, base_pos: int, score_type: str) -> float:
    traj_idx, start_f = dataset.indices[base_pos]
    entry = dataset.full_metadata[traj_idx]
    required_span = (dataset.config.seq_len - 1) * dataset.config.sampling_rate + 1
    action_span = dataset._get_action_slice(entry, start_f, start_f + required_span)
    rel_indices = torch.arange(0, required_span, dataset.config.sampling_rate)
    valid_indices = rel_indices[rel_indices < action_span.shape[0]]
    if len(valid_indices) == 0:
        action_window = torch.zeros((dataset.config.seq_len, dataset.config.action_dim))
    else:
        action_window = action_span[valid_indices]
    return float(compute_bucket_score(action_window, score_type=score_type).item())

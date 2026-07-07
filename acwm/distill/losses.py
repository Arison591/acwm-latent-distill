from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F


FeatureFn = Optional[Callable[[torch.Tensor], torch.Tensor]]


def _features(x: torch.Tensor, feature_fn: FeatureFn = None) -> torch.Tensor:
    return feature_fn(x) if feature_fn is not None else x


def prediction_loss(prediction: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    return F.mse_loss(prediction, target, reduction=reduction)


def kd_loss(
    student_prediction: torch.Tensor,
    teacher_prediction: torch.Tensor,
    feature_fn: FeatureFn = None,
    reduction: str = "mean",
) -> torch.Tensor:
    teacher_feat = _features(teacher_prediction.detach(), feature_fn)
    student_feat = _features(student_prediction, feature_fn)
    return F.mse_loss(student_feat, teacher_feat, reduction=reduction)


def response_kd_loss(
    student_prediction: torch.Tensor,
    student_counterfactual_prediction: torch.Tensor,
    teacher_prediction: torch.Tensor,
    teacher_counterfactual_prediction: torch.Tensor,
    feature_fn: FeatureFn = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Match teacher and student response deltas under counterfactual actions."""
    student_delta = _features(student_prediction, feature_fn) - _features(
        student_counterfactual_prediction, feature_fn
    )
    teacher_delta = _features(teacher_prediction.detach(), feature_fn) - _features(
        teacher_counterfactual_prediction.detach(), feature_fn
    )
    return F.mse_loss(student_delta, teacher_delta, reduction=reduction)

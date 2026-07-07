from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
from torch import nn


SCHEDULER_BUFFERS = {
    "scheduler.sigmas",
    "scheduler.timesteps",
    "scheduler.linear_timesteps_weights",
}


def load_checkpoint_state_dict(path: str, map_location: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    cleaned = {}
    for key, value in state.items():
        key = key[7:] if key.startswith("module.") else key
        if key not in SCHEDULER_BUFFERS:
            cleaned[key] = value
    return cleaned


class BucketTeacherEnsemble(nn.Module):
    """Route samples to specialist teachers by bucket id.

    The ensemble expects already-instantiated teacher modules. Keeping checkpoint
    loading separate makes this usable with both ACWM dynamics modules and small
    smoke-test modules.
    """

    def __init__(self, teachers: Mapping[int, nn.Module]) -> None:
        super().__init__()
        if not teachers:
            raise ValueError("at least one teacher is required")
        self.teachers = nn.ModuleDict({str(int(k)): v for k, v in teachers.items()})
        for teacher in self.teachers.values():
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad_(False)

    def forward(self, bucket_ids: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if bucket_ids.ndim != 1:
            bucket_ids = bucket_ids.reshape(-1)
        outputs: Dict[int, Any] = {}
        with torch.no_grad():
            for bucket_key, teacher in self.teachers.items():
                bucket = int(bucket_key)
                mask = bucket_ids == bucket
                if mask.any():
                    outputs[bucket] = teacher(*_select_args(mask, args), **_select_kwargs(mask, kwargs))
        return outputs


def _select_args(mask: torch.Tensor, args: Any) -> Any:
    return tuple(_select_batch(mask, arg) for arg in args)


def _select_kwargs(mask: torch.Tensor, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: _select_batch(mask, value) for key, value in kwargs.items()}


def _select_batch(mask: torch.Tensor, value: Any) -> Any:
    if torch.is_tensor(value) and value.shape[:1] == mask.shape[:1]:
        return value[mask]
    return value

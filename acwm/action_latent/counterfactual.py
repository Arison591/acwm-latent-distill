from __future__ import annotations

from typing import Dict, Iterable

import torch


def make_counterfactual_actions(
    actions: torch.Tensor,
    modes: Iterable[str] = ("zero", "reverse", "scale_0_5"),
) -> Dict[str, torch.Tensor]:
    """Build action counterfactuals used by response distillation/evaluation."""
    variants: Dict[str, torch.Tensor] = {}
    for mode in modes:
        key = mode.lower()
        if key == "zero":
            variants[key] = torch.zeros_like(actions)
        elif key == "reverse":
            variants[key] = -actions
        elif key in {"scale_0_5", "half"}:
            variants["scale_0_5"] = actions * 0.5
        elif key in {"scale_2", "double"}:
            variants["scale_2"] = actions * 2.0
        elif key == "shuffle":
            if actions.ndim < 2:
                raise ValueError("shuffle counterfactual expects an action sequence")
            idx = torch.arange(actions.shape[1] - 1, -1, -1, device=actions.device)
            variants[key] = actions.index_select(1, idx)
        else:
            raise ValueError(f"Unknown counterfactual action mode: {mode}")
    return variants

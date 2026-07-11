from __future__ import annotations

from typing import Dict, Iterable

import torch


COUNTERFACTUAL_SEMANTICS = {
    "push_cube": {
        "zero": "zero target coordinate",
        "reverse": "negated-target action (not established reverse motion)",
    },
    "reacher": {
        "zero": "zero torque",
        "reverse": "opposite joint torques",
        "scale_0_25": "quarter torque",
        "scale_0_5": "half torque",
        "scale_0_75": "three-quarter torque",
        "scale_1_5": "one-and-a-half torque",
        "scale_2": "doubled torque",
    },
}


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
        elif key in {"half", "double"}:
            canonical = "scale_0_5" if key == "half" else "scale_2"
            variants[canonical] = actions * counterfactual_scale(canonical)
        elif key.startswith("scale_"):
            variants[key] = actions * counterfactual_scale(key)
        elif key == "shuffle":
            if actions.ndim < 2:
                raise ValueError("shuffle counterfactual expects an action sequence")
            idx = torch.arange(actions.shape[1] - 1, -1, -1, device=actions.device)
            variants[key] = actions.index_select(1, idx)
        else:
            raise ValueError(f"Unknown counterfactual action mode: {mode}")
    return variants


def counterfactual_scale(mode: str) -> float:
    """Return alpha for a canonical ``scale_<alpha>`` mode.

    Underscores encode decimal points so command-line modes remain shell-friendly:
    ``scale_0_25`` -> 0.25 and ``scale_1_5`` -> 1.5.
    """
    key = mode.lower()
    if not key.startswith("scale_"):
        raise ValueError(f"not a scale counterfactual: {mode}")
    value = key[len("scale_"):].replace("_", ".")
    try:
        alpha = float(value)
    except ValueError as exc:
        raise ValueError(f"invalid scale counterfactual: {mode}") from exc
    if not torch.isfinite(torch.tensor(alpha)):
        raise ValueError(f"invalid scale counterfactual: {mode}")
    return alpha


def counterfactual_semantics(environment: str, modes: Iterable[str]) -> Dict[str, str]:
    known = COUNTERFACTUAL_SEMANTICS.get(environment, {})
    return {mode: known.get(mode, f"action transformed by generic mode {mode}") for mode in modes}

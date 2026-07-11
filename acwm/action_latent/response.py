"""可复用的动作响应诊断原语。

这里的对象刻意不依赖 Robot Arm、视频骨干或扩散实现；调用方必须把成对
（同观测、同初始噪声）的预测传入，避免把采样噪声误报成动作响应。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class ActionGroup:
    name: str
    indices: tuple[int, ...]
    semantics: str = "unresolved_dimension_indices"
    provenance: str = "未验证；仅稳定维度索引"


@dataclass
class ActionSchema:
    """动作表示及其可审计证据。

    ``unknown`` 表示不能将零化、反向或缩放解释成物理反事实；此时只允许
    局部加性、方向扰动、掩码和时序置换。
    """
    action_dim: int
    groups: list[ActionGroup]
    representation: str = "unknown"
    dimension_names: list[str] | None = None
    provenance: str = "动作语义尚未由官方资料验证"
    group_statistics: dict[str, dict[str, list[float]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action_dim <= 0:
            raise ValueError("action_dim 必须为正数")
        if not self.groups:
            raise ValueError("至少需要一个动作组")
        seen: set[int] = set()
        for group in self.groups:
            if not group.indices:
                raise ValueError(f"动作组 {group.name} 为空")
            for index in group.indices:
                if not 0 <= index < self.action_dim:
                    raise ValueError(f"动作组 {group.name} 包含越界维度 {index}")
                if index in seen:
                    raise ValueError(f"动作维度 {index} 被多个组重复使用")
                seen.add(index)
        if self.dimension_names is not None and len(self.dimension_names) != self.action_dim:
            raise ValueError("dimension_names 长度与 action_dim 不一致")

    @classmethod
    def unresolved(cls, action_dim: int, *, provenance: str) -> "ActionSchema":
        return cls(
            action_dim=action_dim,
            groups=[ActionGroup(f"dim_{index}", (index,)) for index in range(action_dim)],
            dimension_names=[f"dim_{index}" for index in range(action_dim)],
            provenance=provenance,
        )

    def fit(self, actions: torch.Tensor) -> "ActionSchema":
        actions = _validate_actions(actions, self.action_dim)
        self.group_statistics = {}
        for group in self.groups:
            values = actions[..., list(group.indices)].reshape(-1, len(group.indices)).float()
            self.group_statistics[group.name] = {
                "mean": values.mean(0).tolist(),
                "std": values.std(0, unbiased=False).clamp_min(1e-8).tolist(),
                "q05": torch.quantile(values, 0.05, dim=0).tolist(),
                "q95": torch.quantile(values, 0.95, dim=0).tolist(),
            }
        return self

    def mask(self, group_name: str, *, device: torch.device | None = None) -> torch.Tensor:
        group = self.group(group_name)
        mask = torch.zeros(self.action_dim, dtype=torch.bool, device=device)
        mask[list(group.indices)] = True
        return mask

    def group(self, name: str) -> ActionGroup:
        for group in self.groups:
            if group.name == name:
                return group
        raise KeyError(f"未知动作组: {name}")

    def perturbation_scale(self, group_name: str, multiplier: float = 0.1) -> torch.Tensor:
        stats = self.group_statistics.get(group_name)
        if stats is None:
            raise RuntimeError("必须先用训练集调用 ActionSchema.fit，再采样局部扰动")
        return torch.as_tensor(stats["std"], dtype=torch.float32) * multiplier

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "groups": [asdict(group) for group in self.groups]}


class ActionPerturbationSampler:
    """基于训练集尺度生成可重放的局部动作扰动。"""
    def __init__(self, schema: ActionSchema, local_scale: float = 0.1) -> None:
        if local_scale <= 0:
            raise ValueError("local_scale 必须为正数")
        self.schema, self.local_scale = schema, local_scale

    def sample(self, actions: torch.Tensor, group_name: str, kind: str, *, seed: int = 0) -> tuple[torch.Tensor, dict[str, Any]]:
        actions = _validate_actions(actions, self.schema.action_dim)
        group = self.schema.group(group_name)
        indices = list(group.indices)
        variant = actions.clone()
        generator = torch.Generator(device=actions.device).manual_seed(seed)
        if kind == "group_mask":
            variant[..., indices] = 0
        elif kind == "within_group_shuffle":
            order = torch.randperm(actions.shape[-2], generator=generator, device=actions.device)
            variant[..., indices] = actions.index_select(-2, order)[..., indices]
        elif kind == "full_temporal_shuffle":
            order = torch.randperm(actions.shape[-2], generator=generator, device=actions.device)
            variant = actions.index_select(-2, order)
        elif kind in {"local_additive", "signed_direction"}:
            scale = self.schema.perturbation_scale(group_name, self.local_scale).to(actions.device)
            if kind == "local_additive":
                delta = torch.randn((*actions.shape[:-1], len(indices)), generator=generator, device=actions.device) * scale
            else:
                direction = torch.ones(len(indices), device=actions.device) if seed % 2 == 0 else -torch.ones(len(indices), device=actions.device)
                delta = direction * scale
            variant[..., indices] += delta
        elif kind == "zero_action":
            if self.schema.representation == "unknown":
                raise ValueError("动作语义未验证，不能把 zero_action 当作物理反事实")
            variant.zero_()
        else:
            raise ValueError(f"不支持的扰动类型: {kind}")
        return variant, {"kind": kind, "group": group_name, "dimensions": indices, "seed": seed, "local_scale": self.local_scale}


class CounterfactualEvaluator:
    """计算预测场的成对响应指标，要求调用方已经保证噪声配对。"""
    @staticmethod
    def paired_response(factual: torch.Tensor, perturbed: torch.Tensor) -> torch.Tensor:
        if factual.shape != perturbed.shape:
            raise ValueError(f"成对预测形状不一致: {tuple(factual.shape)} != {tuple(perturbed.shape)}")
        return perturbed - factual

    @classmethod
    def response_summary(cls, factual: torch.Tensor, perturbed: torch.Tensor, *, noise_floor: torch.Tensor | None = None) -> dict[str, float]:
        response = cls.paired_response(factual, perturbed).float()
        result = {"paired_response_mse": float(response.square().mean()), "paired_response_l1": float(response.abs().mean())}
        if noise_floor is not None:
            floor = cls.paired_response(factual, noise_floor).float().square().mean()
            result["paired_noise_floor_mse"] = float(floor)
            result["response_to_noise_floor_ratio"] = float(response.square().mean() / floor) if floor > 0 else None
        return result

    @classmethod
    def paired_response_mse(
        cls,
        student_factual: torch.Tensor,
        student_perturbed: torch.Tensor,
        teacher_factual: torch.Tensor,
        teacher_perturbed: torch.Tensor,
    ) -> float:
        student_delta = cls.paired_response(student_factual, student_perturbed).float()
        teacher_delta = cls.paired_response(teacher_factual, teacher_perturbed).float()
        if student_delta.shape != teacher_delta.shape:
            raise ValueError(f"student/teacher 响应形状不一致: {tuple(student_delta.shape)} != {tuple(teacher_delta.shape)}")
        return float((student_delta - teacher_delta).square().mean())

    @classmethod
    def finite_difference_response(cls, factual: torch.Tensor, perturbed: torch.Tensor, epsilon: float) -> torch.Tensor:
        if epsilon <= 0:
            raise ValueError("finite-difference epsilon 必须为正数")
        return cls.paired_response(factual, perturbed).float() / epsilon

    @staticmethod
    def directional_correlation(left: torch.Tensor, right: torch.Tensor) -> float | None:
        left, right = left.float().reshape(-1), right.float().reshape(-1)
        left, right = left - left.mean(), right - right.mean()
        denom = torch.sqrt(left.square().sum() * right.square().sum())
        return float((left * right).sum() / denom) if denom > 0 else None


class ResponseProbe:
    """将采样、模型预测与逐窗口结果连接起来的轻量编排器。"""
    def __init__(self, sampler: ActionPerturbationSampler, evaluator: CounterfactualEvaluator | None = None) -> None:
        self.sampler = sampler
        self.evaluator = evaluator or CounterfactualEvaluator()

    def evaluate_pair(self, factual_prediction: torch.Tensor, perturbed_prediction: torch.Tensor, record: Mapping[str, Any]) -> dict[str, Any]:
        return {**dict(record), **self.evaluator.response_summary(factual_prediction, perturbed_prediction)}


def _validate_actions(actions: torch.Tensor, action_dim: int) -> torch.Tensor:
    if actions.ndim < 2 or actions.shape[-1] != action_dim:
        raise ValueError(f"动作必须以 [..., T, {action_dim}] 表示，收到 {tuple(actions.shape)}")
    if not torch.isfinite(actions).all():
        raise ValueError("动作中存在非有限值")
    return actions

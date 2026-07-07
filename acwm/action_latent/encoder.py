from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn


class IdentityActionEncoder(nn.Module):
    """Pass raw action chunks through unchanged."""

    def __init__(self, action_dim: Optional[int] = None, **_: Any) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = action_dim

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return actions


class MLPActionEncoder(nn.Module):
    """Encode each action independently with a small MLP.

    Input and output keep the time axis: [B, T, action_dim] -> [B, T, latent_dim].
    Keeping this contract makes the encoder easy to insert before ACWM-DiT, whose
    action conditioning already expects a per-step sequence.
    """

    def __init__(
        self,
        action_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        act = _activation(activation)
        layers = []
        in_dim = action_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 3:
            raise ValueError(f"actions must be [B, T, A], got shape {tuple(actions.shape)}")
        bsz, steps, dim = actions.shape
        flat = actions.reshape(bsz * steps, dim)
        encoded = self.net(flat)
        return encoded.reshape(bsz, steps, self.latent_dim)


class ConvActionEncoder(nn.Module):
    """Encode action chunks with temporal 1D convolutions.

    The output keeps one latent vector per timestep: [B, T, action_dim] -> [B, T, latent_dim].
    """

    def __init__(
        self,
        action_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        kernel_size: int = 3,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve sequence length")
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        act = _activation(activation)
        padding = kernel_size // 2
        layers = []
        in_dim = action_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Conv1d(in_dim, hidden_dim, kernel_size, padding=padding))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Conv1d(in_dim, latent_dim, kernel_size, padding=padding))
        self.net = nn.Sequential(*layers)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 3:
            raise ValueError(f"actions must be [B, T, A], got shape {tuple(actions.shape)}")
        x = actions.transpose(1, 2).contiguous()
        z = self.net(x)
        return z.transpose(1, 2).contiguous()


def build_action_encoder(config: Dict[str, Any], action_dim: int) -> nn.Module:
    cfg = dict(config or {})
    name = cfg.pop("type", cfg.pop("name", "mlp")).lower()
    cfg.setdefault("action_dim", action_dim)
    if name in {"identity", "raw"}:
        return IdentityActionEncoder(**cfg)
    if name == "mlp":
        return MLPActionEncoder(**cfg)
    if name in {"conv", "conv1d"}:
        return ConvActionEncoder(**cfg)
    raise ValueError(f"Unknown action encoder type: {name}")


def _activation(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    if name == "gelu":
        return nn.GELU
    if name == "silu":
        return nn.SiLU
    raise ValueError(f"Unsupported activation: {name}")

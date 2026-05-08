import os
import torch
import torch.nn as nn
import einops
from acwm.model.dit.dit import DiT, ActionEmbedder


class DiTWrapper(nn.Module):
    """Wrapper for VideoDiT. WanModel support requires the optional wan_dit module."""

    def __init__(self, model_name="VideoDiT", **model_config):
        super().__init__()
        self.model_name = model_name
        self.model_config = model_config

        if model_name == "VideoDiT" or "VideoDiT" in model_name:
            self.model_type = "dit"
            self.model = DiT(**model_config)
        elif model_name == "WanModel" or "WanModel" in model_name:
            self.model_type = "wan"
            try:
                from acwm.model.dit.wan_dit import WanDiffusionWrapper
            except ImportError:
                raise ImportError(
                    "WanModel requires the optional wan_dit module and Wan 2.1 weights. "
                    "See the project README for setup instructions."
                )
            checkpoint_dir = model_config.get("checkpoint_dir", os.environ.get("WAN_MODEL_DIR", "."))
            self.model = WanDiffusionWrapper(checkpoint_dir=checkpoint_dir)

            action_dim = model_config.get("action_dim", 0)
            if action_dim > 0:
                self.action_embedder = ActionEmbedder(
                    action_dim=action_dim,
                    dim=4096,
                    compress_rate=model_config.get("action_compress_rate", 4),
                )
            else:
                self.action_embedder = None
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    @property
    def action_compress_rate(self):
        if hasattr(self.model, "action_compress_rate"):
            return self.model.action_compress_rate
        return self.model_config.get("action_compress_rate", 4)

    def forward(self, x, t, cond):
        if self.model_type == "dit":
            return self.model(x, t, cond)

        elif self.model_type == "wan":
            x_wan = einops.rearrange(x, "b t h w c -> b t c h w")
            if isinstance(cond, torch.Tensor) and cond.ndim == 3 and self.action_embedder is not None:
                prompt_embeds = self.action_embedder(cond)
                cond_dict = {"prompt_embeds": prompt_embeds}
            elif isinstance(cond, torch.Tensor):
                cond_dict = {"prompt_embeds": cond}
            else:
                cond_dict = cond
            out = self.model(x_wan, cond_dict, t)
            return einops.rearrange(out, "b t c h w -> b t h w c")

    def get_cond(self, t, cond):
        if hasattr(self.model, "get_cond"):
            return self.model.get_cond(t, cond)
        return cond

# model name to class mapping
from acwm.model.dit.dit import DiT
from acwm.model.dit.dit_wrapper import DiTWrapper
from acwm.model.tokenizer.wan_tokenizer import WanVAEWrapper, WanVAEPerFrameWrapper

DIT_CLASS_MAP = {
    'VideoDiT': DiT,
    'DiTWrapper': DiTWrapper
}

VAE_CLASS_MAP = {
    'WanVAE': WanVAEWrapper,
    'WanVAEPerFrame': WanVAEPerFrameWrapper,   # WanVAE per-frame: T_latent = T_pixel, no temporal compression
    'FluxVAE': lambda *args, **kwargs: _load_flux_vae(*args, **kwargs),
}

def _load_flux_vae(*args, **kwargs):
    import os
    from acwm.model.tokenizer.flux_vae import FluxVAEWrapper
    # Allow FLUX_VAE_DEBUG=1 to skip real download (useful on nodes without internet)
    if os.environ.get("FLUX_VAE_DEBUG", "0") == "1":
        return FluxVAEWrapper(debug_mode=True)
    try:
        return FluxVAEWrapper(*args, **kwargs)
    except Exception as e:
        print(f"[FluxVAE] Failed to load real weights ({e}), falling back to debug stub.")
        return FluxVAEWrapper(debug_mode=True)

def get_dynamics_class(name):
    if name == 'Bidirectional_FullTrajectory':
        from acwm.dynamics.bi_fulltrajectory import Bidirectional_FullTrajectory
        return Bidirectional_FullTrajectory
    elif name == 'DiffusionForcing_WM':
        from acwm.dynamics.diffusion_forcing_wm import DiffusionForcing_WM
        return DiffusionForcing_WM
    raise ValueError(f"Unknown dynamics class: {name}")

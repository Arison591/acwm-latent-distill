import os
import sys
import torch

# Add project root to sys.path to allow absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from acwm.model.wan_base.modules.vae import _video_vae


class WanVAEWrapper(torch.nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        
        # Mean and std for scaling latents
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

        # WAN_VAE_PATH env var always wins; bare filenames from configs may live
        # under checkpoints/ in disk-constrained local runs.
        pretrained_path = _resolve_pretrained_path(
            os.environ.get("WAN_VAE_PATH") or pretrained_path or "Wan2.1_VAE.pth"
        )

        # init model
        self.model = _video_vae(
            pretrained_path=pretrained_path,
            z_dim=16,
        ).eval().requires_grad_(False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, num_frames, num_channels, height, width]
        # Convert to [batch_size, num_channels, num_frames, height, width]
        x = x.permute(0, 2, 1, 3, 4)
        
        device, dtype = x.device, x.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]
        
        latents = [
            self.model.encode(u.unsqueeze(0), scale).squeeze(0)
            for u in x
        ]
        latents = torch.stack(latents, dim=0)
        
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        latents = latents.permute(0, 2, 1, 3, 4)
        return latents

    def decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        # latent: [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.decode(u.unsqueeze(0),
                              scale).float().clamp_(-1, 1).squeeze(0)
            for u in zs
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output


def _resolve_pretrained_path(pretrained_path: str) -> str:
    if os.path.exists(pretrained_path):
        return pretrained_path
    if os.path.isabs(pretrained_path):
        return pretrained_path

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    candidates = [
        os.path.join("checkpoints", pretrained_path),
        os.path.join(project_root, pretrained_path),
        os.path.join(project_root, "checkpoints", pretrained_path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return pretrained_path

class WanVAEPerFrameWrapper(WanVAEWrapper):
    """
    WanVAE in per-frame mode: encodes each frame independently.
    Result: T_latent = T_pixel  (no temporal compression).
    Uses pretrained WanVAE weights — no new checkpoint needed.

    WanVAE requires T = 1+4k (minimum T=5 for k=1 due to 3D convs).
    We pad each frame to T=5 by replication, encode → T_lat=2, keep only
    the first latent frame (the one conditioned on the target frame).
    """
    _MIN_T = 5   # minimum valid temporal input for WanVAE

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W] in [-1, 1]
        # returns: [B, T, 16, H/8, W/8]
        B, T, C, H, W = x.shape
        device, dtype = x.device, x.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        frames = []
        for t in range(T):
            # Replicate frame to meet WanVAE's minimum T requirement
            f = x[:, t:t+1, :, :, :]                       # [B, 1, C, H, W]
            f_pad = f.expand(-1, self._MIN_T, -1, -1, -1)  # [B, 5, C, H, W]
            f_pad = f_pad.permute(0, 2, 1, 3, 4)           # [B, C, 5, H, W]

            lat = torch.stack([
                self.model.encode(f_pad[b:b+1], scale).squeeze(0)
                for b in range(B)
            ], dim=0)                 # [B, 16, T_lat, H/8, W/8]  T_lat=2 for T=5

            # Take only first latent frame (represents the single input frame)
            frames.append(lat[:, :, :1, :, :].permute(0, 2, 1, 3, 4))  # [B, 1, 16, H/8, W/8]

        return torch.cat(frames, dim=1)   # [B, T, 16, H/8, W/8]

    def decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        # latent: [B, T, 16, H/8, W/8]
        # returns: [B, T, 3, H, W] in [-1, 1]
        B, T = latent.shape[:2]
        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        frames = []
        for t in range(T):
            z1 = latent[:, t:t+1, :, :, :]                   # [B, 1, 16, H/8, W/8]
            # Replicate to T_lat=2 to satisfy decoder's minimum requirement
            z_pad = z1.expand(-1, 2, -1, -1, -1)             # [B, 2, 16, H/8, W/8]
            z_pad = z_pad.permute(0, 2, 1, 3, 4)             # [B, 16, 2, H/8, W/8]

            pix = torch.stack([
                self.model.decode(z_pad[b:b+1], scale).float().clamp_(-1, 1).squeeze(0)
                for b in range(B)
            ], dim=0)                 # [B, 3, T_pix, H, W]  T_pix=5 for T_lat=2

            # Take only first decoded frame
            frames.append(pix[:, :, :1, :, :].permute(0, 2, 1, 3, 4))  # [B, 1, 3, H, W]

        return torch.cat(frames, dim=1)   # [B, T, 3, H, W]


if __name__ == "__main__":
    # Test code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing WanVAEWrapper on {device}...")
    
    ckpt_path = os.environ.get("WAN_VAE_PATH", "Wan2.1_VAE.pth")
    if not os.path.exists(ckpt_path):
        print(f"Warning: VAE checkpoint not found at {ckpt_path}. "
              "Set WAN_VAE_PATH or download from t1an/ACWM-Phys-checkpoints on HuggingFace.")

    try:
        vae = WanVAEWrapper(pretrained_path=ckpt_path).to(device)
        print("Model loaded successfully.")
        
        # Create dummy video (B, T, C, H, W)
        # T should be 1 + 4*k for Wan VAE (e.g., 5, 9, 13...)
        B, T, C, H, W = 1, 5, 3, 128, 128
        dummy_video = torch.randn(B, T, C, H, W).to(device).clamp(-1, 1)
        
        print(f"Input video shape: {dummy_video.shape}")
        
        with torch.no_grad():
            # Test encode
            latent = vae.encode(dummy_video)
            print(f"Latent shape: {latent.shape}")
            
            # Test decode
            recon = vae.decode_to_pixel(latent)
            print(f"Reconstructed video shape: {recon.shape}")
            
            mse = torch.nn.functional.mse_loss(dummy_video, recon)
            print(f"Reconstruction MSE: {mse.item():.6f}")
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

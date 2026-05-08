import torch
import math

class FlowMatchScheduler(torch.nn.Module):
    """
    A simplified Flow Matching scheduler specifically for the Wan template.
    Supports scalars, [B], [B, T], and higher-dimensional timesteps.
    """

    def __init__(self):
        super().__init__()
        self.num_train_timesteps = 1000
        self.register_buffer("sigmas", None, persistent=False)
        self.register_buffer("timesteps", None, persistent=False)
        self.register_buffer("linear_timesteps_weights", None, persistent=False)
        self.training = False # Renamed from self.training as nn.Module has a training attribute

    @property
    def device(self):
        if self.timesteps is not None:
            return self.timesteps.device
        return torch.device('cpu')

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, shift=5.0, training=False):
        """
        Sets the timesteps and sigmas for the Wan template.
        """
        sigma_min = 0.0
        sigma_max = 1.0
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        
        # Sigmas for Wan template: ensure we include 0.0 for clean samples
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps)
        
        # Apply shift (default is 5 for Wan)
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        
        # Move to the current device of the module
        device = self.device
        sigmas = sigmas.to(device)
        timesteps = (sigmas * self.num_train_timesteps).to(device)
        
        self.register_buffer("sigmas", sigmas, persistent=False)
        self.register_buffer("timesteps", timesteps, persistent=False)
        
        if training:
            self.set_training_weight()
            self.training = True
        else:
            self.training = False

    def set_training_weight(self):
        steps = 1000
        x = self.timesteps
        y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
        y_shifted = y - y.min()
        bsmntw_weighing = y_shifted * (steps / y_shifted.sum())
        if len(self.timesteps) != 1000:
            # This is an empirical formula.
            bsmntw_weighing = bsmntw_weighing * (len(self.timesteps) / steps)
            bsmntw_weighing = bsmntw_weighing + bsmntw_weighing[1]
        
        # Move to the current device of the module
        self.register_buffer("linear_timesteps_weights", bsmntw_weighing.to(self.device), persistent=False)

    def _get_timestep_indices(self, timestep: torch.Tensor):
        """
        Efficiently find the nearest indices in self.timesteps for input timesteps.
        Supports any input shape by flattening, computing, and reshapping.
        """
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, device=self.device)
        
        t_input = timestep.to(self.device)
        orig_shape = t_input.shape
        
        # Flatten input to handle any shape (B, T, ...)
        t_flat = t_input.reshape(-1, 1)
        
        # Broadcast against self.timesteps [N] -> [len(t_flat), N]
        diff = (t_flat - self.timesteps.unsqueeze(0)).abs()
        indices = torch.argmin(diff, dim=-1)
        
        return indices.view(orig_shape)

    def step(self, model_output, timestep, sample, to_final=False):
        indices = self._get_timestep_indices(timestep)
        sigma = self.sigmas[indices]
        
        if to_final:
            sigma_next = torch.zeros_like(sigma)
        else:
            # Get next sigma, clamping to avoid out of bounds
            next_indices = (indices + 1).clamp(max=len(self.sigmas) - 1)
            sigma_next = self.sigmas[next_indices]
            # If we were already at the last step, next sigma is 0
            sigma_next = torch.where(indices + 1 >= len(self.sigmas), torch.zeros_like(sigma), sigma_next)

        # Broadcast sigma diff to match sample shape (e.g. [B, T, C, H, W] or [B, C, H, W])
        sigma_diff = (sigma_next - sigma).view(*sigma.shape, *([1] * (sample.ndim - sigma.ndim)))
        sigma_diff = sigma_diff.to(sample.device)
        return sample + model_output * sigma_diff
    
    def return_to_timestep(self, timestep, sample, sample_stablized):
        indices = self._get_timestep_indices(timestep)
        sigma = self.sigmas[indices]
        sigma_view = sigma.view(*sigma.shape, *([1] * (sample.ndim - sigma.ndim)))
        sigma_view = sigma_view.to(sample.device)
        model_output = (sample - sample_stablized) / sigma_view
        return model_output
    
    def add_noise(self, original_samples, noise, timestep):
        indices = self._get_timestep_indices(timestep)
        sigma = self.sigmas[indices]
        
        # Broadcast sigma to match sample shape (e.g. [B, T, 1, 1, 1])
        sigma_view = sigma.view(*sigma.shape, *([1] * (original_samples.ndim - sigma.ndim)))
        sigma_view = sigma_view.to(original_samples.device)
        
        return (1 - sigma_view) * original_samples + sigma_view * noise
    
    def add_independent_noise(self, original_samples, timestep):
        """
        Helper that samples noise independently for each element in original_samples
        and applies it based on the provided timestep (which should match the leading dims).
        """
        noise = torch.randn_like(original_samples)
        return self.add_noise(original_samples, noise, timestep), noise

    def training_target(self, sample, noise, timestep):
        return noise - sample
    
    def training_weight(self, timestep):
        indices = self._get_timestep_indices(timestep)
        return self.linear_timesteps_weights[indices]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Create results directory
    os.makedirs("results/test_flow_matching", exist_ok=True)

    # 1. Initialize scheduler
    scheduler = FlowMatchScheduler()
    num_steps = 50
    scheduler.set_timesteps(num_inference_steps=num_steps, training=True)
    
    # 2. Test with (B, T) shape
    B, T = 2, 4
    indices_bt = torch.randint(0, num_steps, (B, T))
    timesteps_bt = scheduler.timesteps[indices_bt]
    print(f"Testing with (B, T) shape: {timesteps_bt.shape}")
    
    # Test add_noise with (B, T, C, H, W)
    x0 = torch.randn(B, T, 3, 64, 64)
    noise = torch.randn_like(x0)
    xt = scheduler.add_noise(x0, noise, timesteps_bt)
    print(f"xt shape: {xt.shape}")
    assert xt.shape == x0.shape

    # 3. Visualize Timestep Mapping and Training Weights
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Timestep Mapping Curve
    axes[0].plot(range(len(scheduler.timesteps)), scheduler.timesteps.numpy(), marker='.', color='blue', label='Timesteps')
    axes[0].set_title("Timestep Mapping (Wan Shift=5)")
    axes[0].set_xlabel("Inference Step Index")
    axes[0].set_ylabel("Training Timestep (0-1000)")
    axes[0].grid(True)
    axes[0].legend()

    # Right: Training Weights Curve
    axes[1].plot(scheduler.timesteps.numpy(), scheduler.linear_timesteps_weights.numpy(), marker='.', color='red', label='Weights')
    axes[1].set_title("Training Weights vs Timestep")
    axes[1].set_xlabel("Training Timestep")
    axes[1].set_ylabel("Weight Value")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("results/test_flow_matching/scheduler_curves.png")
    print("Saved curves to results/test_flow_matching/scheduler_curves.png")

    # 4. Visualize x_t interpolation (add_noise)
    # Create a simple grid pattern as original image
    size = 256
    grid = np.zeros((size, size, 3), dtype=np.float32)
    grid[::32, :] = 1.0
    grid[:, ::32] = 1.0
    original_image = torch.from_numpy(grid).permute(2, 0, 1).unsqueeze(0) # [1, 3, 256, 256]
    
    # Random noise
    noise = torch.randn_like(original_image)
    
    # Pick a few steps to visualize
    vis_indices = [0, num_steps//4, num_steps//2, 3*num_steps//4, num_steps-1]
    num_vis = len(vis_indices)
    
    fig_xt, axes_xt = plt.subplots(1, num_vis, figsize=(15, 3))
    for i, idx in enumerate(vis_indices):
        t = scheduler.timesteps[idx]
        xt_img = scheduler.add_noise(original_image, noise, t)
        
        # Denormalize for visualization (clip and permute)
        vis_img = xt_img.squeeze(0).permute(1, 2, 0).numpy()
        vis_img = np.clip(vis_img, 0, 1)
        
        axes_xt[i].imshow(vis_img)
        axes_xt[i].set_title(f"t={t:.1f}")
        axes_xt[i].axis('off')
    
    plt.suptitle("Flow Matching Interpolation (x_t) from Data (left) to Noise (right)")
    plt.tight_layout()
    plt.savefig("results/test_flow_matching/xt_interpolation.png")
    print("Saved x_t interpolation to results/test_flow_matching/xt_interpolation.png")

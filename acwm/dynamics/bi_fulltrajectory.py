# action conditioned generation:
#    o[0] + a[0:T-1] -> o[0:T-1]

import torch
from torch import nn
from acwm.model.interface import DIT_CLASS_MAP, VAE_CLASS_MAP

class Bidirectional_FullTrajectory(nn.Module):
    def __init__(self, model_name, model_config):
        super().__init__()
        self.model_name = model_name
        self.model_config = model_config
    
        # Filter DiT-specific keys to avoid passing VAE/Scheduler config to the DiT constructor
        dit_keys = [
            'in_channels', 'patch_size', 'dim', 'num_layers', 'num_heads', 
            'action_dim', 'action_compress_rate', 'max_frames', 
            'rope_config', 'action_dropout_prob', 'temporal_causal',
            'use_flash_attn'
        ]
        dit_config = {k: v for k, v in model_config.items() if k in dit_keys}
        
        self.model = DIT_CLASS_MAP[model_name](**dit_config)
        self.vae = VAE_CLASS_MAP[model_config['vae_name']](*model_config.get('vae_config', []))
        
        # Handle scheduler instantiation
        scheduler_config = model_config.get('scheduler')
        if isinstance(scheduler_config, str):
            if scheduler_config == "FlowMatch":
                from acwm.model.diffusion.flow_matching import FlowMatchScheduler
                self.scheduler = FlowMatchScheduler()
            else:
                raise ValueError(f"Unknown scheduler type: {scheduler_config}")
        else:
            self.scheduler = scheduler_config
        
        # init the scheduler
        self.scheduler.set_timesteps(model_config['training_timesteps'], training=True)
        
    
    def encode_obs(self, o):
        # o can be [B, T, 3, H, W] or [B, T, H, W, 3], values in [0, 1]
        # return: B, T_latent, H', W', D
        
        with torch.no_grad():
            # 1. Normalize [0, 1] -> [-1, 1]
            o = o * 2.0 - 1.0
            
            # 2. Ensure shape is [B, T, 3, H, W] for WanVAEWrapper
            if o.shape[-1] == 3:
                o = o.permute(0, 1, 4, 2, 3).contiguous()
            elif o.shape[2] == 3:
                # Already [B, T, 3, H, W]
                pass
                
            latent = self.vae.encode(o) # [B, T_latent, 16, H/8, W/8]
            # To [B, T_latent, H/8, W/8, 16] for DiT
            latent = latent.permute(0, 1, 3, 4, 2).contiguous()
        return latent
    
    
    def training_loss(self, z, a):
        # z: B, T', H', W', D
        # a: B, T_pixel, C_a
        # return: loss
        
        # Note: the noise is the same for each timestep in batch, only the first frame is clean so the timestep is 0 / a very small number
        
        # sample timestep in batch, same noise for each timstep in batch
        B, T = z.shape[0], z.shape[1]
        t_indices = torch.randint(0, self.scheduler.timesteps.shape[0], (B, ))
        # repeat t for each timestep in batch
        t = t_indices.unsqueeze(1).expand(-1, T).clone() # B, T
        t_values = self.scheduler.timesteps[t].clone() # B, T
        
        # the tricky thing: set t[0] = 0 or a very small noise
        # set to 0 with probability 0.5
        set_to_0 = torch.rand(B, device=z.device) < 0.5
        for b in range(B):
            if set_to_0[b]:
                t_values[b, 0] = 0
            else:
                # randomly sample a last 5% noise level
                # self.scheduler.timesteps[-5:] are the smallest noise levels (near 0)
                small_noise_indices = torch.randint(len(self.scheduler.timesteps) - int(0.1 * len(self.scheduler.timesteps)), len(self.scheduler.timesteps), (1,), device=z.device)
                t_values[b, 0] = self.scheduler.timesteps[small_noise_indices]
            
        eps = torch.randn_like(z, device=z.device) # B, T, H', W', D
        z_t = self.scheduler.add_noise(z, eps, t_values)
        
        v_pred = self.model(z_t, t_values, a)
        v_target = self.scheduler.training_target(z, eps, t_values)
        
        # Apply training weights
        weights = self.scheduler.training_weight(t_values)
        loss = (weights.view(B, T, 1, 1, 1) * (v_pred - v_target)**2).mean()
        return loss
        
    
    def full_train_loss(self, o_t, a):
        # o_t: B, T_pixel, H, W, 3
        # a: B, T_pixel, C_a
        # return: loss
        
        # zero out the last action since it's not used for training
        a = a.clone()
        a[:, -1, :] = 0 
        
        # encode the obs
        z = self.encode_obs(o_t) # B, T', H', W', D
        
        # add noise and get training loss
        loss = self.training_loss(z, a)
        return loss
    
    
    def generate(self, o_0, a, num_inference_steps=50, noise_level=0.0):
        # o_0: B, H, W, 3
        # a: B, T_pixel, A
        # return: B, T_pixel, H, W, 3
        
        B = o_0.shape[0]
        T_pixel = a.shape[1]
        device = o_0.device
        
        # 1. Encode first frame
        # o_0 is [B, H, W, 3], encode_obs expects [B, T, H, W, 3]
        z_0 = self.encode_obs(o_0.unsqueeze(1)) # [B, 1, H', W', 16]
        
        # 2. Determine latent shape
        # Wan VAE temporal compression is 4x
        T_latent = (T_pixel - 1) // 4 + 1
        H_prime, W_prime = z_0.shape[2], z_0.shape[3]
        D = z_0.shape[4] # 16
        
        # 3. Initialize latent sequence with noise
        z = torch.randn(B, T_latent, H_prime, W_prime, D, device=device)
        
        # 4. Handle first frame noise level
        if noise_level > 0:
            # Add noise to z_0 based on the noise_level (t0)
            eps_0 = torch.randn_like(z_0, device=device).squeeze(1)
            z_0_latent = z_0.squeeze(1)
            # We use scheduler.add_noise which works with t_values
            # t_val_0 = torch.tensor([noise_level] * B, device=device)
            # But add_noise expects (B, T, ...)
            t_val_0 = torch.full((B, 1), noise_level, device=device)
            z_0_noisy = self.scheduler.add_noise(z_0, eps_0.unsqueeze(1), t_val_0).squeeze(1)
            z[:, 0] = z_0_noisy
        else:
            # Fix first frame to the encoded observation
            z[:, 0] = z_0.squeeze(1)
        
        # 5. Set inference timesteps
        # Save old training timesteps to restore later
        old_training = self.scheduler.training
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, training=False)
        
        # 6. Denoising loop
        from tqdm import tqdm
        for i in tqdm(range(len(self.scheduler.timesteps)), desc="Denoising"):
            t_val = self.scheduler.timesteps[i]
            # Create (B, T_latent) timesteps
            t = torch.full((B, T_latent), t_val, device=device)
            
            # Handle first frame timestep
            if noise_level > 0:
                # If current t_val > noise_level, we treat first frame as having noise_level
                # If current t_val < noise_level, we denoise it normally
                t[:, 0] = torch.where(t_val > noise_level, torch.tensor(noise_level, device=device), t_val)
            else:
                # Fix first frame timestep to 0 (clean, no update)
                t[:, 0] = 0
            
            with torch.no_grad():
                v_pred = self.model(z, t, a)
                # step updates z_t towards z_0
                z = self.scheduler.step(v_pred, t, z)
                
                # If no noise, re-force first frame to be exactly z_0 to avoid drift
                if noise_level == 0:
                    z[:, 0] = z_0.squeeze(1)
        
        # Restore scheduler state for training if needed
        if old_training:
            self.scheduler.set_timesteps(self.model_config['training_timesteps'], training=True)
            
        # 7. Decode back to pixels
        # vae.decode_to_pixel expects [B, T, 16, H', W']
        with torch.no_grad():
            z_for_vae = z.permute(0, 1, 4, 2, 3).contiguous()
            video_recon = self.vae.decode_to_pixel(z_for_vae) # [B, T_pixel, 3, H, W] in [-1, 1]
            
            # Permute back to [B, T, H, W, 3] and normalize to [0, 1]
            video_recon = (video_recon + 1.0) / 2.0
            video_recon = video_recon.permute(0, 1, 3, 4, 2).contiguous().clamp(0, 1)
        
        return video_recon

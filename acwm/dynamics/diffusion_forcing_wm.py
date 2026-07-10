import torch
from torch import nn
from acwm.model.interface import DIT_CLASS_MAP, VAE_CLASS_MAP
from acwm.action_latent.encoder import build_action_encoder, IdentityActionEncoder

class DiffusionForcing_WM(nn.Module):
    def __init__(self, model_name, model_config):
        super().__init__()
        self.model_name = model_name
        self.model_config = model_config
    
        # Filter DiT-specific keys to avoid passing VAE/Scheduler config to the DiT constructor
        dit_keys = [
            'in_channels', 'patch_size', 'dim', 'num_layers', 'num_heads',
            'action_dim', 'action_compress_rate', 'max_frames',
            'rope_config', 'action_dropout_prob', 'temporal_causal',
            'use_flash_attn', 'action_conditioning',
        ]
        raw_action_dim = model_config.get('action_dim')
        action_encoder_config = model_config.get('action_encoder')
        if action_encoder_config:
            self.action_encoder = build_action_encoder(action_encoder_config, raw_action_dim)
            encoded_action_dim = getattr(self.action_encoder, 'latent_dim', raw_action_dim)
        else:
            self.action_encoder = IdentityActionEncoder(action_dim=raw_action_dim)
            encoded_action_dim = raw_action_dim

        dit_config = {k: v for k, v in model_config.items() if k in dit_keys}
        dit_config['action_dim'] = encoded_action_dim
        
        # Force temporal_causal=True for Diffusion Forcing
        dit_config['temporal_causal'] = True
        
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
        
    def encode_action(self, a):
        return self.action_encoder(a)
    
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
    
    
    def training_outputs(self, z, a, t_values=None, eps=None):
        """Return flow-matching training outputs for distillation-aware training."""
        B, T = z.shape[0], z.shape[1]
        
        if t_values is None:
            # Sample independent timesteps for ALL frames (including first frame)
            t_indices = torch.randint(0, self.scheduler.timesteps.shape[0], (B, T), device=z.device)
            t_values = self.scheduler.timesteps[t_indices] # [B, T]
        
        if eps is None:
            z_t, eps = self.scheduler.add_independent_noise(z, t_values)
        else:
            z_t = self.scheduler.add_noise(z, eps, t_values)
        
        a_model = self.encode_action(a)
        v_pred = self.model(z_t, t_values, a_model)
        v_target = self.scheduler.training_target(z, eps, t_values)
        
        # Apply training weights
        weights = self.scheduler.training_weight(t_values) # [B, T]
        
        # Brainstormed: Motion-aware weighting for training
        # We can weight the loss by the amount of change in the ground truth latents.
        # This helps the model focus on moving objects.
        motion_weight = 1.0
        # Check both model_config and its nested training/model_config if any
        gamma = self.model_config.get('motion_weighting_gamma', 0.0)
        if gamma > 0:
            # Calculate temporal difference in latent space
            # For t=0, diff is 0. For t>0, diff is |z_t - z_{t-1}|
            z_prev = torch.cat([z[:, :1], z[:, :-1]], dim=1)
            diff = torch.abs(z - z_prev).mean(dim=-1, keepdim=True) # [B, T, H, W, 1]
            # Normalize diff locally or use raw? Let's use raw with a gamma.
            motion_weight = 1.0 + gamma * diff
            
        loss_map = (v_pred - v_target)**2
        
        # Brainstormed: Focal Loss for regression
        # Focus on hard-to-denoise parts (large errors)
        focal_alpha = self.model_config.get('focal_alpha', 0.0)
        if focal_alpha > 0:
            # weight = |error|^alpha
            # we use detached error for the weight to maintain stable gradients of MSE
            error = torch.abs(v_pred - v_target)
            focal_weight = (error.detach() + 1e-6) ** focal_alpha
            loss_map = loss_map * focal_weight

        if gamma > 0:
            loss_map = loss_map * motion_weight
            
        loss = (weights.view(B, T, 1, 1, 1) * loss_map).mean()
        return {
            "loss": loss,
            "v_pred": v_pred,
            "v_target": v_target,
            "z_t": z_t,
            "t_values": t_values,
            "eps": eps,
        }

    def training_loss(self, z, a):
        # z: B, T', H', W', D
        # a: B, T_pixel, C_a
        # return: loss
        outputs = self.training_outputs(z, a)
        loss = outputs["loss"]
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
    
    
    def generate(
        self,
        o_0,
        a,
        num_inference_steps=50,
        noise_level=0.0,
        mode="autoregressive",
        generator=None,
    ):
        # o_0: B, H, W, 3
        # a: B, T_pixel, A
        # return: B, T_pixel, H, W, 3
        
        # Diffusion Forcing (Causal) can be run in two modes:
        # 1. "parallel": Denoise the whole sequence at once (fastest, uses causal mask)
        # 2. "autoregressive": Denoise one frame at a time (more stable for long horizons)
        
        B = o_0.shape[0]
        T_pixel = a.shape[1]
        device = o_0.device
        
        # 1. Encode first frame
        z_0 = self.encode_obs(o_0.unsqueeze(1)) # [B, 1, H', W', 16]

        # 2. Determine latent shape
        # temporal_compress_rate=4 for WanVAE, =1 for FluxVAE (frame-independent)
        tcr = self.model_config.get('temporal_compress_rate', 4)
        T_latent = (T_pixel - 1) // tcr + 1 if tcr > 1 else T_pixel
        H_prime, W_prime = z_0.shape[2], z_0.shape[3]
        D = z_0.shape[4] # 16
        
        # Save old scheduler state
        old_training = self.scheduler.training
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, training=False)
        
        from tqdm import tqdm
        
        if mode == "parallel":
            # 3. Initialize latent sequence with noise
            z = torch.randn(
                B, T_latent, H_prime, W_prime, D,
                device=device,
                generator=generator,
            )
            
            # 4. Handle first frame noise level
            if noise_level > 0:
                t_val_0 = torch.full((B, 1), noise_level, device=device)
                z_0_noisy, _ = self.scheduler.add_independent_noise(z_0, t_val_0)
                z[:, 0] = z_0_noisy.squeeze(1)
            else:
                z[:, 0] = z_0.squeeze(1)
            
            # 5. Denoising loop
            for i in tqdm(range(len(self.scheduler.timesteps)), desc="Denoising (Parallel)"):
                t_val = self.scheduler.timesteps[i]
                t = torch.full((B, T_latent), t_val, device=device)
                
                if noise_level > 0:
                    t[:, 0] = torch.where(t_val > noise_level, torch.tensor(noise_level, device=device), t_val)
                else:
                    t[:, 0] = 0
                
                with torch.no_grad():
                    a_model = self.encode_action(a)
                    v_pred = self.model(z, t, a_model)
                    z = self.scheduler.step(v_pred, t, z)
                    
                    if noise_level == 0:
                        z[:, 0] = z_0.squeeze(1)
                        
        elif mode == "autoregressive":
            # 3. Start with only the first frame
            z_all = z_0.clone() # [B, 1, H', W', D]
            
            # 4. Roll the window frame by frame
            for t_idx in range(1, T_latent):
                # a. Add a new noisy frame to the end
                z_next = torch.randn(
                    B, 1, H_prime, W_prime, D,
                    device=device,
                    generator=generator,
                )
                z_curr = torch.cat([z_all, z_next], dim=1) # [B, t+1, ...]
                
                # b. Denoise only the LAST frame in the current sequence
                for i in range(len(self.scheduler.timesteps)):
                    t_val = self.scheduler.timesteps[i]
                    # History is clean (t=0), last frame is noisy (t=t_val)
                    t_seq = torch.zeros(B, t_idx + 1, device=device)
                    t_seq[:, -1] = t_val
                    
                    # Correct actions for current length
                    L_curr = self.model.action_compress_rate * t_idx + 1
                    a_curr = self.encode_action(a[:, :L_curr])
                    
                    with torch.no_grad():
                        v_pred = self.model(z_curr, t_seq, a_curr)
                        z_curr = self.scheduler.step(v_pred, t_seq, z_curr)
                        
                        # Fix history frames (optional but recommended for stability)
                        z_curr[:, :-1] = z_all
                
                # c. Record the clean result for the next iteration
                z_all = torch.cat([z_all, z_curr[:, -1:]], dim=1)
            
            z = z_all
        else:
            raise ValueError(f"Unknown generation mode: {mode}")

        # Restore scheduler state
        if old_training:
            self.scheduler.set_timesteps(self.model_config['training_timesteps'], training=True)
            
        # 6. Decode back to pixels
        with torch.no_grad():
            z_for_vae = z.permute(0, 1, 4, 2, 3).contiguous()
            video_recon = self.vae.decode_to_pixel(z_for_vae)
            video_recon = (video_recon + 1.0) / 2.0
            video_recon = video_recon.permute(0, 1, 3, 4, 2).contiguous().clamp(0, 1)
        
        return video_recon

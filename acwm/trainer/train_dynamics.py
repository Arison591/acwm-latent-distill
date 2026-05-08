import os
import sys
import yaml
import argparse
import wandb
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm
import time
import numpy as np
import signal
import imageio

# Global variables for signal handling
_model = None
_optimizer = None
_step = 0
_epoch = 0
_ckpt_dir = ""
_wandb_run_id = None

def signal_handler(sig, frame):
    """Save checkpoint on SIGTERM (Slurm timeout/preemption)."""
    global _model, _optimizer, _step, _epoch, _ckpt_dir, _wandb_run_id
    if _model is not None and _ckpt_dir:
        rank = 0
        if dist.is_initialized():
            rank = dist.get_rank()
        
        if rank == 0:
            print(f"\n[SIGNAL {sig}] Saving emergency checkpoint at step {_step}...")
            ckpt_path = os.path.join(_ckpt_dir, f"checkpoint_signal_{_step}.pt")
            save_checkpoint(_model, _optimizer, _step, _epoch, ckpt_path, wandb_run_id=_wandb_run_id)
            print(f"--- Emergency Checkpoint Saved: {ckpt_path} ---")
            wandb.finish()
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGTERM, signal_handler)

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from acwm.model.interface import get_dynamics_class
from acwm.dataset.dataset import RoboticsDatasetWrapper
from acwm.utils.visualization import visualize_layout

def setup_ddp():
    if 'RANK' in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        return 0, 0, 1

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def save_checkpoint(model, optimizer, step, epoch, path, wandb_run_id=None, save_numbered=True):
    checkpoint = {
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'epoch': epoch,
        'wandb_run_id': wandb_run_id
    }
    if save_numbered and path:
        torch.save(checkpoint, path)
    
    # Also save a 'latest.pt' for easy resuming
    ckpt_dir = os.path.dirname(path) if path else _ckpt_dir
    latest_path = os.path.join(ckpt_dir, "latest.pt")
    torch.save(checkpoint, latest_path)

def load_checkpoint(model, optimizer, path, device):
    if not os.path.exists(path):
        return 0, 0, None
    
    print(f"--- Loading Checkpoint from {path} ---")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # Handle DDP/FSDP vs single GPU
    state_dict = checkpoint['model_state_dict']
    
    # Filter out scheduler buffers that might cause size mismatches
    scheduler_buffers = [
        'scheduler.sigmas', 
        'scheduler.timesteps', 
        'scheduler.linear_timesteps_weights'
    ]
    for k in scheduler_buffers:
        if k in state_dict:
            del state_dict[k]
        if f"module.{k}" in state_dict:
            del state_dict[f"module.{k}"]
            
    if hasattr(model, 'module'):
        model.module.load_state_dict(state_dict, strict=False)
    else:
        # If loading a DDP state dict into a non-DDP model, strip 'module.'
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['step'], checkpoint['epoch'], checkpoint.get('wandb_run_id')

class Evaluator:
    """
    Evaluates the model on different splits and computes various metrics.
    Extensible to add more scores later.
    """
    def __init__(self, model, device, config, eval_dir):
        self.model = model
        self.device = device
        self.config = config
        self.eval_dir = eval_dir
        os.makedirs(eval_dir, exist_ok=True)
        
    def evaluate(self, loader, split_name, step, num_batches=None, log_videos=True, max_videos=10):
        self.model.eval()
        metrics = {}
        all_mse = []
        all_masked_mse = []
        all_rel_mse = []
        all_psnr = []
        video_logs = []
        
        # Determine generation parameters
        gen_mode = self.config['training'].get('gen_mode', 'parallel')
        inference_steps = self.config['training'].get('inference_steps', 50)
        dataset_name = self.config['dataset']['name']
        
        with torch.no_grad():
            for i, batch in enumerate(loader):
                # Stop if num_batches reached, unless num_batches is None
                if num_batches is not None and i >= num_batches:
                    if not log_videos or len(video_logs) >= max_videos:
                        break
                
                if i % 5 == 0:
                    print(f"  [{split_name}] Batch {i}/{len(loader)}...")
                
                obs = batch['obs'].to(self.device) # [B, T, C, H, W]
                action = batch['action'].to(self.device) # [B, T, A]
                
                # Ground truth for comparison
                gt_video = obs.permute(0, 1, 3, 4, 2).contiguous() # [B, T, H, W, 3]
                
                # Use first frame to initialize rollout
                o_0 = obs[:, 0].permute(0, 2, 3, 1).contiguous()
                
                # Rollout
                if hasattr(self.model, 'module'):
                    curr_model = self.model.module
                else:
                    curr_model = self.model

                try:
                    pred_video = curr_model.generate(o_0, action, mode=gen_mode, num_inference_steps=inference_steps)
                except TypeError:
                    # Fallback for models that don't support mode/steps
                    pred_video = curr_model.generate(o_0, action)
                
                # Handle potential length mismatch due to temporal VAE downsampling (WanVAE factor 4)
                min_len = min(pred_video.shape[1], gt_video.shape[1])
                pred_video = pred_video[:, :min_len]
                gt_video = gt_video[:, :min_len]
                # Also crop action for visualization consistency if needed
                vis_action = action[:, :min_len]
                
                # Metrics (only up to num_batches, or all if None)
                if num_batches is None or i < num_batches:
                    min_len = min(pred_video.shape[1], gt_video.shape[1])
                    pred_video = pred_video[:, :min_len]
                    gt_video = gt_video[:, :min_len]
                    vis_action = action[:, :min_len]
                    mse_per_sample = torch.mean((pred_video - gt_video) ** 2, dim=(1, 2, 3, 4))
                    all_mse.append(mse_per_sample.mean().item())
                    
                    # 1. Rel-MSE
                    first_frame_repeat = gt_video[:, :1].repeat(1, gt_video.shape[1], 1, 1, 1)
                    baseline_mse = torch.mean((first_frame_repeat - gt_video) ** 2, dim=(1, 2, 3, 4))
                    rel_mse = mse_per_sample / (baseline_mse + 1e-8)
                    all_rel_mse.append(rel_mse.mean().item())

                    # 2. Masked-MSE
                    motion_diff = torch.abs(gt_video - gt_video[:, :1])
                    motion_mask = torch.max(motion_diff, dim=4, keepdim=True)[0]
                    motion_mask = torch.max(motion_mask, dim=1, keepdim=True)[0]
                    weight = (0.1 + motion_mask).expand_as(gt_video)
                    masked_mse = (weight * (pred_video - gt_video) ** 2).sum() / (weight.sum() + 1e-8)
                    all_masked_mse.append(masked_mse.item())

                    # 3. PSNR
                    psnr = 10 * torch.log10(1.0 / (mse_per_sample + 1e-10))
                    all_psnr.append(psnr.mean().item())

                # Visualization (up to max_videos)
                if log_videos and len(video_logs) < max_videos:
                    for b in range(obs.shape[0]):
                        if len(video_logs) >= max_videos:
                            break
                        
                        # Prepare for visualize_layout: [T, C, H, W]
                        gt_vis_input = gt_video[b].permute(0, 3, 1, 2).cpu().numpy()
                        pred_vis_input = pred_video[b].permute(0, 3, 1, 2).cpu().numpy()
                        
                        gt_vis = visualize_layout(gt_vis_input, vis_action[b].cpu().numpy(), dataset_name)
                        pred_vis = visualize_layout(pred_vis_input, vis_action[b].cpu().numpy(), dataset_name)
                        
                        combined = np.concatenate([gt_vis, pred_vis], axis=2) # [T, H, 2W, 3]
                        
                        # Save to disk
                        vid_filename = f"{split_name}_step{step}_s{len(video_logs)}.mp4"
                        vid_path = os.path.join(self.eval_dir, vid_filename)
                        try:
                            imageio.mimsave(vid_path, (combined * 255).astype(np.uint8), fps=10)
                        except Exception as e:
                            print(f"Error saving video to {vid_path}: {e}")
                        
                        # Prepare for wandb
                        combined_t = combined.transpose(0, 3, 1, 2)
                        video_logs.append(wandb.Video(combined_t, fps=10, format="mp4", caption=f"{split_name} Step {step} - Sample {len(video_logs)}"))
        
        avg_mse = sum(all_mse) / len(all_mse) if all_mse else 0
        avg_masked_mse = sum(all_masked_mse) / len(all_masked_mse) if all_masked_mse else 0
        avg_rel_mse = sum(all_rel_mse) / len(all_rel_mse) if all_rel_mse else 0
        avg_psnr = sum(all_psnr) / len(all_psnr) if all_psnr else 0

        metrics[f"{split_name}/mse_rollout"] = avg_mse
        metrics[f"{split_name}/mse_masked"] = avg_masked_mse
        metrics[f"{split_name}/mse_rel"] = avg_rel_mse
        metrics[f"{split_name}/psnr"] = avg_psnr
        
        if video_logs:
            metrics[f"{split_name}/videos"] = video_logs
            
        self.model.train()
        return metrics

def main():
    global _model, _optimizer, _step, _epoch, _ckpt_dir, _wandb_run_id
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in wandb dir")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Explicit path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    # Optional: Load model config from a separate file if provided
    model_type = config.get('model_type', None)
    if model_type:
        model_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "model", f"{model_type}.yaml")
        if os.path.exists(model_config_path):
            with open(model_config_path, 'r') as mf:
                model_spec = yaml.safe_load(mf)
                if 'model_config' in model_spec:
                    # Update model_config from the spec file, prioritizing local config overrides if any
                    base_model_config = config.get('model_config', {})
                    base_model_config.update(model_spec['model_config'])
                    config['model_config'] = base_model_config
                if rank == 0:
                    print(f"--- Loaded Model Config: {model_type} from {model_config_path} ---")
        else:
            if rank == 0:
                print(f"WARNING: Model config file {model_config_path} not found.")

    # Automatically set action_dim from dataset registry
    from acwm.dataset.data_config import get_config_by_name
    dataset_name = config['dataset']['name']
    ds_config = get_config_by_name(dataset_name)
    if rank == 0:
        print(f"--- Dataset {dataset_name} action_dim: {ds_config.action_dim} ---")
    config['model_config']['action_dim'] = ds_config.action_dim

    # Setup Checkpoint and Eval Directories
    run_name = config['wandb']['run_name']
    
    # Add resolution to run name if available
    obs_shape = config['dataset'].get('obs_shape')
    if obs_shape:
        # Assuming obs_shape is [C, H, W] or (C, H, W)
        h, w = obs_shape[1], obs_shape[2]
        run_name = f"{run_name}_{h}x{w}"
        
    ckpt_dir = os.path.join("checkpoints", run_name)
    eval_dir = os.path.join("eval_videos", run_name)
    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
    _ckpt_dir = ckpt_dir
    
    # 1. Initialize model
    dynamics_class_name = config['dynamics_class']
    model_name = config['model_name']
    model_config = config['model_config']
    
    if rank == 0:
        print(f"--- Initializing Dynamics Model: {dynamics_class_name} ({model_name}) ---")

    dynamics_class = get_dynamics_class(dynamics_class_name)
    dynamics_model = dynamics_class(model_name, model_config).to(device)
    
    # 2. Optimizer
    optimizer = torch.optim.AdamW(dynamics_model.parameters(), lr=float(config['training']['learning_rate']))
    _optimizer = optimizer

    # 3. Resume Logic
    start_step = 0
    start_epoch = 0
    wandb_run_id = None
    
    resume_path = args.ckpt_path
    if args.resume and resume_path is None:
        potential_latest = os.path.join(ckpt_dir, "latest.pt")
        if os.path.exists(potential_latest):
            resume_path = potential_latest

    if resume_path:
        start_step, start_epoch, wandb_run_id = load_checkpoint(dynamics_model, optimizer, resume_path, device)
        _wandb_run_id = wandb_run_id

    # Distributed wrapper
    if config['distributed']['use_fsdp']:
        model = FSDP(dynamics_model)
    elif world_size > 1:
        model = DDP(dynamics_model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        model = dynamics_model
    _model = model

    if rank == 0:
        params = sum(p.numel() for p in dynamics_model.model.parameters() if p.requires_grad)
        print(f"Model Parameters: {params / 1e6:.2f}M")
        print(f"--- Distributed Setup Finished ---")

    # 4. Initialize WandB
    if rank == 0:
        if config['wandb'].get('api_key') and config['wandb']['api_key'] != "YOUR_WANDB_API_KEY_HERE":
            os.environ["WANDB_API_KEY"] = config['wandb']['api_key']
        
        wandb.init(
            project=config['wandb']['project'],
            name=run_name,
            config=config,
            id=wandb_run_id,
            resume="allow"
        )
        wandb_run_id = wandb.run.id
        _wandb_run_id = wandb_run_id
        
    # Dataset and Dataloader
    dataset_name = config['dataset']['name']
    if rank == 0:
        print(f"--- Loading Dataset: {dataset_name} ---")
    
    train_seq_len = config['dataset'].get('train_seq_len', config['dataset'].get('seq_len'))
    eval_seq_len = config['dataset'].get('eval_seq_len', config['dataset'].get('seq_len'))
    
    dataset_kwargs = config['dataset'].copy()
    dataset_kwargs.pop('name', None)
    
    train_kwargs = dataset_kwargs.copy()
    train_kwargs['seq_len'] = train_seq_len
    
    val_kwargs = dataset_kwargs.copy()
    val_kwargs['seq_len'] = eval_seq_len
    # Pass test_cuts to validation/test datasets
    val_kwargs['test_cuts'] = config['dataset'].get('test_cuts')

    # Determine if we have explicit splits
    from acwm.dataset.data_config import DATASET_ROOT
    has_explicit_splits = False
    # Check if we can find ind_train directory in the dataset root
    from acwm.dataset.data_config import DATASET_REGISTRY
    ds_base_config = DATASET_REGISTRY.get(dataset_name)
    if ds_base_config:
        root = ds_base_config['root_dir']
        if os.path.exists(os.path.join(root, 'ind_train')):
            has_explicit_splits = True

    if has_explicit_splits:
        if rank == 0:
            print(f"Detected explicit splits for {dataset_name}")
        train_dataset = RoboticsDatasetWrapper.get_dataset(dataset_name, split='ind_train', 
                                                           max_trajs=config['dataset'].get('train_size'), **train_kwargs)
        ind_test_dataset = RoboticsDatasetWrapper.get_dataset(dataset_name, split='ind_test', 
                                                              max_trajs=config['dataset'].get('ind_test_size'), **val_kwargs)
        ood_test_dataset = RoboticsDatasetWrapper.get_dataset(dataset_name, split='ood_test', 
                                                              max_trajs=config['dataset'].get('ood_test_size'), **val_kwargs)
        
        # Use ind_test for frequent validation
        val_dataset = ind_test_dataset
        
        if rank == 0:
            print(f"Splits: train={len(train_dataset)}, ind_test={len(ind_test_dataset)}, ood_test={len(ood_test_dataset)}")
    else:
        # Fallback to legacy behavior: split the root dataset
        dataset_full = RoboticsDatasetWrapper.get_dataset(dataset_name, **train_kwargs)
        val_dataset_full = RoboticsDatasetWrapper.get_dataset(dataset_name, **val_kwargs)
        
        unique_traj_ids = sorted(list(set([idx[0] for idx in dataset_full.indices])))
        num_total_trajs = len(unique_traj_ids)
        
        # Determine split sizes
        train_size = config['dataset'].get('train_size')
        ind_test_size = config['dataset'].get('ind_test_size')
        
        import random
        random.seed(42)
        random.shuffle(unique_traj_ids)
        
        if train_size and ind_test_size:
            # Use exact sizes if provided
            train_traj_ids = set(unique_traj_ids[:train_size])
            val_traj_ids = set(unique_traj_ids[train_size:train_size+ind_test_size])
        else:
            # Fallback to ratio
            split_ratio = config['dataset'].get('train_test_split', 10)
            num_val_trajs = max(1, num_total_trajs // (split_ratio + 1))
            val_traj_ids = set(unique_traj_ids[:num_val_trajs])
            train_traj_ids = set(unique_traj_ids[num_val_trajs:])
        
        train_indices = [i for i, (t_idx, _) in enumerate(dataset_full.indices) if t_idx in train_traj_ids]
        val_indices = [i for i, (t_idx, _) in enumerate(val_dataset_full.indices) if t_idx in val_traj_ids]
        
        train_dataset = Subset(dataset_full, train_indices)
        val_dataset = Subset(val_dataset_full, val_indices)
        ind_test_dataset = val_dataset
        ood_test_dataset = None
        
        if rank == 0:
            print(f"Legacy Split: train_windows={len(train_indices)}, val_windows={len(val_indices)}")
            if train_size: print(f"Target train_trajs: {train_size}, test_trajs: {ind_test_size}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        sampler=train_sampler, 
        shuffle=(train_sampler is None),
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # Validation loaders (no DistributedSampler needed for eval if done only on rank 0)
    def get_eval_loader(ds):
        if ds is None: return None
        return DataLoader(
            ds, 
            batch_size=config['training'].get('val_batch_size', config['training']['batch_size']),
            shuffle=False,
            num_workers=config['training']['num_workers']
        )

    ind_test_loader = get_eval_loader(ind_test_dataset)
    ood_test_loader = get_eval_loader(ood_test_dataset)
    
    # Initialize Evaluator
    evaluator = Evaluator(model, device, config, eval_dir)

    # Training Loop
    num_epochs = config['training'].get('num_epochs', 200)
    total_steps = config['training'].get('total_steps', None)
    step = start_step
    _step = step
    _epoch = start_epoch
    
    if rank == 0:
        if total_steps:
            print(f"--- Starting Training Loop: {total_steps} Total Steps (Max {num_epochs} Epochs) ---")
        else:
            print(f"--- Starting Training Loop: {num_epochs} Epochs ---")

    for epoch in range(start_epoch, num_epochs):
        _epoch = epoch
        if total_steps and step >= total_steps:
            break
            
        if train_sampler:
            train_sampler.set_epoch(epoch)
            
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=(rank != 0))
        
        last_step_end = time.time()
        for batch in pbar:
            if total_steps and step >= total_steps:
                break
                
            iter_start_time = time.time()
            data_time = iter_start_time - last_step_end
            obs = batch['obs'].to(device)
            action = batch['action'].to(device)
            
            optimizer.zero_grad()
            
            # Encode
            with torch.no_grad():
                z = model.module.encode_obs(obs) if hasattr(model, 'module') else model.encode_obs(obs)
            
            # Forward & Backward
            loss = model.module.training_loss(z, action) if hasattr(model, 'module') else model.training_loss(z, action)
            loss.backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            optimizer.step()
            
            train_step_time = time.time() - iter_start_time
            step += 1
            _step = step
            
            if rank == 0:
                step_time_total = time.time() - last_step_end
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "st": f"{step_time_total:.2f}s"})
                
                if step % config['training']['log_freq'] == 0:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/grad_norm": grad_norm,
                        "train/epoch": epoch,
                        "time/data_loading": data_time,
                        "time/training_step": train_step_time,
                        "time/seconds_per_step": step_time_total,
                    }, step=step)
                
                # Periodic Evaluation (Trigger at val_freq or step 10 for quick verification)
                val_freq = config['training'].get('val_freq', 1000)
                if step % val_freq == 0 or step == 10:
                    print(f"\n--- Running Evaluation at Step {step} ---")
                    eval_start = time.time()
                    
                    # Evaluate In-Distribution
                    ind_metrics = evaluator.evaluate(ind_test_loader, "ind_test", step, num_batches=10, log_videos=True)
                    wandb.log(ind_metrics, step=step)
                    
                    # Evaluate Out-of-Distribution
                    if ood_test_loader:
                        ood_metrics = evaluator.evaluate(ood_test_loader, "ood_test", step, num_batches=10, log_videos=True)
                        wandb.log(ood_metrics, step=step)
                    
                    eval_time = time.time() - eval_start
                    wandb.log({"time/evaluation": eval_time}, step=step)
                    print(f"--- Evaluation Finished (Took {eval_time:.2f}s) ---")
                
                # Checkpoints
                ckpt_freq = config['training'].get('checkpoint_freq', 2000)
                if step % ckpt_freq == 0:
                    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{step}.pt")
                    save_checkpoint(model, optimizer, step, epoch, ckpt_path, wandb_run_id=wandb_run_id)
                elif step % 500 == 0:
                    save_checkpoint(model, optimizer, step, epoch, None, wandb_run_id=wandb_run_id, save_numbered=False)
            
            # Critical: Sync all processes before starting next step to disentangle eval time
            if world_size > 1:
                dist.barrier()
            last_step_end = time.time()
                    
        if world_size > 1:
            dist.barrier()
        
    if rank == 0:
        wandb.finish()
    cleanup_ddp()

if __name__ == "__main__":
    main()

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
import copy

# Global variables for signal handling
_model = None
_optimizer = None
_step = 0
_epoch = 0
_ckpt_dir = ""
_wandb_run_id = None
_checkpoint_enabled = True

def signal_handler(sig, frame):
    """Save checkpoint on SIGTERM (Slurm timeout/preemption)."""
    global _model, _optimizer, _step, _epoch, _ckpt_dir, _wandb_run_id
    if _checkpoint_enabled and _model is not None and _ckpt_dir:
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
from acwm.action_latent.buckets import assign_magnitude_buckets
from acwm.action_latent.counterfactual import make_counterfactual_actions
from acwm.action_latent.dataset_filter import (
    estimate_action_bucket_threshold,
    filter_dataset_by_action_bucket,
)
from acwm.distill.losses import kd_loss, response_kd_loss
from acwm.distill.teacher import load_checkpoint_state_dict

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
    if save_numbered and path:
        try:
            if os.path.exists(latest_path):
                os.remove(latest_path)
            os.link(path, latest_path)
            return
        except OSError:
            pass
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

def build_distillation_teachers(config, dynamics_class, model_name, model_config, device, rank=0):
    distill_cfg = config.get('distillation', {})
    if not distill_cfg.get('enabled', False):
        return {}

    ckpt_map = distill_cfg.get('teacher_checkpoints', {})
    if not ckpt_map:
        raise ValueError("distillation.enabled=true requires distillation.teacher_checkpoints")

    teachers = {}
    for bucket_name, ckpt_path in ckpt_map.items():
        if ckpt_path is None:
            continue
        bucket_id = _bucket_name_to_id(bucket_name)
        teacher = dynamics_class(model_name, copy.deepcopy(model_config)).to(device)
        state_dict = load_checkpoint_state_dict(ckpt_path, map_location=device)
        missing, unexpected = teacher.load_state_dict(state_dict, strict=False)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        teachers[bucket_id] = teacher
        if rank == 0:
            print(
                f"--- Loaded teacher bucket={bucket_name}({bucket_id}) from {ckpt_path} "
                f"missing={len(missing)} unexpected={len(unexpected)} ---"
            )

    if not teachers:
        raise ValueError("distillation.enabled=true but no valid teacher checkpoints were provided")
    return teachers

def compute_distillation_loss(core_model, teachers, distill_cfg, action_bucket_cfg, z, action, student_outputs):
    if not teachers:
        return z.new_tensor(0.0), {}

    lambda_kd = float(distill_cfg.get('lambda_kd', distill_cfg.get('loss_weights', {}).get('lambda_kd', 0.25)))
    mu_resp = float(distill_cfg.get('mu_resp', distill_cfg.get('loss_weights', {}).get('mu_resp', 0.5)))
    bucket_threshold = distill_cfg.get('bucket_threshold', action_bucket_cfg.get('threshold'))
    counterfactual_modes = distill_cfg.get('counterfactuals', ['zero', 'reverse', 'scale_0_5'])
    bucket_ids, used_threshold = assign_magnitude_buckets(
        action.detach(),
        threshold=bucket_threshold,
        score_type=action_bucket_cfg.get('score_type', 'auto'),
    )

    total = z.new_tensor(0.0)
    kd_total = z.new_tensor(0.0)
    resp_total = z.new_tensor(0.0)
    used_buckets = 0

    for bucket_id, teacher in teachers.items():
        mask = bucket_ids == int(bucket_id)
        if not mask.any():
            continue
        used_buckets += 1
        with torch.no_grad():
            teacher_outputs = teacher.training_outputs(
                z[mask].detach(),
                action[mask],
                t_values=student_outputs['t_values'][mask],
                eps=student_outputs['eps'][mask],
            )
        kd = kd_loss(student_outputs['v_pred'][mask], teacher_outputs['v_pred'])
        kd_total = kd_total + kd

        if mu_resp > 0 and counterfactual_modes:
            variants = make_counterfactual_actions(action, counterfactual_modes)
            for _, cf_action in variants.items():
                student_cf = core_model.training_outputs(
                    z[mask],
                    cf_action[mask],
                    t_values=student_outputs['t_values'][mask],
                    eps=student_outputs['eps'][mask],
                )
                with torch.no_grad():
                    teacher_cf = teacher.training_outputs(
                        z[mask].detach(),
                        cf_action[mask],
                        t_values=student_outputs['t_values'][mask],
                        eps=student_outputs['eps'][mask],
                    )
                resp_total = resp_total + response_kd_loss(
                    student_outputs['v_pred'][mask],
                    student_cf['v_pred'],
                    teacher_outputs['v_pred'],
                    teacher_cf['v_pred'],
                ) / max(1, len(variants))

    if used_buckets > 0:
        kd_total = kd_total / used_buckets
        resp_total = resp_total / used_buckets
    total = lambda_kd * kd_total + mu_resp * resp_total
    logs = {
        "train/kd_loss": kd_total.detach(),
        "train/response_kd_loss": resp_total.detach(),
        "train/action_bucket_threshold": used_threshold,
    }
    return total, logs

def _bucket_name_to_id(name):
    key = str(name).lower()
    if key == "small":
        return 0
    if key == "large":
        return 1
    return int(name)

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
    global _model, _optimizer, _step, _epoch, _ckpt_dir, _wandb_run_id, _checkpoint_enabled
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in wandb dir")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Explicit path to checkpoint to resume from")
    parser.add_argument("--no_checkpoints", action="store_true",
                        help="Disable all checkpoint writes, including emergency saves.")
    parser.add_argument("--export_model_path", type=str, default=None,
                        help="Optionally export model weights only at the end (use /dev/shm for disk-safe runs).")
    args = parser.parse_args()
    _checkpoint_enabled = not args.no_checkpoints
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    # Optional: Load model config from a separate file if provided
    model_type = config.get('model_type', None)
    if model_type:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        model_config_path = os.path.join(project_root, "configs", "model", f"{model_type}.yaml")
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
    distillation_teachers = build_distillation_teachers(
        config, dynamics_class, model_name, model_config, device, rank=rank
    )
    
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
        wandb_cfg = config.get('wandb', {})
        if wandb_cfg.get('api_key') and wandb_cfg['api_key'] != "YOUR_WANDB_API_KEY_HERE":
            os.environ["WANDB_API_KEY"] = wandb_cfg['api_key']
        elif "WANDB_MODE" not in os.environ and "WANDB_API_KEY" not in os.environ:
            os.environ["WANDB_MODE"] = "offline"
            print("--- WANDB_API_KEY not set; defaulting WANDB_MODE=offline ---")
        
        wandb.init(
            project=wandb_cfg['project'],
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

    bucket_cfg = config.get('action_bucket', {})
    train_filter_bucket = bucket_cfg.get('train_filter_bucket')
    if (train_filter_bucket is not None or config.get('distillation', {}).get('enabled', False)) and bucket_cfg.get('threshold') is None:
        quantile = float(bucket_cfg.get('quantile', 0.5))
        threshold, score_type = estimate_action_bucket_threshold(
            train_dataset, quantile=quantile, return_score_type=True
        )
        bucket_cfg['threshold'] = threshold
        bucket_cfg['score_type'] = score_type
        if rank == 0:
            print(
                f"--- Estimated action bucket threshold={bucket_cfg['threshold']:.6f} "
                f"(quantile={quantile}, score_type={score_type}) ---"
            )
    if train_filter_bucket is not None:
        threshold = bucket_cfg.get('threshold')
        quantile = float(bucket_cfg.get('quantile', 0.5))
        score_type = bucket_cfg.get('score_type', 'auto')
        before = len(train_dataset)
        train_dataset = filter_dataset_by_action_bucket(
            train_dataset,
            bucket=train_filter_bucket,
            threshold=threshold,
            quantile=quantile,
            score_type=score_type,
        )
        if rank == 0:
            print(
                f"--- Action bucket filter: train={train_filter_bucket}, "
                f"windows {before} -> {len(train_dataset)} ---"
            )

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
            core_model = model.module if hasattr(model, 'module') else model
            if config.get('distillation', {}).get('enabled', False):
                student_outputs = core_model.training_outputs(z, action)
                base_loss = student_outputs['loss']
                distill_loss, distill_logs = compute_distillation_loss(
                    core_model,
                    distillation_teachers,
                    config.get('distillation', {}),
                    config.get('action_bucket', {}),
                    z,
                    action,
                    student_outputs,
                )
                loss = base_loss + distill_loss
            else:
                base_loss = None
                distill_logs = {}
                loss = core_model.training_loss(z, action)
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
                    log_payload = {
                        "train/loss": loss.item(),
                        "train/grad_norm": grad_norm,
                        "train/epoch": epoch,
                        "time/data_loading": data_time,
                        "time/training_step": train_step_time,
                        "time/seconds_per_step": step_time_total,
                    }
                    if base_loss is not None:
                        log_payload["train/base_loss"] = base_loss.item()
                    for key, value in distill_logs.items():
                        log_payload[key] = value.item() if torch.is_tensor(value) else value
                    wandb.log(log_payload, step=step)
                
                # Periodic Evaluation (Trigger at val_freq or step 10 for quick verification)
                val_freq = config['training'].get('val_freq', 1000)
                if step % val_freq == 0 or step == 10:
                    print(f"\n--- Running Evaluation at Step {step} ---")
                    eval_start = time.time()
                    
                    # Evaluate In-Distribution
                    log_videos = config['training'].get('eval_log_videos', True)
                    eval_batches = config['training'].get('eval_num_batches', 10)
                    ind_metrics = evaluator.evaluate(
                        ind_test_loader, "ind_test", step,
                        num_batches=eval_batches, log_videos=log_videos,
                    )
                    if rank == 0:
                        print(f"[eval step={step}] {ind_metrics}")
                    wandb.log(ind_metrics, step=step)
                    
                    # Evaluate Out-of-Distribution
                    if ood_test_loader:
                        ood_metrics = evaluator.evaluate(
                            ood_test_loader, "ood_test", step,
                            num_batches=eval_batches, log_videos=log_videos,
                        )
                        if rank == 0:
                            print(f"[eval step={step}] {ood_metrics}")
                        wandb.log(ood_metrics, step=step)
                    
                    eval_time = time.time() - eval_start
                    wandb.log({"time/evaluation": eval_time}, step=step)
                    print(f"--- Evaluation Finished (Took {eval_time:.2f}s) ---")
                
                # Checkpoints
                ckpt_freq = config['training'].get('checkpoint_freq', 2000)
                if _checkpoint_enabled and ckpt_freq and step % ckpt_freq == 0:
                    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{step}.pt")
                    save_checkpoint(model, optimizer, step, epoch, ckpt_path, wandb_run_id=wandb_run_id)
                elif _checkpoint_enabled and step % 500 == 0:
                    save_checkpoint(model, optimizer, step, epoch, None, wandb_run_id=wandb_run_id, save_numbered=False)
            
            # Critical: Sync all processes before starting next step to disentangle eval time
            if world_size > 1:
                dist.barrier()
            last_step_end = time.time()
                    
        if world_size > 1:
            dist.barrier()
        
    if rank == 0 and args.export_model_path:
        export_path = os.path.abspath(args.export_model_path)
        export_dir = os.path.dirname(export_path)
        if export_dir:
            os.makedirs(export_dir, exist_ok=True)
        export_model = model.module if hasattr(model, 'module') else model
        torch.save({
            'model_state_dict': export_model.state_dict(),
            'step': step,
            'epoch': _epoch,
        }, export_path)
        print(f"--- Exported model weights only: {export_path} ---")

    if rank == 0:
        wandb.finish()
    cleanup_ddp()

if __name__ == "__main__":
    main()

import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any, List, Optional
from collections import OrderedDict
import time

# Import configurations from data_config.py
try:
    from acwm.dataset.data_config import DatasetConfig, get_config_by_name
except ImportError:
    from data_config import DatasetConfig, get_config_by_name

# --- Dataset Implementation ---

class BaseRoboticsDataset(Dataset):
    """
    Unified Dataset for robotics data. Handles MP4 loading with window sampling and caching.
    """
    def __init__(self, config: DatasetConfig, split: Optional[str] = None, max_trajs: Optional[int] = None):
        self.config = config
        self.split = split
        self.test_cuts = config.test_cuts
        
        # If split is provided, adjust root_dir
        effective_root = config.root_dir
        if split:
            effective_root = os.path.join(config.root_dir, split)
        
        self.effective_root = effective_root
        self.metadata_path = os.path.join(effective_root, "metadata.pt")
        self.metadata_lite_path = os.path.join(effective_root, "metadata_lite.pt")
        
        # Load lite metadata for initialization if it exists, otherwise use full
        if os.path.exists(self.metadata_lite_path):
            print(f"[{config.name}]({split if split else 'root'}) Initializing from LITE metadata...")
            self.init_metadata = torch.load(self.metadata_lite_path, weights_only=False)
        elif os.path.exists(self.metadata_path):
            print(f"[{config.name}]({split if split else 'root'}) Initializing from FULL metadata (lite not found)...")
            self.init_metadata = torch.load(self.metadata_path, weights_only=False)
        else:
            print(f"[{config.name}]({split if split else 'root'}) WARNING: metadata.pt not found at {self.metadata_path}")
            self.init_metadata = []

        metadata_items = list(enumerate(self.init_metadata))

        # Deterministic Sampling if max_trajs is provided
        if max_trajs and len(self.init_metadata) > max_trajs:
            import random
            # Use seed 0 for test splits as requested, 42 for others
            seed = 0 if (self.split and 'test' in self.split) else 42
            rng = random.Random(seed) 
            sampled_indices = sorted(rng.sample(range(len(self.init_metadata)), max_trajs))
            metadata_items = [(i, self.init_metadata[i]) for i in sampled_indices]
            print(f"[{config.name}]({split if split else 'root'}) Sampled {max_trajs} trajectories deterministically.")
            
        # Build indices efficiently
        self.indices = []
        missing_videos = []
        required_len = (config.seq_len - 1) * config.sampling_rate + 1
        for traj_idx, entry in metadata_items:
            video_rel_path = entry.get('video_path')
            video_path = os.path.join(self.effective_root, video_rel_path) if video_rel_path else None
            if not video_path or not os.path.isfile(video_path):
                missing_videos.append(video_rel_path or f"trajectory {traj_idx}")
                continue
            t_len = entry['length'] if 'length' in entry else self._get_traj_len(entry)
            if t_len >= required_len:
                if self.test_cuts and self.split and 'test' in self.split:
                    # Deterministic uniform cutting for test sets
                    if self.test_cuts == 1:
                        self.indices.append((traj_idx, 0))
                    else:
                        # Uniformly space the start indices
                        # max possible start index is t_len - required_len
                        max_start = t_len - required_len
                        # Use floor to ensure integer indices
                        start_indices = np.linspace(0, max_start, self.test_cuts, dtype=int)
                        for start_f in start_indices:
                            self.indices.append((traj_idx, int(start_f)))
                else:
                    # Standard sliding window for training or when test_cuts not set
                    # Add all valid starting positions
                    # A window of span required_len can start at 0, 1, ..., t_len - required_len
                    for start_f in range(t_len - required_len + 1):
                        self.indices.append((traj_idx, start_f))
        
        # Free up init_metadata memory
        self.init_metadata = None
        self._full_metadata = None
        
        print(f"[{config.name}]({split if split else 'root'}) Initialized: {len(self.indices)} windows.")
        if missing_videos:
            example = ", ".join(missing_videos[:3])
            print(
                f"[{config.name}]({split if split else 'root'}) Skipped "
                f"{len(missing_videos)} trajectories with missing video files (e.g. {example})."
            )
        self.cache = OrderedDict()

    @property
    def full_metadata(self):
        if self._full_metadata is None:
            print(f"[{self.config.name}]({self.split if self.split else 'root'}) Lazy-loading FULL metadata...")
            self._full_metadata = torch.load(self.metadata_path, weights_only=False, mmap=True)
        return self._full_metadata

    def _get_traj_len(self, entry: Dict[str, Any]) -> int:
        if 'actions' in entry:
            return entry['actions'].shape[0]
        if 'length' in entry:
            return entry['length']
        if 'commands' in entry:
            if isinstance(entry['commands'], dict):
                return entry['commands']['linear_velocity'].shape[0]
            return entry['commands'].shape[0]
        return 0

    def _load_video(self, video_rel_path: str) -> torch.Tensor:
        if video_rel_path in self.cache:
            self.cache.move_to_end(video_rel_path)
            return self.cache[video_rel_path]

        video_path = os.path.join(self.effective_root, video_rel_path)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(
                f"video file is missing: {video_path}. Download the referenced split before training/evaluation."
            )
        cap = cv2.VideoCapture(video_path)
        frames = []
        target_h, target_w = self.config.obs_shape[1], self.config.obs_shape[2]
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[:2] != (target_h, target_w):
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if not frames:
            raise RuntimeError(f"could not decode any frames from {video_path}")

        # (T, H, W, C) -> (T, C, H, W)
        video_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()
        
        if len(self.cache) >= self.config.cache_size:
            self.cache.popitem(last=False)
        self.cache[video_rel_path] = video_tensor
        return video_tensor

    def _get_action_slice(self, entry: Dict[str, Any], start: int, end: int) -> torch.Tensor:
        """Extract raw action slice without padding."""
        # Check for 'actions' or 'commands' in entry
        if 'actions' in entry:
            return entry['actions'][start:end]
        elif 'commands' in entry:
            # RECON commands are linear_velocity and angular_velocity
            cmds = entry['commands']
            if isinstance(cmds, dict):
                lin = cmds['linear_velocity'][start:end]
                ang = cmds['angular_velocity'][start:end]
                return torch.stack([lin, ang], dim=-1)
            else:
                return cmds[start:end]
        return torch.zeros((end - start, self.config.action_dim))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        traj_idx, start_f = self.indices[idx]
        entry = self.full_metadata[traj_idx]
        
        full_video = self._load_video(entry['video_path'])
        
        # Calculate indices to sample
        required_span = (self.config.seq_len - 1) * self.config.sampling_rate + 1
        sample_indices = torch.arange(start_f, start_f + required_span, self.config.sampling_rate)
        
        # Filter indices that are within video bounds
        valid_mask = sample_indices < full_video.shape[0]
        valid_indices = sample_indices[valid_mask]
        
        if len(valid_indices) > 0:
            obs_window = full_video[valid_indices]
        else:
            obs_window = torch.zeros((0, *self.config.obs_shape), dtype=torch.uint8)

        # Handle padding if video is shorter than expected or indices are out of bounds
        if obs_window.shape[0] < self.config.seq_len:
            if obs_window.shape[0] == 0:
                if full_video.shape[0] > 0:
                    last_frame = full_video[-1:]
                    obs_window = last_frame.repeat(self.config.seq_len, 1, 1, 1)
                else:
                    obs_window = torch.zeros((self.config.seq_len, *self.config.obs_shape), dtype=torch.float32)
            else:
                last_frame = obs_window[-1:]
                pad_len = self.config.seq_len - obs_window.shape[0]
                padding = last_frame.repeat(pad_len, 1, 1, 1)
                obs_window = torch.cat([obs_window, padding], dim=0)
        
        # Sample actions
        # We need to extract the full span first then sample, or just sample directly if supported
        full_action_span = self._get_action_slice(entry, start_f, start_f + required_span)
        
        # Indices relative to start_f
        rel_indices = torch.arange(0, required_span, self.config.sampling_rate)
        
        # Filter relative indices that are within the slice bounds
        action_valid_mask = rel_indices < full_action_span.shape[0]
        action_valid_indices = rel_indices[action_valid_mask]
        
        if len(action_valid_indices) > 0:
            action_window = full_action_span[action_valid_indices]
        else:
            action_window = torch.zeros((0, self.config.action_dim))

        # Handle padding for actions
        if action_window.shape[0] < self.config.seq_len:
            pad_len = self.config.seq_len - action_window.shape[0]
            if action_window.shape[0] > 0:
                padding = action_window[-1:].repeat(pad_len, 1)
            else:
                padding = torch.zeros((pad_len, self.config.action_dim))
            action_window = torch.cat([action_window, padding], dim=0)

        res = {
            "obs": obs_window.float() / 255.0 if obs_window.dtype == torch.uint8 else obs_window,
            "action": action_window,
            "traj_idx": traj_idx,
            "start_f": start_f
        }
        
        if 'task_id' in entry:
            res['task_id'] = entry['task_id']
            
        return res

class RoboticsDatasetWrapper:
    """
    Helper to instantiate datasets by name using pre-defined configs.
    """
    @staticmethod
    def get_dataset(name: str, split: Optional[str] = None, max_trajs: Optional[int] = None, **kwargs) -> BaseRoboticsDataset:
        """
        Instantiates a BaseRoboticsDataset by looking up the configuration by name.
        kwargs can be used to override default configuration parameters.
        """
        config = get_config_by_name(name, **kwargs)
        return BaseRoboticsDataset(config, split=split, max_trajs=max_trajs)

if __name__ == "__main__":
    # To run this script directly, we need to handle the relative import
    # This block allows running `python wm/dataset/dataset.py` from the project root
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    
    print("\n--- Testing Individual Datasets ---")
    
    for name in ["language_table", "dreamer4"]:
        print(f"\nTesting {name}...")
        try:
            dataset = RoboticsDatasetWrapper.get_dataset(name, seq_len=5, obs_shape=(3, 64, 64))
            loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
            
            
            start_time = time.time()
            for i, batch in enumerate(loader):
                if i == 0:
                    print(f"  Obs Shape: {batch['obs'].shape}")
                    print(f"  Action Shape: {batch['action'].shape}")
                if i >= 4: break
            
            end_time = time.time()
            print(f"  Load time for 5 batches: {end_time - start_time:.2f}s")
        except Exception as e:
            print(f"  Failed to test {name}: {e}")

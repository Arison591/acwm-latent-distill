import os
import inspect
from dataclasses import dataclass
from typing import Tuple, Optional

# Set ACWM_DATA_ROOT to the directory containing the downloaded ACWM-Phys dataset.
# Structure expected:
#   $ACWM_DATA_ROOT/rigid_dynamics/{push_block,stack_cube}/
#   $ACWM_DATA_ROOT/deformable/{push_rope,clothmove}/
#   $ACWM_DATA_ROOT/particle/{push_sand,pour_water}/
#   $ACWM_DATA_ROOT/kinematics/{robot_arm_64,reacher}/
DATASET_ROOT = os.environ.get("ACWM_DATA_ROOT", "./data")


@dataclass
class DatasetConfig:
    name: str
    root_dir: str
    action_dim: int
    obs_shape: Tuple[int, int, int] = (3, 128, 128)
    seq_len: int = 10
    sampling_rate: int = 1
    fps: float = 10.0
    cache_size: int = 50
    train_dataset_size: Optional[int] = None
    test_ind_dataset_size: Optional[int] = None
    test_ood_dataset_size: Optional[int] = None
    test_cuts: Optional[int] = None


# ── ACWM-Phys core environments ───────────────────────────────────────────────

PUSH_CUBE_CONFIG = {
    "name": "push_cube",
    "action_dim": 2,
    "root_dir": os.path.join(DATASET_ROOT, "rigid_dynamics/push_block/"),
    "seq_len": 37,
    "obs_shape": (3, 240, 240),
    "train_dataset_size": 1987,
    "test_ind_dataset_size": 100,
    "test_ood_dataset_size": 100,
}

STACK_CUBE_CONFIG = {
    "name": "stack_cube",
    "action_dim": 7,
    "root_dir": os.path.join(DATASET_ROOT, "rigid_dynamics/stack_cube/"),
    "seq_len": 37,
    "obs_shape": (3, 240, 240),
    "train_dataset_size": 1987,
    "test_ind_dataset_size": 100,
    "test_ood_dataset_size": 100,
}

PUSH_ROPE_CONFIG = {
    "name": "push_rope",
    "action_dim": 2,
    "root_dir": os.path.join(DATASET_ROOT, "deformable/push_rope/"),
    "seq_len": 37,
    "obs_shape": (3, 240, 240),
    "train_dataset_size": 1987,
    "test_ind_dataset_size": 100,
    "test_ood_dataset_size": 100,
}

CLOTHMOVE_CONFIG = {
    "name": "clothmove",
    "action_dim": 8,
    "root_dir": os.path.join(DATASET_ROOT, "deformable/clothmove/"),
    "seq_len": 37,
    "obs_shape": (3, 240, 240),
    "train_dataset_size": 1987,
    "test_ind_dataset_size": 100,
    "test_ood_dataset_size": 100,
}

PUSH_SAND_CONFIG = {
    "name": "push_sand",
    "action_dim": 7,
    "root_dir": os.path.join(DATASET_ROOT, "particle/push_sand/"),
    "seq_len": 37,
    "obs_shape": (3, 240, 400),
    "train_dataset_size": 1784,
    "test_ind_dataset_size": 100,
    "test_ood_dataset_size": 100,
}

POUR_WATER_CONFIG = {
    "name": "pour_water",
    "action_dim": 4,
    "root_dir": os.path.join(DATASET_ROOT, "particle/pour_water/"),
    "seq_len": 16,
    "obs_shape": (3, 128, 128),
    "fps": 10.0,
    "train_dataset_size": 1000,
    "test_ind_dataset_size": 50,
    "test_ood_dataset_size": 50,
}

ROBOT_ARM_CONFIG = {
    "name": "robot_arm",
    "action_dim": 7,
    "root_dir": os.path.join(DATASET_ROOT, "kinematics/robot_arm_64/"),
    "seq_len": 37,
    "obs_shape": (3, 240, 240),
    "train_dataset_size": 1987,
    "test_ind_dataset_size": 100,
    "test_ood_dataset_size": 100,
}

REACHER_CONFIG = {
    "name": "reacher",
    "action_dim": 2,
    "root_dir": os.path.join(DATASET_ROOT, "kinematics/reacher/"),
    "seq_len": 37,
    "obs_shape": (3, 240, 240),
    "train_dataset_size": 1987,
    "test_ind_dataset_size": 100,
    "test_ood_dataset_size": 100,
}


DATASET_REGISTRY = {
    "push_cube":  PUSH_CUBE_CONFIG,
    "stack_cube": STACK_CUBE_CONFIG,
    "push_rope":  PUSH_ROPE_CONFIG,
    "clothmove":  CLOTHMOVE_CONFIG,
    "push_sand":  PUSH_SAND_CONFIG,
    "pour_water": POUR_WATER_CONFIG,
    "robot_arm":  ROBOT_ARM_CONFIG,
    "reacher":    REACHER_CONFIG,
}


def get_config_by_name(name: str, **kwargs) -> DatasetConfig:
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    config_dict = DATASET_REGISTRY[name].copy()
    if "compression" in kwargs and "sampling_rate" not in kwargs:
        kwargs["sampling_rate"] = kwargs.pop("compression")
    config_dict.update(kwargs)
    valid_keys = inspect.signature(DatasetConfig).parameters.keys()
    filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
    return DatasetConfig(**filtered)

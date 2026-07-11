from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from acwm.action_latent.buckets import assign_magnitude_buckets, compute_bucket_score, compute_chunk_magnitude
from acwm.action_latent.counterfactual import counterfactual_scale, counterfactual_semantics, make_counterfactual_actions
from acwm.action_latent.action_stats import action_statistics, magnitude_split_diagnostics, motion_effect_diagnostics
from acwm.action_latent.encoder import ConvActionEncoder, IdentityActionEncoder, MLPActionEncoder
from acwm.distill.losses import kd_loss, prediction_loss, response_kd_loss
from acwm.dataset.data_config import DatasetConfig
from acwm.dataset.dataset import BaseRoboticsDataset


def test_action_encoders() -> None:
    actions = torch.randn(3, 5, 2)
    assert IdentityActionEncoder(action_dim=2)(actions).shape == (3, 5, 2)
    assert MLPActionEncoder(action_dim=2, latent_dim=64)(actions).shape == (3, 5, 64)
    assert ConvActionEncoder(action_dim=2, latent_dim=32)(actions).shape == (3, 5, 32)


def test_buckets_and_counterfactuals() -> None:
    actions = torch.tensor(
        [
            [[0.0, 0.0], [0.1, 0.0]],
            [[2.0, 0.0], [2.0, 0.0]],
        ]
    )
    mags = compute_chunk_magnitude(actions)
    assert mags.shape == (2,)
    scores = compute_bucket_score(actions)
    assert scores.shape == (2,)
    buckets, threshold = assign_magnitude_buckets(actions)
    assert threshold > 0
    assert buckets.tolist() == [0, 1]

    unit_actions = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[-1.0, 0.0], [-1.0, 0.0]],
        ]
    )
    unit_buckets, _ = assign_magnitude_buckets(unit_actions)
    assert sorted(unit_buckets.tolist()) == [0, 1]

    variants = make_counterfactual_actions(actions, ["zero", "reverse", "scale_0_25", "scale_0_5", "scale_0_75", "scale_1_5", "scale_2", "shuffle"])
    assert torch.equal(variants["zero"], torch.zeros_like(actions))
    assert torch.equal(variants["reverse"], -actions)
    assert torch.allclose(variants["scale_0_5"], actions * 0.5)
    assert torch.allclose(variants["scale_0_25"], actions * 0.25)
    assert torch.allclose(variants["scale_0_75"], actions * 0.75)
    assert torch.allclose(variants["scale_1_5"], actions * 1.5)
    assert torch.allclose(variants["scale_2"], actions * 2.0)
    assert counterfactual_scale("scale_1_5") == 1.5
    assert "opposite" in counterfactual_semantics("reacher", ["reverse"])["reverse"]
    assert torch.equal(variants["shuffle"], actions.flip(1))


def test_distillation_losses_detach_teacher() -> None:
    student = torch.randn(2, 3, 4, requires_grad=True)
    target = torch.randn(2, 3, 4)
    teacher = torch.randn(2, 3, 4, requires_grad=True)
    student_cf = torch.randn(2, 3, 4, requires_grad=True)
    teacher_cf = torch.randn(2, 3, 4, requires_grad=True)

    loss = prediction_loss(student, target) + kd_loss(student, teacher)
    loss = loss + response_kd_loss(student, student_cf, teacher, teacher_cf)
    loss.backward()

    assert student.grad is not None
    assert student_cf.grad is not None
    assert teacher.grad is None
    assert teacher_cf.grad is None


def test_action_statistics() -> None:
    metadata = [
        {"actions": torch.tensor([[1.0, 0.0], [0.0, 2.0], [-1.0, 0.0]])},
        {"actions": torch.tensor([[0.0, -0.5], [0.0, -1.5], [1.0, 1.0]])},
    ]
    stats = action_statistics(metadata, max_lag=1)
    assert stats["action_dim"] == 2
    assert stats["step_l2_norm"]["count"] == 6
    assert sum(item["count"] for item in stats["quadrant_distribution"].values()) == 6
    split = magnitude_split_diagnostics(metadata)
    assert split["low_count"] + split["high_count"] == 2
    motion = motion_effect_diagnostics(torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([0.1, 0.2, 0.5, 0.7]), 2.0)
    assert motion["high_motion_mean"] > motion["low_motion_mean"]
    assert motion["pearson_action_motion"] > 0


def test_dataset_skips_missing_videos() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        torch.save(
            [{
                "video_path": "missing.mp4",
                "length": 2,
                "actions": torch.zeros(2, 1),
            }],
            root / "metadata.pt",
        )
        dataset = BaseRoboticsDataset(
            DatasetConfig(
                name="smoke",
                root_dir=str(root),
                action_dim=1,
                obs_shape=(3, 8, 8),
                seq_len=2,
            )
        )
        assert len(dataset) == 0


if __name__ == "__main__":
    test_action_encoders()
    test_buckets_and_counterfactuals()
    test_distillation_losses_detach_teacher()
    test_action_statistics()
    test_dataset_skips_missing_videos()
    print("ALRD smoke tests passed")

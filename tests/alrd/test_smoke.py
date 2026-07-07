from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from acwm.action_latent.buckets import assign_magnitude_buckets, compute_bucket_score, compute_chunk_magnitude
from acwm.action_latent.counterfactual import make_counterfactual_actions
from acwm.action_latent.encoder import ConvActionEncoder, IdentityActionEncoder, MLPActionEncoder
from acwm.distill.losses import kd_loss, prediction_loss, response_kd_loss


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

    variants = make_counterfactual_actions(actions, ["zero", "reverse", "scale_0_5", "shuffle"])
    assert torch.equal(variants["zero"], torch.zeros_like(actions))
    assert torch.equal(variants["reverse"], -actions)
    assert torch.allclose(variants["scale_0_5"], actions * 0.5)
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


if __name__ == "__main__":
    test_action_encoders()
    test_buckets_and_counterfactuals()
    test_distillation_losses_detach_teacher()
    print("ALRD smoke tests passed")

import math

import pytest
import torch

from qwen35_moderation.preference import discrete_dpo_loss


def test_identical_policy_and_reference_have_log_two_loss() -> None:
    logits = torch.tensor([[1.0, 0.0, -1.0], [0.0, 2.0, 1.0]])
    result = discrete_dpo_loss(
        logits, logits.clone(), torch.tensor([0, 1]), torch.tensor([2, 0]), beta=0.5
    )
    assert result.loss.item() == pytest.approx(math.log(2.0))
    assert torch.equal(result.advantages, torch.zeros(2))


def test_preferred_action_improvement_reduces_loss() -> None:
    reference = torch.zeros((2, 3))
    improved = torch.tensor([[3.0, 0.0, -2.0], [-1.0, 2.0, 0.0]])
    result = discrete_dpo_loss(
        improved, reference, torch.tensor([0, 1]), torch.tensor([2, 0]), beta=1.0
    )
    assert result.loss.item() < math.log(2.0)
    assert torch.all(result.policy_margins > 0)
    assert torch.all(result.advantages > 0)


def test_equal_actions_are_rejected() -> None:
    with pytest.raises(ValueError, match="不能相同"):
        discrete_dpo_loss(
            torch.zeros((1, 3)),
            torch.zeros((1, 3)),
            torch.tensor([1]),
            torch.tensor([1]),
            beta=1.0,
        )

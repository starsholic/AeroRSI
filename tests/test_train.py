import pytest
import torch

from aerorsi.train import epoch_plan, flush_left


def test_epoch_plan_scales_the_tail():
    indices, weights = epoch_plan(19, 16)
    assert len(indices) == 32 and indices[:19] == list(range(19)) and indices[19:] == [0] * 13
    assert weights[:16] == [1.0] * 16 and weights[16:19] == [16 / 3] * 3 and weights[19:] == [0.0] * 13
    assert sum(weights) == pytest.approx(32)  # the tail counts as one full batch


def test_flush_left():
    mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 0, 0]])
    values = torch.tensor([[9, 9, 2, 3, 4], [9, 5, 6, 9, 9]])
    new_mask, new_values = flush_left(mask, values)
    assert torch.equal(new_mask, torch.tensor([[1, 1, 1], [1, 1, 0]]))
    assert torch.equal(new_values[:, :2], torch.tensor([[2, 3], [5, 6]]))

import pytest
import torch

from supertriplets.distance import CosineDistance, EuclideanDistance
from supertriplets.loss import (
    BaseBatchTripletMiner,
    BatchHardSoftMarginTripletLoss,
    BatchHardTripletLoss,
)


def test_base_batch_triplet_miner(torch_labels):
    miner = BaseBatchTripletMiner()
    pos_expected = torch.tensor(
        [
            [False, False, False, True],
            [False, False, True, False],
            [False, True, False, False],
            [True, False, False, False],
        ]
    )
    neg_expected = torch.tensor(
        [[False, True, True, False], [True, False, False, True], [True, False, False, True], [False, True, True, False]]
    )
    assert miner.get_anchor_positive_triplet_mask(labels=torch_labels).equal(pos_expected)
    assert miner.get_anchor_negative_triplet_mask(labels=torch_labels).equal(neg_expected)


@pytest.mark.parametrize(
    "distance, expected_loss",
    [
        (EuclideanDistance(squared=False), torch.tensor(5.4048)),
        (CosineDistance(alredy_l2_normalized_vectors=False), torch.tensor(5.6197)),
    ],
)
def test_batch_hard_triplet_loss(distance, expected_loss):
    embeddings = torch.tensor([[0.1, 0.3, 0.5], [0.6, 0.8, 1.0], [0.8, 0.6, 0.0], [0.9, 0.2, -0.5]])
    labels = torch.tensor([0, 1, 1, 0])
    criterion = BatchHardTripletLoss(distance=distance, margin=5.0)
    loss = criterion(embeddings=embeddings, labels=labels)
    assert isinstance(loss, torch.Tensor)
    assert torch.allclose(loss, expected_loss, atol=1e-3)


@pytest.mark.parametrize(
    "distance, expected_loss",
    [
        (EuclideanDistance(squared=False), torch.tensor(0.9191)),
        (CosineDistance(alredy_l2_normalized_vectors=False), torch.tensor(1.0709)),
    ],
)
def test_batch_hard_soft_margin_triplet_loss(distance, expected_loss):
    embeddings = torch.tensor([[0.1, 0.3, 0.5], [0.6, 0.8, 1.0], [0.8, 0.6, 0.0], [0.9, 0.2, -0.5]])
    labels = torch.tensor([0, 1, 1, 0])
    criterion = BatchHardSoftMarginTripletLoss(distance=distance)
    loss = criterion(embeddings=embeddings, labels=labels)
    assert isinstance(loss, torch.Tensor)
    assert torch.allclose(loss, expected_loss, atol=1e-3)

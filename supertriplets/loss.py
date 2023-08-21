from typing import Union

import torch
import torch.nn as nn
from torch import Tensor

from .distance import CosineDistance, EuclideanDistance


class BaseBatchTripletMiner:
    @staticmethod
    def get_anchor_positive_triplet_mask(labels: Tensor) -> Tensor:
        indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
        indices_not_equal = ~indices_equal
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        return labels_equal & indices_not_equal

    @staticmethod
    def get_anchor_negative_triplet_mask(labels: Tensor) -> Tensor:
        return ~(labels.unsqueeze(0) == labels.unsqueeze(1))


class BatchHardTripletLoss(BaseBatchTripletMiner, nn.Module):
    def __init__(
        self,
        distance: Union[EuclideanDistance, CosineDistance] = EuclideanDistance(squared=False),
        margin: float = 5,
    ) -> None:
        super(BatchHardTripletLoss, self).__init__()
        self.distance = distance
        self.margin = margin

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        pairwise_dist = self.distance(embeddings, embeddings)
        mask_anchor_positive = self.get_anchor_positive_triplet_mask(labels).float()
        anchor_positive_dist = mask_anchor_positive * pairwise_dist
        hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

        mask_anchor_negative = self.get_anchor_negative_triplet_mask(labels).float()
        max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
        hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

        tl = hardest_positive_dist - hardest_negative_dist + self.margin
        tl[tl < 0] = 0
        triplet_loss = tl.mean()
        return triplet_loss


class BatchHardSoftMarginTripletLoss(BaseBatchTripletMiner, nn.Module):
    def __init__(
        self,
        distance: Union[EuclideanDistance, CosineDistance] = EuclideanDistance(squared=False),
    ) -> None:
        super(BatchHardSoftMarginTripletLoss, self).__init__()
        self.distance = distance

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        pairwise_dist = self.distance(embeddings, embeddings)
        mask_anchor_positive = self.get_anchor_positive_triplet_mask(labels).float()
        anchor_positive_dist = mask_anchor_positive * pairwise_dist
        hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

        mask_anchor_negative = self.get_anchor_negative_triplet_mask(labels).float()
        max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
        hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

        tl = torch.log1p(torch.exp(hardest_positive_dist - hardest_negative_dist))
        triplet_loss = tl.mean()
        return triplet_loss

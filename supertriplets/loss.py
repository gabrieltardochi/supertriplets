from typing import Union

import torch
import torch.nn as nn
from torch import Tensor

from .distance import CosineDistance, EuclideanDistance


class BaseBatchTripletMiner:
    """
    A base class providing utility methods for triplet mining loss computation.
    """

    @staticmethod
    def get_anchor_positive_triplet_mask(labels: Tensor) -> Tensor:
        """
        Generate a mask for valid anchor-positive pairs.

        Args:
            labels (Tensor):
                The labels of the samples.

        Returns:
            Tensor: A mask indicating valid anchor-positive pairs.
        """
        indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
        indices_not_equal = ~indices_equal
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        return labels_equal & indices_not_equal

    @staticmethod
    def get_anchor_negative_triplet_mask(labels: Tensor) -> Tensor:
        """
        Generate a mask for valid anchor-negative pairs.

        Args:
            labels (Tensor):
                The labels of the samples.

        Returns:
            Tensor: A mask indicating valid anchor-negative pairs.
        """
        return ~(labels.unsqueeze(0) == labels.unsqueeze(1))


class BatchHardTripletLoss(BaseBatchTripletMiner, nn.Module):
    """
    This class implements the Batch Hard Triplet Loss, a loss function used for training
    neural networks in tasks like metric learning and embedding learning. It enforces the
    network to map similar inputs closer in the embedding space and dissimilar inputs
    farther apart by selecting the hardest positive and hardest negative examples per batch
    for each anchor.

    Args:
        distance (Union[EuclideanDistance, CosineDistance]):
            The distance metric used to compute pairwise distances between embeddings.
        margin (float):
            The margin by which the positive distance should be less than the negative
            distance in order to incur a loss. Default is 5.

    Attributes:
        distance (Union[EuclideanDistance, CosineDistance]):
            The distance metric used to compute pairwise distances between embeddings.
        margin (float):
            The margin by which the positive distance should be less than the negative
            distance in order to incur a loss.

    Methods:
        forward(embeddings: Tensor, labels: Tensor) -> Tensor:
            Computes the Batch Hard Triplet Loss given the input embeddings and labels.
            Args:
                embeddings (Tensor): The embedding vectors of the examples in the batch.
                labels (Tensor): The corresponding labels of the examples in the batch.
            Returns:
                Tensor: The computed Batch Hard Triplet Loss.

    """

    def __init__(
        self,
        distance: Union[EuclideanDistance, CosineDistance] = EuclideanDistance(squared=False),
        margin: float = 5,
    ) -> None:
        """
        Initializes the BatchHardTripletLoss instance.

        Args:
            distance (Union[EuclideanDistance, CosineDistance]):
                The distance metric used to compute pairwise distances between embeddings.
            margin (float):
                The margin by which the positive distance should be less than the negative
                distance in order to incur a loss. Default is 5.
        """
        super(BatchHardTripletLoss, self).__init__()
        self.distance = distance
        self.margin = margin

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        """
        Computes the Batch Hard Triplet Loss given the input embeddings and labels.

        Args:
            embeddings (Tensor): The embedding vectors of the examples in the batch.
            labels (Tensor): The corresponding labels of the examples in the batch.

        Returns:
            Tensor: The computed Batch Hard Triplet Loss.
        """
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
    """
    This class implements the Batch Hard Soft-Margin Triplet Loss, a loss function used
    for training neural networks in tasks like metric learning and embedding learning.
    Similar to Batch Hard Triplet Loss, it enforces the network to map similar inputs
    closer in the embedding space and dissimilar inputs farther apart. However, it uses
    a soft-margin formulation based on the logistic loss to handle positive and negative
    distances.

    Args:
        distance (Union[EuclideanDistance, CosineDistance]):
            The distance metric used to compute pairwise distances between embeddings.

    Attributes:
        distance (Union[EuclideanDistance, CosineDistance]):
            The distance metric used to compute pairwise distances between embeddings.

    Methods:
        forward(embeddings: Tensor, labels: Tensor) -> Tensor:
            Computes the Batch Hard Soft-Margin Triplet Loss given the input embeddings and
            labels.
            Args:
                embeddings (Tensor): The embedding vectors of the examples in the batch.
                labels (Tensor): The corresponding labels of the examples in the batch.
            Returns:
                Tensor: The computed Batch Hard Soft-Margin Triplet Loss.

    """

    def __init__(
        self,
        distance: Union[EuclideanDistance, CosineDistance] = EuclideanDistance(squared=False),
    ) -> None:
        """
        Initializes the BatchHardSoftMarginTripletLoss instance.

        Args:
            distance (Union[EuclideanDistance, CosineDistance]):
                The distance metric used to compute pairwise distances between embeddings.
        """
        super(BatchHardSoftMarginTripletLoss, self).__init__()
        self.distance = distance

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        """
        Computes the Batch Hard Soft-Margin Triplet Loss given the input embeddings and labels.

        Args:
            embeddings (Tensor): The embedding vectors of the examples in the batch.
            labels (Tensor): The corresponding labels of the examples in the batch.

        Returns:
            Tensor: The computed Batch Hard Soft-Margin Triplet Loss.
        """
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

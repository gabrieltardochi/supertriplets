import torch
from torch import Tensor


class CosineDistance:
    """
    Compute the cosine distance between pairs of vectors.

    Args:
        already_l2_normalized_vectors (bool, optional): If True, assumes that the input vectors are already L2 normalized.
            Defaults to False.

    Methods:
        __call__(self, a: Tensor, b: Tensor) -> Tensor:
            Calculate the cosine distances between two sets of vectors.

    Note:
        If 'already_l2_normalized_vectors' is False, the input vectors are normalized before calculating distances.
    """

    def __init__(self, alredy_l2_normalized_vectors: bool = False) -> None:
        """
        Initialize the CosineDistance instance.

        Args:
            already_l2_normalized_vectors (bool, optional): If True, assumes that the input vectors are already L2 normalized.
                Defaults to False.
        """
        self.alredy_l2_normalized_vectors = alredy_l2_normalized_vectors

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Calculate the cosine distances between two sets of vectors.

        Args:
            a (Tensor): The first set of vectors.
            b (Tensor): The second set of vectors.

        Returns:
            Tensor: A tensor containing the cosine distances between the input vectors.
        """
        if not self.alredy_l2_normalized_vectors:
            a = torch.nn.functional.normalize(a, p=2, dim=1)
            b = torch.nn.functional.normalize(b, p=2, dim=1)
        distances = 1 - torch.mm(a, b.transpose(0, 1))
        return distances


class EuclideanDistance:
    """
    Compute the Euclidean distance between pairs of vectors.

    Args:
        squared (bool, optional): If True, returns squared Euclidean distances. If False, returns regular Euclidean distances.
            Defaults to False.

    Methods:
        __call__(self, a: Tensor, b: Tensor) -> Tensor:
            Calculate the Euclidean distances between two sets of vectors.

    Note:
        If 'squared' is False, small values are added to distances to prevent division by zero.
    """

    def __init__(self, squared: bool = False) -> None:
        """
        Initialize the EuclideanDistance instance.

        Args:
            squared (bool, optional): If True, returns squared Euclidean distances. If False, returns regular Euclidean distances.
                Defaults to False.
        """
        self.squared = squared

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Calculate the Euclidean distances between two sets of vectors.

        Args:
            a (Tensor): The first set of vectors.
            b (Tensor): The second set of vectors.

        Returns:
            Tensor: A tensor containing the Euclidean distances between the input vectors.
        """
        dot_product = torch.matmul(a, b.t())
        square_norm = torch.diag(dot_product)
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances[distances < 0] = 0
        if not self.squared:
            mask = distances.eq(0).float()
            distances = distances + mask * 1e-16
            distances = (1.0 - mask) * torch.sqrt(distances)
        return distances

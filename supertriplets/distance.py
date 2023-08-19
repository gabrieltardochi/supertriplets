import torch
from torch import Tensor


class CosineDistance:
    def __init__(self, alredy_l2_normalized_vectors=False) -> None:
        self.alredy_l2_normalized_vectors = alredy_l2_normalized_vectors

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        if not self.alredy_l2_normalized_vectors:
            a = torch.nn.functional.normalize(a, p=2, dim=1)
            b = torch.nn.functional.normalize(b, p=2, dim=1)
        distances = 1 - torch.mm(a, b.transpose(0, 1))
        return distances


class EuclideanDistance:
    def __init__(self, squared=False) -> None:
        self.squared = squared

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        dot_product = torch.matmul(a, b.t())
        square_norm = torch.diag(dot_product)
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances[distances < 0] = 0
        if not self.squared:
            mask = distances.eq(0).float()
            distances = distances + mask * 1e-16
            distances = (1.0 - mask) * torch.sqrt(distances)
        return distances

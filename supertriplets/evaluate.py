import warnings
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)
from tqdm import tqdm

from .index import TripletMiningEmbeddingsIndex
from .sample import ImageSample, TextImageSample, TextSample


class TripletEmbeddingsEvaluator:
    """
    A class for evaluating triplet embeddings using different distance metrics.

    Attributes:
        calculate_by_cosine (bool): Whether to calculate accuracy using cosine distances.
        calculate_by_manhattan (bool): Whether to calculate accuracy using Manhattan distances.
        calculate_by_euclidean (bool): Whether to calculate accuracy using Euclidean distances.

    Methods:
        __init__(self, calculate_by_cosine=True, calculate_by_manhattan=True, calculate_by_euclidean=True)
            Initializes the TripletEmbeddingsEvaluator with specified distance metric options.

        evaluate(self, embeddings_anchors, embeddings_positives, embeddings_negatives) -> Dict:
            Evaluates triplet embeddings using various distance metrics and returns accuracy metrics.
    """

    def __init__(
        self,
        calculate_by_cosine: bool = True,
        calculate_by_manhattan: bool = True,
        calculate_by_euclidean: bool = True,
    ) -> None:
        """
        Initialize the TripletEmbeddingsEvaluator.

        Args:
            calculate_by_cosine (bool, optional): Whether to calculate accuracy using cosine distances.
            calculate_by_manhattan (bool, optional): Whether to calculate accuracy using Manhattan distances.
            calculate_by_euclidean (bool, optional): Whether to calculate accuracy using Euclidean distances.
        """
        self.calculate_by_cosine = calculate_by_cosine
        self.calculate_by_manhattan = calculate_by_manhattan
        self.calculate_by_euclidean = calculate_by_euclidean

    def evaluate(
        self,
        embeddings_anchors: np.ndarray,
        embeddings_positives: np.ndarray,
        embeddings_negatives: np.ndarray,
    ) -> Dict:
        """
        Evaluate triplet embeddings using specified distance metrics and return accuracy metrics.

        Args:
            embeddings_anchors (np.ndarray): Embeddings of anchor samples.
            embeddings_positives (np.ndarray): Embeddings of positive samples.
            embeddings_negatives (np.ndarray): Embeddings of negative samples.

        Returns:
            Dict: A dictionary containing accuracy metrics based on selected distance metrics.
        """
        metrics = {}

        if self.calculate_by_cosine:
            pos_cosine_distances = paired_cosine_distances(embeddings_anchors, embeddings_positives)
            neg_cosine_distances = paired_cosine_distances(embeddings_anchors, embeddings_negatives)
            metrics["accuracy_cosine"] = sum(pos_cosine_distances < neg_cosine_distances) / len(pos_cosine_distances)

        if self.calculate_by_manhattan:
            pos_manhattan_distances = paired_manhattan_distances(embeddings_anchors, embeddings_positives)
            neg_manhattan_distances = paired_manhattan_distances(embeddings_anchors, embeddings_negatives)
            metrics["accuracy_manhattan"] = sum(pos_manhattan_distances < neg_manhattan_distances) / len(
                pos_manhattan_distances
            )

        if self.calculate_by_euclidean:
            pos_euclidean_distances = paired_euclidean_distances(embeddings_anchors, embeddings_positives)
            neg_euclidean_distances = paired_euclidean_distances(embeddings_anchors, embeddings_negatives)
            metrics["accuracy_euclidean"] = sum(pos_euclidean_distances < neg_euclidean_distances) / len(
                pos_euclidean_distances
            )

        return metrics


class HardTripletsMiner:
    """
    A class for mining hard triplet samples based on given embeddings and examples.

    Attributes:
        use_gpu_powered_index (bool): Whether to use GPU-powered indexing for faster computations if available.

    Methods:
        __init__(self, use_gpu_powered_index_if_available=True)
            Initializes the HardTripletsMiner with the option to use GPU-powered indexing.

        mine(self, examples, embeddings, normalize_l2=True, sample_from_topk_hardest=10)
            Mines hard triplet samples from given embeddings and examples.

    """

    def __init__(
        self,
        use_gpu_powered_index_if_available: bool = True,
    ) -> None:
        """
        Initialize the HardTripletsMiner.

        Args:
            use_gpu_powered_index_if_available (bool, optional):
                Whether to use GPU-powered indexing for faster computations if available.

        """
        self.use_gpu_powered_index = True if torch.cuda.is_available() and use_gpu_powered_index_if_available else False

    def mine(
        self,
        examples: List[Union[TextSample, ImageSample, TextImageSample]],
        embeddings: np.ndarray,
        normalize_l2: bool = True,
        sample_from_topk_hardest: int = 10,
    ) -> Tuple[
        List[Union[TextSample, ImageSample, TextImageSample]],
        List[Union[TextSample, ImageSample, TextImageSample]],
        List[Union[TextSample, ImageSample, TextImageSample]],
    ]:
        """
        Mine hard triplet samples from given embeddings and examples.

        Args:
            examples (List[Union[TextSample, ImageSample, TextImageSample]]):
                List of sample objects representing anchor samples.
            embeddings (np.ndarray): Embeddings associated with the anchor samples.
            normalize_l2 (bool, optional): Whether to normalize the embeddings to unit L2 norm.
            sample_from_topk_hardest (int, optional): Number of hardest samples to consider for each triplet.

        Returns:
            Tuple[List[Union[TextSample, ImageSample, TextImageSample]],
                  List[Union[TextSample, ImageSample, TextImageSample]],
                  List[Union[TextSample, ImageSample, TextImageSample]]]:
                Tuple containing lists of anchor, positive, and negative samples forming hard triplets.
        """
        index = TripletMiningEmbeddingsIndex(
            examples=examples,
            embeddings=embeddings,
            gpu=self.use_gpu_powered_index,
            normalize_l2=normalize_l2,
        )
        anchors = []
        positives = []
        negatives = []
        count_samples_without_pairs = 0
        for anchor, embedding in tqdm(
            zip(examples, embeddings),
            desc="Mining hard triplets for each sample",
        ):
            (
                possible_positives,
                possible_negatives,
            ) = index.search_pos_and_neg_samples(
                array_of_queries=np.expand_dims(embedding, axis=0),
                sample=anchor,
                k=2048,
            )
            possible_hard_positives = possible_positives[-sample_from_topk_hardest:]
            possible_hard_negatives = possible_negatives[:sample_from_topk_hardest]
            if len(possible_hard_positives) > 0 and len(possible_hard_negatives) > 0:
                positive = np.random.choice(
                    possible_hard_positives
                )  # sample from top 'sample_from_topk_hardest' most different positives
                negative = np.random.choice(
                    possible_hard_negatives
                )  # sample from top 'sample_from_topk_hardest' most similar negatives
                anchors.append(anchor)
                positives.append(positive)
                negatives.append(negative)
            else:
                count_samples_without_pairs += 1
        if count_samples_without_pairs > 0:
            warnings.warn(
                f"No positive and negative pairs found for {count_samples_without_pairs} samples (when considered anchors)."
            )
        return anchors, positives, negatives

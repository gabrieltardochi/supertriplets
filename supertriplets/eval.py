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
    def __init__(
        self,
        calculate_by_cosine: bool = True,
        calculate_by_manhattan: bool = True,
        calculate_by_euclidean: bool = True,
    ) -> None:
        self.calculate_by_cosine = calculate_by_cosine
        self.calculate_by_manhattan = calculate_by_manhattan
        self.calculate_by_euclidean = calculate_by_euclidean

    def __call__(
        self,
        embeddings_anchors: np.ndarray,
        embeddings_positives: np.ndarray,
        embeddings_negatives: np.ndarray,
    ) -> Dict:
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
    def __init__(
        self,
        examples: List[Union[TextSample, ImageSample, TextImageSample]],
        embeddings: np.ndarray,
        use_gpu_powered_index_if_available: bool = True,
    ) -> None:
        self.examples = examples
        self.embeddings = embeddings
        self.use_gpu_powered_index = True if torch.cuda.is_available() and use_gpu_powered_index_if_available else False
        self.index = self._get_index()

    def _get_index(self) -> None:
        return TripletMiningEmbeddingsIndex(
            examples=self.examples,
            embeddings=self.embeddings,
            gpu=self.use_gpu_powered_index,
            normalize_l2=True,
        )

    def mine(
        self, sample_from_topk_hardest: int = 10
    ) -> Tuple[
        List[Union[TextSample, ImageSample, TextImageSample]],
        List[Union[TextSample, ImageSample, TextImageSample]],
        List[Union[TextSample, ImageSample, TextImageSample]],
    ]:
        anchors = []
        positives = []
        negatives = []
        count_samples_without_pairs = 0
        for anchor, embedding in tqdm(
            zip(self.examples, self.embeddings),
            desc="Mining hard triplets for each sample",
        ):
            (
                possible_positives,
                possible_negatives,
            ) = self.index.search_pos_and_neg_samples(
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

from typing import List, Tuple, Union

import faiss
import numpy as np

from .sample import ImageSample, TextImageSample, TextSample


class TripletMiningEmbeddingsIndex:
    """
    A class that constructs and manages an embeddings index for triplet mining.

    This class allows for efficient nearest neighbor search in an embeddings space
    to find positive and negative samples for triplet mining.

    Args:
        examples (List[Union[TextSample, ImageSample, TextImageSample]]):
            List of example samples for constructing the index.
        embeddings (np.ndarray):
            An array of embeddings corresponding to the examples.
        gpu (bool, optional):
            Flag indicating whether to use GPU resources for the index (default is True).
        normalize_l2 (bool, optional):
            Flag indicating whether to normalize the embeddings to unit L2 norm (default is True).

    Attributes:
        examples (List[Union[TextSample, ImageSample, TextImageSample]]):
            List of example samples used for constructing the index.
        db_vectors (np.ndarray):
            Array of embeddings for the index.
        db_ids (np.ndarray):
            Array of IDs corresponding to the embeddings.
        dim (int):
            Dimensionality of the embeddings.
        normalize_l2 (bool):
            Flag indicating whether the embeddings are L2 normalized.
        gpu (bool):
            Flag indicating whether GPU resources are used.
        index (faiss.Index):
            Faiss index for nearest neighbor search.
    """

    def __init__(
        self,
        examples: List[Union[TextSample, ImageSample, TextImageSample]],
        embeddings: np.ndarray,
        gpu: bool = True,
        normalize_l2: bool = True,
    ) -> None:
        """
        Initialize the TripletMiningEmbeddingsIndex.

        Args:
            examples (List[Union[TextSample, ImageSample, TextImageSample]]):
                List of example samples for constructing the index.
            embeddings (np.ndarray):
                An array of embeddings corresponding to the examples.
            gpu (bool, optional):
                Flag indicating whether to use GPU resources for the index (default is True).
            normalize_l2 (bool, optional):
                Flag indicating whether to normalize the embeddings to unit L2 norm (default is True).
        """
        self.examples = examples
        self.db_vectors = embeddings.copy().astype(np.float32)
        self.db_ids = np.array(list(range(len(embeddings)))).astype(np.int64)
        self.dim = len(self.db_vectors[0])
        self.normalize_l2 = normalize_l2
        self.gpu = gpu
        if self.normalize_l2:
            faiss.normalize_L2(self.db_vectors)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIDMap(self.index)
        self.index.add_with_ids(self.db_vectors, self.db_ids)
        if self.gpu:
            gpu_res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(gpu_res, 0, self.index)

    def search_pos_and_neg_samples(
        self, array_of_queries: np.ndarray, sample: [TextSample, ImageSample, TextImageSample], k: int = 2048
    ) -> Tuple[
        List[Union[TextSample, ImageSample, TextImageSample]],
        List[Union[TextSample, ImageSample, TextImageSample]],
        List[Union[TextSample, ImageSample, TextImageSample]],
    ]:
        """
        Perform a nearest neighbor search to find positive and negative samples for triplet mining.

        Args:
            array_of_queries (np.ndarray):
                Array of query embeddings for which to find nearest neighbors.
            sample (TextSample or ImageSample or TextImageSample):
                The target sample for which to find positive and negative samples.
            k (int, optional):
                Number of nearest neighbors to retrieve (default is 2048).

        Returns:
            Tuple of three lists containing positive, negative, and remaining samples:
                - pos_samples (List[Union[TextSample, ImageSample, TextImageSample]]):
                    List of positive samples that belong to the same class as the target sample.
                - neg_samples (List[Union[TextSample, ImageSample, TextImageSample]]):
                    List of negative samples that belong to different classes than the target sample.
        """
        search_query = array_of_queries.copy().astype(np.float32)
        if self.normalize_l2:
            faiss.normalize_L2(search_query)
        scores, ids = self.index.search(search_query, min(k, len(self.db_ids)))
        pos_samples = []
        neg_samples = []
        for s, i in zip(scores.squeeze().tolist(), ids.squeeze().tolist()):
            if s >= 1.0:
                continue
            this_similar_sample = self.examples[i]
            if this_similar_sample.label == sample.label:
                pos_samples.append(this_similar_sample)
            else:
                neg_samples.append(this_similar_sample)
        return pos_samples, neg_samples

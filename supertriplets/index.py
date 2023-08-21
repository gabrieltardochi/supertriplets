from typing import List, Tuple, Union

import faiss
import numpy as np

from .sample import ImageSample, TextImageSample, TextSample


class TripletMiningEmbeddingsIndex:
    def __init__(
        self,
        examples: List[Union[TextSample, ImageSample, TextImageSample]],
        embeddings: np.ndarray,
        gpu: bool = True,
        normalize_l2: bool = True,
    ) -> None:
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
        search_query = array_of_queries.copy().astype(np.float32)
        if self.normalize_l2:
            faiss.normalize_L2(search_query)
        scores, ids = self.index.search(search_query, k)
        pos_samples = []
        neg_samples = []
        for s, i in zip(scores.squeeze().tolist(), ids.squeeze().tolist()):
            if s >= 1.0:
                continue
            this_similar_sample = self.samples[i]
            if this_similar_sample.label == sample.label:
                pos_samples.append(this_similar_sample)
            else:
                neg_samples.append(this_similar_sample)
        return pos_samples, neg_samples

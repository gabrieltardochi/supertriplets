import warnings

import numpy as np
import torch
from sklearn.metrics.pairwise import (paired_cosine_distances,
                                      paired_euclidean_distances,
                                      paired_manhattan_distances)
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import SampleEncodingDataset
from .index import TripletMiningEmbeddingsIndex
from .models import load_pretrained_model
from .sample import ImageSample, TextImageSample, TextSample


class TripletEmbeddingsEvaluator:
    def __init__(
        self,
        calculate_by_cosine=True,
        calculate_by_manhattan=True,
        calculate_by_euclidean=True,
    ) -> None:
        self.calculate_by_cosine = calculate_by_cosine
        self.calculate_by_manhattan = calculate_by_manhattan
        self.calculate_by_euclidean = calculate_by_euclidean

    def __call__(
        self,
        embeddings_anchors: np.ndarray,
        embeddings_positives: np.ndarray,
        embeddings_negatives: np.ndarray,
    ) -> dict:
        metrics = {}

        if self.calculate_by_cosine:
            pos_cosine_distance = paired_cosine_distances(
                embeddings_anchors, embeddings_positives
            )
            neg_cosine_distances = paired_cosine_distances(
                embeddings_anchors, embeddings_negatives
            )
            metrics["accuracy_cosine"] = (
                pos_cosine_distance < neg_cosine_distances
            ) / len(pos_cosine_distance)

        if self.calculate_by_manhattan:
            pos_manhattan_distance = paired_manhattan_distances(
                embeddings_anchors, embeddings_positives
            )
            neg_manhattan_distances = paired_manhattan_distances(
                embeddings_anchors, embeddings_negatives
            )
            metrics["accuracy_manhattan"] = (
                pos_manhattan_distance < neg_manhattan_distances
            ) / len(pos_manhattan_distance)

        if self.calculate_by_euclidean:
            pos_euclidean_distance = paired_euclidean_distances(
                embeddings_anchors, embeddings_positives
            )
            neg_euclidean_distances = paired_euclidean_distances(
                embeddings_anchors, embeddings_negatives
            )
            metrics["accuracy_euclidean"] = (
                pos_euclidean_distance < neg_euclidean_distances
            ) / len(pos_euclidean_distance)

        return metrics


class HardTripletsMiner:
    def __init__(
        self,
        examples: list[TextSample, ImageSample, TextImageSample],
        modality="text-multilingual",
        batch_size=32,
        device=None,
        use_gpu_powered_index_if_available=True,
    ) -> None:
        self.examples = examples
        self.device = (
            device
            if device is not None
            else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.use_gpu_powered_index = (
            True
            if torch.cuda.is_available() and use_gpu_powered_index_if_available
            else False
        )
        self.modality = modality
        self.batch_size = batch_size
        self.model = None
        self.pretrained_model_name = None
        self.dataset = None
        self.dataloader = None
        self.embeddings = None
        self.index = None
        self.anchors = []
        self.positives = []
        self.negatives = []
        self.count_samples_without_pairs = 0
        self.already_mined_triplets = False

    def _init_model(self):
        if self.modality == "text_english":
            self.pretrained_model_name = "STAllEnglishMiniLML12V2Encoder"
            self.model = load_pretrained_model(model_name=self.pretrained_model_name)
        elif self.modality == "text_multilingual":
            self.pretrained_model_name = "STParaphraseMultilingualMiniLML12V2Encoder"
            self.model = load_pretrained_model(model_name=self.pretrained_model_name)
        elif self.modality == "text_english-image":
            self.pretrained_model_name = "CLIPViTB32EnglishEncoder"
            self.model = load_pretrained_model(model_name=self.pretrained_model_name)
        elif self.modality == "text_multilingual-image":
            self.pretrained_model_name = "CLIPViTB32MultilingualEncoder"
            self.model = load_pretrained_model(model_name=self.pretrained_model_name)
        elif self.modality == "image":
            self.pretrained_model_name = "TIMMEfficientNetB0Encoder"
            self.model = load_pretrained_model(model_name=self.pretrained_model_name)
        else:
            raise NotImplementedError(f"Modality '{self.modality}' is not implemented.")
        self.model.to(self.device)
        self.model.eval()

    def _init_dataset(self):
        self.dataset = SampleEncodingDataset(
            examples=self.examples, sample_loading_func=self.model.load_input_example
        )

    def _init_dataloader(self):
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

    def _calculate_embeddings(self):
        batch_embeddings = []
        with torch.no_grad():
            for batch in tqdm(
                self.dataloader, total=len(self.dataloader), desc="Encoding samples"
            ):
                inputs = batch["samples"]
                if "text_input" in batch:
                    text_inputs = {
                        k: v.to(self.device) for k, v in inputs["text_input"].items()
                    }
                if "image_input" in batch:
                    image_inputs = {
                        k: v.to(self.device) for k, v in inputs["image_input"].items()
                    }
                this_batch_embeddings = self.model(image_inputs, text_inputs)
                batch_embeddings.append(this_batch_embeddings.cpu())
        self.embeddings = torch.cat(batch_embeddings, dim=0).numpy()

    def _build_index(self):
        self.index = TripletMiningEmbeddingsIndex(
            samples=self.examples,
            embeddings=self.embeddings,
            gpu=self.use_gpu_powered_index,
            normalize_l2=True,
        )

    def mine_triplets(self, sample_from_topk_hardest=10):
        assert (
            not self.already_mined_triplets
        ), "Already mined triplets, you can find it on the 'positives', 'negatives' and 'anchors' attributes."
        self._init_model()
        self._init_dataset()
        self._init_dataloader()
        self._calculate_embeddings()
        self._build_index()
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
                self.anchors.append(anchor)
                self.positives.append(positive)
                self.negatives.append(negative)
            else:
                self.count_samples_without_pairs += 1
        self.already_mined_triplets = True
        if self.count_samples_without_pairs > 0:
            warnings.warn(
                f"No positive and negative pairs found for {self.count_samples_without_pairs} samples (when considered anchors)."
            )
        return self.anchors, self.positives, self.negatives

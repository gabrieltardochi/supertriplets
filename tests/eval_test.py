import numpy as np
import pytest

from supertriplets.eval import HardTripletsMiner, TripletEmbeddingsEvaluator
from supertriplets.sample import TextSample


@pytest.fixture(scope="module")
def embeddings_anchors():
    return np.array([[0.1, 0.3, 0.5], [0.7, 0.8, 0.9], [0.11, 0.31, 0.55], [0.77, 0.82, 0.945]])


@pytest.fixture(scope="module")
def embeddings_positives():
    return np.array([[0.11, 0.31, 0.51], [0.71, 0.81, 0.91], [0.12, 0.32, 0.56], [0.2, 0.3, 0.4]])


@pytest.fixture(scope="module")
def embeddings_negatives():
    return np.array([[0.2, 0.6, 0.3], [0.2, 0.3, 0.5], [0.8, 0.3, 0.8], [0.7, 0.86, 0.9]])


@pytest.fixture(scope="module")
def examples():
    return [TextSample("a", 0), TextSample("b", 0), TextSample("c", 1), TextSample("d", 1)]


@pytest.fixture(scope="module")
def embeddings():
    return np.array([[0.1, 0.3, 0.5], [0.7, 0.8, 0.9], [0.11, 0.31, 0.51], [0.71, 0.81, 0.91]])


def test_triplet_embeddings_evaluator(embeddings_anchors, embeddings_positives, embeddings_negatives):
    evaluator = TripletEmbeddingsEvaluator(
        calculate_by_cosine=True, calculate_by_manhattan=True, calculate_by_euclidean=True
    )
    metrics = evaluator.evaluate(
        embeddings_anchors=embeddings_anchors,
        embeddings_positives=embeddings_positives,
        embeddings_negatives=embeddings_negatives,
    )
    assert metrics["accuracy_cosine"] == 0.75
    assert metrics["accuracy_manhattan"] == 0.75
    assert metrics["accuracy_euclidean"] == 0.75
    assert len(metrics.keys()) == 3


def test_hard_triplet_miner(examples, embeddings):
    miner = HardTripletsMiner(use_gpu_powered_index_if_available=True)
    anchors, positives, negatives = miner.mine(
        examples=examples, embeddings=embeddings, normalize_l2=True, sample_from_topk_hardest=1
    )
    assert anchors == examples
    assert positives == [examples[i] for i in [1, 0, 3, 2]]
    assert negatives == [examples[i] for i in [2, 3, 0, 1]]

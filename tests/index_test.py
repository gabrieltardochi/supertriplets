import numpy as np
import pytest

from supertriplets.index import TripletMiningEmbeddingsIndex
from supertriplets.sample import TextSample


@pytest.fixture
def examples():
    return [TextSample("a", 0), TextSample("b", 0), TextSample("c", 1), TextSample("d", 1)]


@pytest.fixture
def embeddings():
    return np.array([[0.1, 0.3, 0.5], [0.7, 0.8, 0.9], [0.1, 0.3, 0.5], [0.7, 0.8, 0.9]])


def test_index_search_pos_and_neg_samples(examples, embeddings):
    anchor = TextSample("foo", 0)
    anchor_embedding = np.array([0.11, 0.31, 0.51])
    index = TripletMiningEmbeddingsIndex(examples=examples, embeddings=embeddings, gpu=False, normalize_l2=True)
    possible_positives, possible_negatives = index.search_pos_and_neg_samples(
        array_of_queries=np.expand_dims(anchor_embedding, axis=0),
        sample=anchor,
        k=10,
    )
    assert possible_positives[0] == examples[0]
    assert set(possible_positives) == set(examples[:2])
    assert set(possible_negatives) == set(examples[2:])
    assert possible_negatives[0] == examples[2]
    assert len(possible_positives) == 2
    assert len(possible_negatives) == 2

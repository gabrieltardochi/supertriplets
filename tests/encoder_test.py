import numpy as np
import pytest

from supertriplets.encoder import PretrainedSampleEncoder
from supertriplets.sample import TextSample


@pytest.fixture(scope="module")
def examples():
    return [
        TextSample("yada yada", 0),
        TextSample("foo bar", 0),
        TextSample("hello world", 1),
        TextSample("gotta test it", 1),
    ]


def test_valid_pretrained_sample_encoder(examples):
    encoder = PretrainedSampleEncoder(modality="text_english")
    embeddings = encoder.encode(examples=examples, device="cpu", batch_size=8)
    assert isinstance(embeddings, np.ndarray)
    assert len(embeddings) == len(examples)


def test_invalid_pretrained_sample_encoder():
    with pytest.raises(Exception):
        PretrainedSampleEncoder(modality="not a modality")

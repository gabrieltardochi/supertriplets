import pytest
import torch
from torch.utils.data import DataLoader

from supertriplets.dataset import (
    OnlineTripletsDataset,
    SampleEncodingDataset,
    StaticTripletsDataset,
)
from supertriplets.models import load_pretrained_model
from supertriplets.sample import TextSample


@pytest.fixture(scope="module")
def binary_examples():
    return [
        TextSample("i love helmets", 0),
        TextSample("gotta love testing", 0),
        TextSample("hi there!", 1),
        TextSample("hello, how are you doing?", 1),
    ]


@pytest.fixture(scope="module")
def multiclass_examples():
    return [
        TextSample("i love helmets", 0),
        TextSample("gotta love testing", 1),
        TextSample("hi there!", 2),
        TextSample("hello, how are you doing?", 3),
    ]


@pytest.fixture(scope="module")
def pretrained_model():
    model = load_pretrained_model(model_name="STAllEnglishMiniLML12V2Encoder")
    model.to("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model


@pytest.fixture(scope="module")
def examples_sample_loading_kwargs():
    return {"max_length": 20}


def test_static_triplets_dataset(binary_examples, pretrained_model, examples_sample_loading_kwargs):
    dataset = StaticTripletsDataset(
        anchor_examples=binary_examples,
        positive_examples=[binary_examples[i] for i in [1, 0, 3, 2]],
        negative_examples=[binary_examples[i] for i in [2, 3, 0, 1]],
        sample_loading_func=pretrained_model.load_input_example,
        sample_loading_kwargs=examples_sample_loading_kwargs,
    )
    first_anchor_loaded = pretrained_model.load_input_example(
        **binary_examples[0].data(), **examples_sample_loading_kwargs
    )
    first_positive_loaded = pretrained_model.load_input_example(
        **binary_examples[1].data(), **examples_sample_loading_kwargs
    )
    first_negative_loaded = pretrained_model.load_input_example(
        **binary_examples[2].data(), **examples_sample_loading_kwargs
    )
    dataset_first_triplet = dataset[0]
    assert len(dataset) == len(binary_examples)
    assert all(
        [
            torch.allclose(first_anchor_loaded["text_input"][k], v, atol=1e-6)
            for k, v in dataset_first_triplet["anchors"]["text_input"].items()
        ]
    )
    assert all(
        [
            torch.allclose(first_positive_loaded["text_input"][k], v, atol=1e-6)
            for k, v in dataset_first_triplet["positives"]["text_input"].items()
        ]
    )
    assert all(
        [
            torch.allclose(first_negative_loaded["text_input"][k], v, atol=1e-6)
            for k, v in dataset_first_triplet["negatives"]["text_input"].items()
        ]
    )


def test_online_triplets_dataset(multiclass_examples, pretrained_model, examples_sample_loading_kwargs):
    dataset = OnlineTripletsDataset(
        examples=multiclass_examples,
        in_batch_num_samples_per_label=1,
        batch_size=2,
        sample_loading_func=pretrained_model.load_input_example,
        sample_loading_kwargs=examples_sample_loading_kwargs,
    )
    existing_labels = [example.label for example in multiclass_examples]
    found_labels = []
    dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=False, drop_last=False)
    for batch in dataloader:
        found_labels.extend(batch["samples"]["label"].numpy().tolist())
    assert len(found_labels) == len(multiclass_examples)
    assert set(existing_labels) == set(found_labels)


def test_sample_encoding_dataset(binary_examples, pretrained_model, examples_sample_loading_kwargs):
    dataset = SampleEncodingDataset(
        examples=binary_examples,
        sample_loading_func=pretrained_model.load_input_example,
        sample_loading_kwargs=examples_sample_loading_kwargs,
    )
    first_example_loaded = pretrained_model.load_input_example(
        **binary_examples[0].data(), **examples_sample_loading_kwargs
    )
    dataset_first_example = dataset[0]
    assert len(dataset) == len(binary_examples)
    assert all(
        [
            torch.allclose(first_example_loaded["text_input"][k], v, atol=1e-6)
            for k, v in dataset_first_example["samples"]["text_input"].items()
        ]
    )

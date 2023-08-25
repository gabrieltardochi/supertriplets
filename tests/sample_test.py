import pytest

from supertriplets.sample import ImageSample, TextImageSample, TextSample


@pytest.fixture(params=["foo", "bar"])
def valid_text(request):
    return request.param


@pytest.fixture(params=["tests/data/cat.jpg", "tests/data/dog.jpg"])
def valid_image_path(request):
    return request.param


@pytest.fixture(params=[0, 1])
def valid_label(request):
    return request.param


@pytest.fixture(params=[None, 0])
def invalid_text(request):
    return request.param


@pytest.fixture(params=[None, "tests/data/cat.png", "tests/data/not_a_dog.jpg"])
def invalid_image_path(request):
    return request.param


@pytest.fixture(params=[None, 1.0, "0"])
def invalid_label(request):
    return request.param


def test_valid_text_sample(valid_text, valid_label):
    text_sample = TextSample(text=valid_text, label=valid_label)
    expected_text_sample_data = {"text": valid_text, "label": valid_label}
    assert text_sample.data() == expected_text_sample_data


def test_valid_text_image_sample(valid_text, valid_image_path, valid_label):
    text_image_sample = TextImageSample(text=valid_text, image_path=valid_image_path, label=valid_label)
    expected_text_image_sample_data = {"text": valid_text, "image_path": valid_image_path, "label": valid_label}
    assert text_image_sample.data() == expected_text_image_sample_data


def test_valid_image_sample(valid_image_path, valid_label):
    image_sample = ImageSample(image_path=valid_image_path, label=valid_label)
    expected_image_sample_data = {"image_path": valid_image_path, "label": valid_label}
    assert image_sample.data() == expected_image_sample_data


def test_invalid_text_sample(invalid_text, invalid_label):
    with pytest.raises(Exception):
        TextSample(text=invalid_text, label=invalid_label)


def test_invalid_text_image_sample(invalid_text, invalid_image_path, invalid_label):
    with pytest.raises(Exception):
        TextImageSample(text=invalid_text, image_path=invalid_image_path, label=invalid_label)


def test_invalid_image_sample(invalid_image_path, invalid_label):
    with pytest.raises(Exception):
        ImageSample(image_path=invalid_image_path, label=invalid_label)

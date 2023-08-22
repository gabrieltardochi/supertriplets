import pytest
from supertriplets.sample import TextSample, ImageSample, TextImageSample


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
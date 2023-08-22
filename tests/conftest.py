import pytest


@pytest.fixture(scope="session", params=["foo", "bar"])
def valid_text(request):
    return request.param

@pytest.fixture(scope="session", params=["tests/data/cat.jpg", "tests/data/dog.jpg"])
def valid_image_path(request):
    return request.param

@pytest.fixture(scope="session", params=[0, 1])
def valid_label(request):
    return request.param

@pytest.fixture(scope="session", params=[None, 0])
def invalid_text(request):
    return request.param

@pytest.fixture(scope="session", params=[None, "tests/data/cat.png", "tests/data/not_a_dog.jpg"])
def invalid_image_path(request):
    return request.param

@pytest.fixture(scope="session", params=[None, 1.0, '0'])
def invalid_label(request):
    return request.param


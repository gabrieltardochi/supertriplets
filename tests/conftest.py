import pytest
import torch

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

@pytest.fixture(scope="session")
def torch_embeddings():
    return torch.tensor([[0.1, 0.3, 0.5],
                        [0.6, 0.8, 1.0],
                        [0.8, 0.6, 0.0],
                        [0.9, 0.2, -0.5]])

@pytest.fixture(scope="session")
def torch_labels():
    return torch.tensor([0, 1, 1, 0])

@pytest.fixture(scope="session")
def torch_example_vectors():
    return torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
import torch

from supertriplets.distance import CosineDistance, EuclideanDistance


def test_cosine_distance(torch_example_vectors):
    cosine_distance = CosineDistance()
    distances = cosine_distance(torch_example_vectors, torch_example_vectors)

    expected_distances = torch.tensor(
        [
            [
                [1.1921e-07, 2.5368e-02, 4.0588e-02],
                [2.5368e-02, 0.0000e00, 1.8091e-03],
                [4.0588e-02, 1.8091e-03, 0.0000e00],
            ]
        ]
    )

    assert torch.allclose(distances, expected_distances, atol=1e-3)


def test_euclidean_distance(torch_example_vectors):
    euclidean_distance = EuclideanDistance()
    distances = euclidean_distance(torch_example_vectors, torch_example_vectors)

    expected_distances = torch.tensor(
        [[0.0, 5.19615242, 10.39230485], [5.19615242, 0.0, 5.19615242], [10.39230485, 5.19615242, 0.0]]
    )

    assert torch.allclose(distances, expected_distances, atol=1e-3)


def test_cosine_distance_normalized(torch_example_vectors):
    cosine_distance = CosineDistance(alredy_l2_normalized_vectors=True)
    distances = cosine_distance(torch_example_vectors, torch_example_vectors)

    expected_distances = torch.tensor([[-13.0, -31.0, -49.0], [-31.0, -76.0, -121.0], [-49.0, -121.0, -193.0]])

    assert torch.allclose(distances, expected_distances, atol=1e-3)


def test_euclidean_distance_squared(torch_example_vectors):
    euclidean_distance = EuclideanDistance(squared=True)
    distances = euclidean_distance(torch_example_vectors, torch_example_vectors)

    expected_distances = torch.tensor([[0.0, 27.0, 108.0], [27.0, 0.0, 27.0], [108.0, 27.0, 0.0]])

    assert torch.allclose(distances, expected_distances, atol=1e-3)


def test_euclidean_distance_non_negative(torch_example_vectors):
    euclidean_distance = EuclideanDistance()
    distances = euclidean_distance(torch_example_vectors, torch_example_vectors)
    assert torch.all(distances >= 0)


def test_cosine_distance_result_range(torch_example_vectors):
    cosine_distance = CosineDistance()
    distances = cosine_distance(torch_example_vectors, torch_example_vectors)
    assert torch.all(distances >= -1) and torch.all(distances <= 1)

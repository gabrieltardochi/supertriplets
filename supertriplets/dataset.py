from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset

from .sample import ImageSample, TextImageSample, TextSample


class StaticTripletsDataset(Dataset):
    """
    A dataset class for handling static triplets of anchor, positive, and negative examples.

    Args:
        anchor_examples (List[Union[TextSample, ImageSample, TextImageSample]]): List of anchor examples.
        positive_examples (List[Union[TextSample, ImageSample, TextImageSample]]): List of positive examples.
        negative_examples (List[Union[TextSample, ImageSample, TextImageSample]]): List of negative examples.
        sample_loading_func (Callable): A function for loading samples.
        sample_loading_kwargs (Dict, optional): Additional keyword arguments to pass to the sample loading function.
            Defaults to {}.

    Methods:
        __len__(self) -> int:
            Get the number of triplets in the dataset.

        __getitem__(self, idx) -> Dict[str, Dict[str, Tensor]]:
            Get a triplet of anchor, positive, and negative examples with loaded data.

    Note:
        This class is designed for loading existing triplets for inference purposes.
    """

    def __init__(
        self,
        anchor_examples: List[Union[TextSample, ImageSample, TextImageSample]],
        positive_examples: List[Union[TextSample, ImageSample, TextImageSample]],
        negative_examples: List[Union[TextSample, ImageSample, TextImageSample]],
        sample_loading_func: Callable,
        sample_loading_kwargs: Dict = {},
    ) -> None:
        """
        Initialize the StaticTripletsDataset.

        Args:
            anchor_examples (List[Union[TextSample, ImageSample, TextImageSample]]): List of anchor examples.
            positive_examples (List[Union[TextSample, ImageSample, TextImageSample]]): List of positive examples.
            negative_examples (List[Union[TextSample, ImageSample, TextImageSample]]): List of negative examples.
            sample_loading_func (Callable): A function for loading samples.
            sample_loading_kwargs (Dict, optional): Additional keyword arguments to pass to the sample loading function.
                Defaults to {}.
        """
        super().__init__()
        self.anchor_examples = anchor_examples
        self.positive_examples = positive_examples
        self.negative_examples = negative_examples
        self.sample_loading_func = sample_loading_func
        self.sample_loading_kwargs = sample_loading_kwargs

    def __len__(self) -> int:
        """
        Get the number of triplets in the dataset.

        Returns:
            int: The number of triplets in the dataset.
        """
        return len(self.anchor_examples)

    def __getitem__(self, idx) -> Dict[str, Dict[str, Tensor]]:
        """
        Get a triplet of anchor, positive, and negative examples with loaded data.

        Args:
            idx (int): Index of the triplet.

        Returns:
            Dict[str, Dict[str, Tensor]]: A dictionary containing anchor, positive, and negative examples with loaded data.
        """
        anchor = self.anchor_examples[idx]
        positive = self.positive_examples[idx]
        negative = self.negative_examples[idx]

        item = {
            "anchors": self.sample_loading_func(**anchor.data(), **self.sample_loading_kwargs),
            "positives": self.sample_loading_func(**positive.data(), **self.sample_loading_kwargs),
            "negatives": self.sample_loading_func(**negative.data(), **self.sample_loading_kwargs),
        }
        return item


class OnlineTripletsDataset(IterableDataset):
    """An torch.utils.data.IterableDataset implementation for generating online triplets.

    Args:
        examples (List[Union[TextSample, ImageSample, TextImageSample]]):
            List of examples containing samples with labels.
        in_batch_num_samples_per_label (int):
            Number of samples per label in each batch.
        batch_size (int):
            Total number of samples in each batch.
        sample_loading_func (Callable):
            A callable function to load samples from the dataset.
        sample_loading_kwargs (Dict, optional):
            Additional keyword arguments for the sample loading function.
            Defaults to an empty dictionary.

    Attributes:
        examples (List[Union[TextSample, ImageSample, TextImageSample]]):
            List of examples containing samples with labels.
        in_batch_num_samples_per_label (int):
            Number of samples per label in each batch.
        sample_loading_func (Callable):
            A callable function to load samples from the dataset.
        sample_loading_kwargs (Dict):
            Additional keyword arguments for the sample loading function.
        grouped_inputs (List[Union[TextSample, ImageSample, TextImageSample]]):
            List of samples grouped by label.
        groups_right_border (List[int]):
            Right border indices for each group in grouped_inputs.
        label_range (np.ndarray):
            Shuffled range of labels used for iteration.

    Methods:
        _group_examples_by_label(self) -> Tuple[List[Union[...]], List[int]]:
            Groups examples by label and calculates right borders of groups.
        _generate_shuffled_label_range(self) -> np.ndarray:
            Generates a shuffled range of labels.
        __iter__(self) -> Optional[Dict[str, Dict[str, Tensor]]]:
            Generates batches of samples with online triplet selection.
        __len__(self) -> int:
            Returns the total number of samples in the dataset.

    Note:
        This class is designed for generating online triplets for training purposes.
    """

    def __init__(
        self,
        examples: List[Union[TextSample, ImageSample, TextImageSample]],
        in_batch_num_samples_per_label: int,
        batch_size: int,
        sample_loading_func: Callable,
        sample_loading_kwargs: Dict = {},
    ) -> None:
        """
        Initialize the OnlineTripletsDataset.

        Args:
            examples (List[Union[TextSample, ImageSample, TextImageSample]]):
                List of examples containing samples with labels.
            in_batch_num_samples_per_label (int):
                Number of samples per label in each batch. Labels with fewer samples
                than this value will be discarded.
            batch_size (int):
                Total number of samples in each batch. It must be a multiple of
                'in_batch_num_samples_per_label', ensuring each batch contains an
                equal number of samples for each label.
            sample_loading_func (Callable):
                A callable function to load samples from the dataset.
            sample_loading_kwargs (Dict, optional):
                Additional keyword arguments for the sample loading function.
                Defaults to an empty dictionary.
        """
        super().__init__()
        assert (
            batch_size % in_batch_num_samples_per_label
        ) == 0, "'batch_size' should be a multiple of 'in_batch_num_samples_per_label'"
        self.examples = examples
        self.in_batch_num_samples_per_label = in_batch_num_samples_per_label
        self.sample_loading_func = sample_loading_func
        self.sample_loading_kwargs = sample_loading_kwargs
        self.grouped_inputs, self.groups_right_border = self._group_examples_by_label()
        self.label_range = self._generate_shuffled_label_range()

    def _group_examples_by_label(self) -> Tuple[List[Union[TextSample, ImageSample, TextImageSample]], List[int]]:
        """
        Groups examples by label and calculates right borders of groups.

        Returns:
            Tuple containing grouped examples and right borders.
        """
        label2ex = {}
        for example in self.examples:
            if example.label not in label2ex:
                label2ex[example.label] = []
            label2ex[example.label].append(example)

        grouped_inputs = []
        groups_right_border = []
        num_labels = 0

        for label, label_examples in label2ex.items():
            if len(label_examples) >= self.in_batch_num_samples_per_label:
                grouped_inputs.extend(label_examples)
                groups_right_border.append(len(grouped_inputs))
                num_labels += 1

        return grouped_inputs, groups_right_border

    def _generate_shuffled_label_range(self) -> np.ndarray:
        """
        Generates a shuffled range of labels.

        Returns:
            np.ndarray: Shuffled range of labels.
        """
        label_range = np.arange(len(self.groups_right_border))
        np.random.shuffle(label_range)
        return label_range

    def __iter__(self) -> Optional[Dict[str, Dict[str, Tensor]]]:
        """
        Generates batches of samples with online triplet selection.

        Yields:
            Optional[Dict[str, Dict[str, Tensor]]]:
                A dictionary containing samples in the 'samples' key.
        """
        label_idx = 0
        count = 0
        already_seen = {}

        while count < len(self.grouped_inputs):
            label = self.label_range[label_idx]
            if label not in already_seen:
                already_seen[label] = set()

            left_border = 0 if label == 0 else self.groups_right_border[label - 1]
            right_border = self.groups_right_border[label]

            selection = [i for i in np.arange(left_border, right_border) if i not in already_seen[label]]

            if len(selection) >= self.in_batch_num_samples_per_label:
                for element_idx in np.random.choice(selection, self.in_batch_num_samples_per_label, replace=False):
                    count += 1
                    already_seen[label].add(element_idx)
                    yield {
                        "samples": self.sample_loading_func(
                            **self.grouped_inputs[element_idx].data(), **self.sample_loading_kwargs
                        )
                    }

            label_idx += 1
            if label_idx >= len(self.label_range):
                label_idx = 0
                already_seen = {}
                np.random.shuffle(self.label_range)

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.grouped_inputs)


class SampleEncodingDataset(Dataset):
    """A torch.utils.data.Dataset implementation for loading and encoding samples.

    Args:
        examples (List[Union[TextSample, ImageSample, TextImageSample]]):
            List of examples containing samples to be encoded.
        sample_loading_func (Callable):
            A callable function to load and encode samples from the dataset.
        sample_loading_kwargs (Dict, optional):
            Additional keyword arguments for the sample loading function.
            Defaults to an empty dictionary.

    Attributes:
        examples (List[Union[TextSample, ImageSample, TextImageSample]]):
            List of examples containing samples to be encoded.
        sample_loading_func (Callable):
            A callable function to load and encode samples from the dataset.
        sample_loading_kwargs (Dict):
            Additional keyword arguments for the sample loading function.

    Methods:
        __len__(self) -> int:
            Returns the total number of samples in the dataset.
        __getitem__(self, idx) -> Dict[str, Dict[str, Tensor]]:
            Returns a dictionary containing the encoded sample at the given index.

    Note:
        This class is designed for loading and encoding samples for downstream tasks.
    """

    def __init__(
        self,
        examples: List[Union[TextSample, ImageSample, TextImageSample]],
        sample_loading_func: Callable,
        sample_loading_kwargs: Dict = {},
    ) -> None:
        """
        Initialize the SampleEncodingDataset.

        Args:
            examples (List[Union[TextSample, ImageSample, TextImageSample]]):
                A list of sample objects containing data for encoding.
            sample_loading_func (Callable):
                A callable function used to generate encodings from sample data.
            sample_loading_kwargs (Dict, optional):
                Additional keyword arguments to be passed to the sample loading function.
                Defaults to an empty dictionary.
        """
        super().__init__()
        self.examples = examples
        self.sample_loading_func = sample_loading_func
        self.sample_loading_kwargs = sample_loading_kwargs

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset.

        Returns:
            int: The number of examples in the dataset.
        """
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, Dict[str, Tensor]]:
        """
        Retrieves a single example from the dataset and encodes it using the specified function.

        Args:
            idx (int): Index of the example to retrieve.

        Returns:
            Dict[str, Dict[str, torch.Tensor]]: A dictionary containing the encoded sample.
        """
        sample = self.examples[idx]

        item = {
            "samples": self.sample_loading_func(**sample.data(), **self.sample_loading_kwargs),
        }
        return item

import numpy as np
from torch.utils.data import Dataset, IterableDataset

from .sample import ImageSample, TextImageSample, TextSample


class StaticTripletsDataset(Dataset):
    def __init__(
        self,
        anchor_examples: list[TextSample, ImageSample, TextImageSample],
        positive_examples: list[TextSample, ImageSample, TextImageSample],
        negative_examples: list[TextSample, ImageSample, TextImageSample],
        sample_loading_func,
        sample_loading_kwargs,
    ):
        super().__init__()
        self.anchor_examples = anchor_examples
        self.positive_examples = positive_examples
        self.negative_examples = negative_examples
        self.sample_loading_func = sample_loading_func
        self.sample_loading_kwargs = sample_loading_kwargs

    def __len__(self):
        return len(self.anchor_examples)

    def __getitem__(self, idx):
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
    def __init__(
        self,
        examples: list[TextSample, ImageSample, TextImageSample],
        in_batch_num_samples_per_label: int,
        batch_size: int,
        sample_loading_func,
        sample_loading_kwargs,
    ):
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

    def _group_examples_by_label(self):
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

    def _generate_shuffled_label_range(self):
        label_range = np.arange(len(self.groups_right_border))
        np.random.shuffle(label_range)
        return label_range

    def __iter__(self):
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

    def __len__(self):
        return len(self.grouped_inputs)


class SampleEncodingDataset(Dataset):
    def __init__(
        self,
        examples: list[TextSample, ImageSample, TextImageSample],
        sample_loading_func,
        sample_loading_kwargs,
    ):
        super().__init__()
        self.examples = examples
        self.sample_loading_func = sample_loading_func
        self.sample_loading_kwargs = sample_loading_kwargs

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        sample = self.examples[idx]

        item = {
            "samples": self.sample_loading_func(sample.data(), **self.sample_loading_kwargs),
        }
        return item

<!--- BADGES: START --->
![GitHub - License](https://img.shields.io/github/license/gabrieltardochi/supertriplets?logo=github&style=plastic)
![PyPI - Downloads](https://img.shields.io/pypi/dm/supertriplets?logo=pypi&style=plastic)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/supertriplets?logo=python&style=plastic)
![PyPI - Package Version](https://img.shields.io/pypi/v/supertriplets?logo=pypi&style=plastic)
<!--- BADGES: END --->
# SuperTriplets
SuperTriplets is a toolbox for supervised online hard triplet learning, currently supporting different kinds of data: text, image, and even text + image (multimodal).
It by no means tries to automate the training and evaluation loop for you. Instead, it provides useful PyTorch-based utilities you can couple to your existing code, making the process as easy as performing other everyday supervised learning tasks, such as classification and regression.
## Installation and Supported Versions
SuperTriplets is available on PyPI:
```console
$ pip install supertriplets
```
SuperTriplets officially supports Python 3.8+.
## Quick Start
### Evaluation
Mine hard triplets with pretrained models to construct your static testing dataset:
```python
import torch
from supertriplets.sample import TextImageSample
from supertriplets.encoder import PretrainedSampleEncoder
from supertriplets.evaluate import HardTripletsMiner
from supertriplets.dataset import StaticTripletsDataset

# ... omitted code to load the pandas.Dataframe `test_df`

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # always use cuda if available

# SuperTriplets provides very basic `sample` classes to store and manipulate datapoints
test_examples = [TextImageSample(text=text, image_path=image_path, label=label) for text, image_path, label in zip(test_df['text'], test_df['image_path'], test_df['label'])]

# Leverage general purpose pretrained models per language and data format or bring your own `test_embeddings`
pretrained_encoder = PretrainedSampleEncoder(modality="text_english-image")
test_embeddings = pretrained_encoder.encode(examples=test_examples, device=device, batch_size=32)

# Index `test_examples` using `test_embeddings` and perform nearest neighbor search to sample hard positives and hard negatives
hard_triplet_miner = HardTripletsMiner(use_gpu_powered_index_if_available=True)
test_anchors, test_positives, test_negatives = hard_triplet_miner.mine(
    examples=test_examples, embeddings=test_embeddings, normalize_l2=True, sample_from_topk_hardest=10
)

def my_sample_loading_func(text, image_path, label, *args, **kwargs):
    # ... implement your own preprocessing logic: e.g. tokenization, augmentations, etc
    # this usually contains similar logic to what you would use inside a torch.utils.data.Dataset `__get_item__` method
    # the only requirement is that it should at least have a few parameters named like its `sample` class attributes
    loaded_sample = {"text_input": prep_text, "image_input": prep_image}
    return loaded_sample

# just another subclass of torch.utils.data.Dataset
test_dataset = StaticTripletsDataset(
    anchor_examples=test_anchor_examples,
    positive_examples=test_positive_examples,
    negative_examples=test_negative_examples,
    sample_loading_func=my_sample_loading_func,
    sample_loading_kwargs={}  # you could add parameters to `my_sample_loading_func` and pass them here
)
```

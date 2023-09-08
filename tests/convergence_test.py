import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from supertriplets.dataset import OnlineTripletsDataset, StaticTripletsDataset
from supertriplets.distance import EuclideanDistance
from supertriplets.encoder import PretrainedSampleEncoder
from supertriplets.evaluate import HardTripletsMiner, TripletEmbeddingsEvaluator
from supertriplets.loss import BatchHardTripletLoss
from supertriplets.models import load_pretrained_model
from supertriplets.sample import TextImageSample
from supertriplets.utils import move_tensors_to_device


def prepare_tinymmimdb_split(split):
    df = pd.read_csv("tests/data/tinymmimdb/data.csv")
    df = df[df["split"] == split].reset_index(drop=True)
    df["text"] = df["plot outline"].astype(str)
    df["image_path"] = "tests/data/tinymmimdb/images/" + df["image_path"].astype(str)
    df["label"] = df["genre_id"].astype(int)
    df = df[["text", "image_path", "label"]]
    return df


@pytest.fixture(scope="module")
def tinymmimdb_train():
    return prepare_tinymmimdb_split("train")


@pytest.fixture(scope="module")
def tinymmimdb_dev():
    return prepare_tinymmimdb_split("dev")


@pytest.fixture(scope="module")
def tinymmimdb_test():
    return prepare_tinymmimdb_split("test")


def test_tinymmimdb_convergence(tinymmimdb_train, tinymmimdb_dev, tinymmimdb_test):
    device = "cuda:0"
    train_examples = [
        TextImageSample(text=text, image_path=image_path, label=label)
        for text, image_path, label in zip(
            tinymmimdb_train["text"], tinymmimdb_train["image_path"], tinymmimdb_train["label"]
        )
    ]
    dev_examples = [
        TextImageSample(text=text, image_path=image_path, label=label)
        for text, image_path, label in zip(
            tinymmimdb_dev["text"], tinymmimdb_dev["image_path"], tinymmimdb_dev["label"]
        )
    ]
    test_examples = [
        TextImageSample(text=text, image_path=image_path, label=label)
        for text, image_path, label in zip(
            tinymmimdb_test["text"], tinymmimdb_test["image_path"], tinymmimdb_test["label"]
        )
    ]

    pretrained_encoder = PretrainedSampleEncoder(modality="text_english-image")
    dev_embeddings = pretrained_encoder.encode(examples=dev_examples, device=device, batch_size=32)
    test_embeddings = pretrained_encoder.encode(examples=test_examples, device=device, batch_size=32)
    del pretrained_encoder

    hard_triplet_miner = HardTripletsMiner(use_gpu_powered_index_if_available=True)
    dev_anchor_examples, dev_positive_examples, dev_negative_examples = hard_triplet_miner.mine(
        examples=dev_examples, embeddings=dev_embeddings, normalize_l2=True, sample_from_topk_hardest=10
    )
    test_anchor_examples, test_positive_examples, test_negative_examples = hard_triplet_miner.mine(
        examples=test_examples, embeddings=test_embeddings, normalize_l2=True, sample_from_topk_hardest=10
    )
    del hard_triplet_miner

    model = load_pretrained_model(model_name="CLIPViTB32EnglishEncoder")
    model.to(device)
    model.eval()

    trainset = OnlineTripletsDataset(
        examples=train_examples,
        in_batch_num_samples_per_label=2,
        batch_size=32,
        sample_loading_func=model.load_input_example,
    )
    devset = StaticTripletsDataset(
        anchor_examples=dev_anchor_examples,
        positive_examples=dev_positive_examples,
        negative_examples=dev_negative_examples,
        sample_loading_func=model.load_input_example,
    )
    testset = StaticTripletsDataset(
        anchor_examples=test_anchor_examples,
        positive_examples=test_positive_examples,
        negative_examples=test_negative_examples,
        sample_loading_func=model.load_input_example,
    )

    trainloader = DataLoader(dataset=trainset, batch_size=32, num_workers=0, drop_last=True)
    devloader = DataLoader(dataset=devset, batch_size=32, shuffle=False, num_workers=0, drop_last=False)
    testloader = DataLoader(dataset=testset, batch_size=32, shuffle=False, num_workers=0, drop_last=False)

    def get_triplet_embeddings(dataloader, model, device):
        model.eval()
        embeddings = {"anchors": [], "positives": [], "negatives": []}
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader)):
                for input_type in ["anchors", "positives", "negatives"]:
                    inputs = {k: v for k, v in batch[input_type].items() if k != "label"}
                    inputs = move_tensors_to_device(obj=inputs, device=device)
                    batch_embeddings = model(**inputs).cpu()
                    embeddings[input_type].append(batch_embeddings)
        embeddings = {k: torch.cat(v, dim=0).numpy() for k, v in embeddings.items()}
        return embeddings

    triplet_embeddings_evaluator = TripletEmbeddingsEvaluator(
        calculate_by_cosine=True, calculate_by_manhattan=True, calculate_by_euclidean=True
    )
    dev_triplet_embeddings = get_triplet_embeddings(dataloader=devloader, model=model, device=device)
    test_triplet_embeddings = get_triplet_embeddings(dataloader=testloader, model=model, device=device)

    dev_start_accuracy = triplet_embeddings_evaluator.evaluate(
        embeddings_anchors=dev_triplet_embeddings["anchors"],
        embeddings_positives=dev_triplet_embeddings["positives"],
        embeddings_negatives=dev_triplet_embeddings["negatives"],
    )
    test_start_accuracy = triplet_embeddings_evaluator.evaluate(
        embeddings_anchors=test_triplet_embeddings["anchors"],
        embeddings_positives=test_triplet_embeddings["positives"],
        embeddings_negatives=test_triplet_embeddings["negatives"],
    )

    num_epochs = 1
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=2e-5)
    criterion = BatchHardTripletLoss(distance=EuclideanDistance(squared=False), margin=5)

    for epoch in range(1, num_epochs + 1):
        model.train()
        for batch in tqdm(trainloader, total=len(trainloader), desc=f"Epoch {epoch}"):
            data = batch["samples"]
            labels = move_tensors_to_device(obj=data.pop("label"), device=device)
            inputs = move_tensors_to_device(obj=data, device=device)

            optimizer.zero_grad()

            embeddings = model(**inputs)
            loss = criterion(embeddings=embeddings, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

    dev_triplet_embeddings = get_triplet_embeddings(dataloader=devloader, model=model, device=device)
    test_triplet_embeddings = get_triplet_embeddings(dataloader=testloader, model=model, device=device)
    dev_final_accuracy = triplet_embeddings_evaluator.evaluate(
        embeddings_anchors=dev_triplet_embeddings["anchors"],
        embeddings_positives=dev_triplet_embeddings["positives"],
        embeddings_negatives=dev_triplet_embeddings["negatives"],
    )
    test_final_accuracy = triplet_embeddings_evaluator.evaluate(
        embeddings_anchors=test_triplet_embeddings["anchors"],
        embeddings_positives=test_triplet_embeddings["positives"],
        embeddings_negatives=test_triplet_embeddings["negatives"],
    )

    assert all({dev_final_accuracy[k] > v for k, v in dev_start_accuracy.items()})
    assert all({test_final_accuracy[k] > v for k, v in test_start_accuracy.items()})

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


class STParaphraseMultilingualMiniLML12V2Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hf_text_model_name = (
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.text_config = AutoConfig.from_pretrained(self.hf_text_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_text_model_name, config=self.text_config
        )
        self.text_model = AutoModel.from_pretrained(
            self.hf_text_model_name, config=self.text_config
        )

    def load_input_example(
        self,
        text,
        label,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=None,
    ):
        text_input = self.tokenizer(
            text,
            return_tensors=return_tensors,
            truncation=truncation,
            padding=padding,
            max_length=max_length,
        )
        text_input = {k: v.squeeze() for k, v in text_input.items()}
        label = torch.tensor(label)
        return {"text_input": text_input, "label": label}

    def forward(self, text_inputs):
        model_output = self.text_model(**text_inputs)
        text_features = self.mean_pooling(model_output, text_inputs["attention_mask"])
        return text_features.squeeze()

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )


class STAllEnglishMiniLML12V2Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hf_text_model_name = "sentence-transformers/all-MiniLM-L12-v2"
        self.text_config = AutoConfig.from_pretrained(self.hf_text_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_text_model_name, config=self.text_config
        )
        self.text_model = AutoModel.from_pretrained(
            self.hf_text_model_name, config=self.text_config
        )

    def load_input_example(
        self,
        text,
        label,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=None,
    ):
        text_input = self.tokenizer(
            text,
            return_tensors=return_tensors,
            truncation=truncation,
            padding=padding,
            max_length=max_length,
        )
        text_input = {k: v.squeeze() for k, v in text_input.items()}
        label = torch.tensor(label)
        return {"text_input": text_input, "label": label}

    def forward(self, text_inputs):
        model_output = self.text_model(**text_inputs)
        text_features = self.mean_pooling(model_output, text_inputs["attention_mask"])
        return text_features.squeeze()

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

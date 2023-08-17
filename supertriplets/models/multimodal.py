import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from torch import nn
from transformers import (AutoConfig, AutoTokenizer, CLIPImageProcessor,
                          CLIPModel, CLIPTokenizer)


class CLIPViTB32MultilingualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hf_image_model_name = "openai/clip-vit-base-patch32"
        self.hf_text_model_name = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

        self.image_config = AutoConfig.from_pretrained(self.hf_image_model_name)
        self.text_config = AutoConfig.from_pretrained(self.hf_text_model_name)

        self.processor = CLIPImageProcessor.from_pretrained(
            self.hf_image_model_name, config=self.image_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_text_model_name, config=self.text_config
        )

        clip_image = CLIPModel.from_pretrained(
            self.hf_image_model_name, config=self.image_config
        )
        clip_text = SentenceTransformer(self.hf_text_model_name)

        self.vision_model = clip_image.vision_model
        self.visual_projection = clip_image.visual_projection
        self.text_model = clip_text[0].auto_model
        self.textual_projection = clip_text[2].linear

    def load_input_example(
        self,
        text,
        image_path,
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
        image_input = self.processor(
            Image.open(image_path), return_tensors=return_tensors
        )
        image_input["pixel_values"] = image_input["pixel_values"].squeeze()
        label = torch.tensor(label)
        return {"text_input": text_input, "image_input": image_input, "label": label}

    def forward(self, image_inputs, text_inputs):
        image_features = self.get_image_features(image_inputs)
        text_features = self.get_text_features(text_inputs)
        multimodal_features = image_features + text_features
        return multimodal_features.squeeze()

    def get_image_features(self, image_inputs):
        vision_pooled_output = self.vision_model(**image_inputs)[1]  # pooled_output
        image_features = self.visual_projection(vision_pooled_output)
        return image_features

    def get_text_features(self, text_inputs):
        text_token_embeddings = self.text_model(**text_inputs)[
            0
        ]  # all token embeddings
        text_pooled_output = self.text_mean_pooling(
            text_token_embeddings, text_inputs["attention_mask"]
        )
        text_features = self.textual_projection(text_pooled_output)
        return text_features

    @staticmethod
    def text_mean_pooling(token_embeddings, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )


class CLIPViTB32EnglishEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hf_image_and_text_model_name = "openai/clip-vit-base-patch32"

        self.image_and_text_config = AutoConfig.from_pretrained(
            self.hf_image_and_text_model_name
        )

        self.processor = CLIPImageProcessor.from_pretrained(
            self.hf_image_and_text_model_name, config=self.image_and_text_config
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.hf_image_and_text_model_name, config=self.image_and_text_config
        )

        clip_image_and_text = CLIPModel.from_pretrained(
            self.hf_image_and_text_model_name, config=self.image_and_text_config
        )

        self.vision_model = clip_image_and_text.vision_model
        self.visual_projection = clip_image_and_text.visual_projection
        self.text_model = clip_image_and_text.text_model
        self.textual_projection = clip_image_and_text.text_projection

    def load_input_example(
        self,
        text,
        image_path,
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
        image_input = self.processor(
            Image.open(image_path), return_tensors=return_tensors
        )
        image_input["pixel_values"] = image_input["pixel_values"].squeeze()
        label = torch.tensor(label)
        return {"text_input": text_input, "image_input": image_input, "label": label}

    def forward(self, image_inputs, text_inputs):
        image_features = self.get_image_features(image_inputs)
        text_features = self.get_text_features(text_inputs)
        multimodal_features = image_features + text_features
        return multimodal_features.squeeze()

    def get_image_features(self, image_inputs):
        vision_pooled_output = self.vision_model(**image_inputs)[1]  # pooled_output
        image_features = self.visual_projection(vision_pooled_output)
        return image_features

    def get_text_features(self, text_inputs):
        text_token_embeddings = self.text_model(**text_inputs)[
            0
        ]  # all token embeddings
        text_pooled_output = self.text_mean_pooling(
            text_token_embeddings, text_inputs["attention_mask"]
        )
        text_features = self.textual_projection(text_pooled_output)
        return text_features

    @staticmethod
    def text_mean_pooling(token_embeddings, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

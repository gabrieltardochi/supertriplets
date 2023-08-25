from typing import Dict, List

import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor


class TIMMResNet18Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.timm_image_model_name = "resnet_18"
        self.image_model = timm.create_model(self.timm_image_model_name, pretrained=True, num_classes=0)

    def load_input_example(
        self,
        image_path: str,
        label: int,
        resize_hw: int = 224,
        normalize_mean: List[float] = [0.485, 0.456, 0.406],
        normalize_std: List[float] = [0.229, 0.224, 0.225],
    ) -> Dict[str, Tensor]:
        image = Image.open(image_path).convert("RGB")
        processor = transforms.Compose(
            [
                transforms.Resize(resize_hw),
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize_mean, std=normalize_std),
            ]
        )
        image_input = processor(image).unsqueeze(0)
        label = torch.tensor(label)
        return {"image_input": image_input, "label": label}

    def forward(self, image_input: Tensor) -> Tensor:
        image_features = self.model(image_input)
        return image_features.squeeze()


class TIMMEfficientNetB0Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.timm_image_model_name = "efficientnet_b0"
        self.image_model = timm.create_model(self.timm_image_model_name, pretrained=True, num_classes=0)

    def load_input_example(
        self,
        image_path: str,
        label: int,
        resize_hw: int = 224,
        normalize_mean: List[float] = [0.485, 0.456, 0.406],
        normalize_std: List[float] = [0.229, 0.224, 0.225],
    ) -> Dict[str, Tensor]:
        image = Image.open(image_path).convert("RGB")
        processor = transforms.Compose(
            [
                transforms.Resize(resize_hw),
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize_mean, std=normalize_std),
            ]
        )
        image_input = processor(image).unsqueeze(0)
        label = torch.tensor(label)
        return {"image_input": image_input, "label": label}

    def forward(self, image_input: Tensor) -> Tensor:
        image_features = self.model(image_input)
        return image_features.squeeze()

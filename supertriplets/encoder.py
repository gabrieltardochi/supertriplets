from typing import List, Literal, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import SampleEncodingDataset
from .models import load_pretrained_model
from .sample import ImageSample, TextImageSample, TextSample


class PretrainedSampleEncoder:
    """
    A class for encoding text and image samples using various pretrained models.

    Args:
        modality (Literal["text_english", "text_multilingual", "text_english-image", "text_multilingual-image", "image"]):
            The modality to specify the type of samples and model to use for encoding.

    Attributes:
        modality (str): The modality specifying the encoding type.
        model (nn.Module): The pretrained model for encoding samples.

    Methods:
        _get_model(self) -> nn.Module:
            Private method to load the appropriate pretrained model based on the modality.

        _get_dataloader(self, examples: List[Union[TextSample, ImageSample, TextImageSample]], batch_size: int = 32) -> DataLoader:
            Private method to create a data loader for the provided examples.

        _calculate_embeddings(self, dataloader: DataLoader, device: Union[str, torch.device]) -> np.ndarray:
            Private method to calculate embeddings for the provided samples using the pretrained model.

        encode(self, examples: List[Union[TextSample, ImageSample, TextImageSample]], device: Union[str, torch.device], batch_size: int = 32) -> np.ndarray:
            Encodes a list of samples into embeddings using the pretrained model.
    """

    def __init__(
        self,
        modality: Literal[
            "text_english", "text_multilingual", "text_english-image", "text_multilingual-image", "image"
        ],
    ) -> None:
        """
        Initialize the PretrainedSampleEncoder.

        Args:
            modality (Literal["text_english", "text_multilingual", "text_english-image", "text_multilingual-image", "image"]):
                The modality to specify the type of samples and model to use for encoding.
        """
        self.modality = modality
        self.model = self._get_model()

    def _get_model(self) -> nn.Module:
        """
        Load and return the appropriate pretrained model based on the specified modality.

        Returns:
            nn.Module: The pretrained model for encoding samples.
        """
        if self.modality == "text_english":
            return load_pretrained_model(model_name="STAllEnglishMiniLML12V2Encoder")
        elif self.modality == "text_multilingual":
            return load_pretrained_model(model_name="STParaphraseMultilingualMiniLML12V2Encoder")
        elif self.modality == "text_english-image":
            return load_pretrained_model(model_name="CLIPViTB32EnglishEncoder")
        elif self.modality == "text_multilingual-image":
            return load_pretrained_model(model_name="CLIPViTB32MultilingualEncoder")
        elif self.modality == "image":
            return load_pretrained_model(model_name="TIMMEfficientNetB0Encoder")
        else:
            raise NotImplementedError(f"Model loading for modality '{self.modality}' is not implemented.")

    def _get_dataloader(
        self, examples: List[Union[TextSample, ImageSample, TextImageSample]], batch_size: int = 32
    ) -> DataLoader:
        """
        Create a data loader for the provided examples.

        Args:
            examples (List[Union[TextSample, ImageSample, TextImageSample]]): A list of sample objects to be encoded.
            batch_size (int, optional): The batch size for the data loader. Defaults to 32.

        Returns:
            DataLoader: A data loader for the provided examples.
        """
        dataset = SampleEncodingDataset(examples=examples, sample_loading_func=self.model.load_input_example)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        return dataloader

    def _calculate_embeddings(self, dataloader: DataLoader, device: Union[str, torch.device]) -> np.ndarray:
        """
        Calculate embeddings for the provided samples using the pretrained model.

        Args:
            dataloader (DataLoader): The data loader containing the samples.
            device (Union[str, torch.device]): The device to use for calculations.

        Returns:
            np.ndarray: An array containing the calculated embeddings.
        """
        batch_embeddings = []
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader), desc="Encoding samples"):
                inputs = batch["samples"]
                del inputs["label"]
                if "text_input" in inputs:
                    inputs["text_input"] = {k: v.to(device) for k, v in inputs["text_input"].items()}
                if "image_input" in inputs:
                    inputs["image_input"] = {k: v.to(device) for k, v in inputs["image_input"].items()}
                this_batch_embeddings = self.model(**inputs)
                batch_embeddings.append(this_batch_embeddings.cpu())
        embeddings = torch.cat(batch_embeddings, dim=0).numpy()
        return embeddings

    def encode(
        self,
        examples: List[Union[TextSample, ImageSample, TextImageSample]],
        device: Union[str, torch.device],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Encode a list of samples into embeddings using the pretrained model.

        Args:
            examples (List[Union[TextSample, ImageSample, TextImageSample]]): A list of sample objects to be encoded.
            device (Union[str, torch.device]): The device to use for calculations.
            batch_size (int, optional): The batch size for encoding. Defaults to 32.

        Returns:
            np.ndarray: An array containing the encoded embeddings.
        """
        self.model.to(device)
        self.model.eval()
        dataloader = self._get_dataloader(examples=examples, batch_size=batch_size)
        embeddings = self._calculate_embeddings(dataloader=dataloader, device=device)
        return embeddings

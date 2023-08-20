import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import SampleEncodingDataset
from .models import load_pretrained_model
from .sample import ImageSample, TextImageSample, TextSample


class PretrainedSampleEncoder:
    def __init__(
        self,
        modality="text-multilingual",
    ) -> None:
        self.modality = modality
        self.model = self._get_model()

    def _get_model(self):
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

    def _get_dataloader(self, examples: list[TextSample, ImageSample, TextImageSample], batch_size: int = 32):
        dataset = SampleEncodingDataset(examples=examples, sample_loading_func=self.model.load_input_example)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        return dataloader

    def _calculate_embeddings(self, dataloader, device):
        batch_embeddings = []
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader), desc="Encoding samples"):
                inputs = batch["samples"]
                if "text_input" in batch:
                    text_inputs = {k: v.to(device) for k, v in inputs["text_input"].items()}
                if "image_input" in batch:
                    image_inputs = {k: v.to(device) for k, v in inputs["image_input"].items()}
                this_batch_embeddings = self.model(image_inputs, text_inputs)
                batch_embeddings.append(this_batch_embeddings.cpu())
        embeddings = torch.cat(batch_embeddings, dim=0).numpy()
        return embeddings

    def encode(self, examples: list[TextSample, ImageSample, TextImageSample], device, batch_size):
        self.model.to(device)
        self.model.eval()
        dataloader = self._get_dataloader(examples=examples, batch_size=batch_size)
        embeddings = self._calculate_embeddings(dataloader=dataloader, device=device)
        return embeddings

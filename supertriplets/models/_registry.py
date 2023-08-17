from ..models.language import (STAllEnglishMiniLML12V2Encoder,
                               STParaphraseMultilingualMiniLML12V2Encoder)
from ..models.multimodal import (CLIPViTB32EnglishEncoder,
                                 CLIPViTB32MultilingualEncoder)
from ..models.vision import TIMMEfficientNetB0Encoder, TIMMResNet18Encoder

MODEL_ZOO = {
    "STAllEnglishMiniLML12V2Encoder": STAllEnglishMiniLML12V2Encoder,
    "STParaphraseMultilingualMiniLML12V2Encoder": STParaphraseMultilingualMiniLML12V2Encoder,
    "TIMMResNet18Encoder": TIMMResNet18Encoder,
    "TIMMEfficientNetB0Encoder": TIMMEfficientNetB0Encoder,
    "CLIPViTB32EnglishEncoder": CLIPViTB32EnglishEncoder,
    "CLIPViTB32MultilingualEncoder": CLIPViTB32MultilingualEncoder,
}

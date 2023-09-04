from typing import Dict, Literal, Optional, Tuple

from torch import nn

from ..models._registry import MODEL_ZOO


def load_pretrained_model(
    model_name: Literal[
        "STAllEnglishMiniLML12V2Encoder",
        "STParaphraseMultilingualMiniLML12V2Encoder",
        "TIMMResNet18Encoder",
        "TIMMEfficientNetB0Encoder",
        "CLIPViTB32EnglishEncoder",
        "CLIPViTB32MultilingualEncoder",
    ],
    *model_init_args: Optional[Tuple],
    **model_init_kwargs: Optional[Dict],
) -> nn.Module:
    """
    Load a pretrained encoder model from the model registry.

    This function loads a pretrained encoder model from the specified model name and
    initializes it with the provided arguments and keyword arguments.

    Args:
        model_name (Literal): The name of the pretrained model to load. It should be one
            of the supported model names listed in the Literal type.
        *model_init_args (Optional[Tuple]): Variable-length argument list to pass to the
            model constructor.
        **model_init_kwargs (Optional[Dict]): Keyword arguments to pass to the model
            constructor.

    Returns:
        nn.Module: A pretrained encoder model instance.

    Raises:
        NotImplementedError: If the specified model name is not implemented, thus not
            available in the model registry.
    """
    if model_name in MODEL_ZOO:
        model_class = MODEL_ZOO[model_name]
        return model_class(*model_init_args, **model_init_kwargs)
    else:
        raise NotImplementedError(f"Model '{model_name}' is not implemented.")

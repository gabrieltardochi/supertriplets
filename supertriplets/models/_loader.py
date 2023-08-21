from typing import Dict, Optional, Tuple

from torch import nn

from ..models._registry import MODEL_ZOO


def load_pretrained_model(
    model_name: str, *model_init_args: Optional[Tuple], **model_init_kwargs: Optional[Dict]
) -> nn.Module:
    if model_name in MODEL_ZOO:
        model_class = MODEL_ZOO[model_name]
        return model_class(*model_init_args, **model_init_kwargs)
    else:
        raise NotImplementedError(f"Model '{model_name}' is not implemented.")

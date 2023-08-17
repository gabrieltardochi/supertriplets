from ..models._registry import MODEL_ZOO


def load_pretrained_model(model_name, *model_init_args, **model_init_kwargs):
    if model_name in MODEL_ZOO:
        model_class = MODEL_ZOO[model_name]
        return model_class(*model_init_args, **model_init_kwargs)
    else:
        raise NotImplementedError(f"Model '{model_name}' is not implemented.")

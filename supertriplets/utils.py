import torch


def move_tensors_to_device(obj, device):
    """
    Recursively moves tensors within a nested structure to the specified device.

    This function traverses through a nested structure, which can include tensors,
    dictionaries, and lists, and moves all tensors to the specified device while
    maintaining the overall structure of the input. Tensors are moved using the
    `to` method of the PyTorch tensor class.

    Args:
        obj (Union[torch.Tensor, dict, list]): The input structure containing tensors,
            dictionaries, and/or lists.
        device (torch.device or str): The target device to which tensors should be moved.
            This should be a valid PyTorch device (e.g., "cpu" or "cuda:0").

    Returns:
        Union[torch.Tensor, dict, list]: A new structure with tensors moved to the specified device,
            while preserving the original structure of dictionaries and lists.

    Raises:
        TypeError: If the input object has an unsupported type.
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_tensors_to_device(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_tensors_to_device(v, device))
        return res
    else:
        raise TypeError(f"Invalid obj type: {type(obj)}")

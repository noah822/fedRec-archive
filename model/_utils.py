import torch

def _remove_component(model, layer_name):
    setattr(model, layer_name, torch.nn.Identity())
    return model

def _infer_device(x):
    if isinstance(x, torch.Tensor):
        return x.device
    elif isinstance(x, torch.nn.Module):
        return next(x.parameters()).device

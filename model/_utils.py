import torch

def _remove_component(model, layer_name):
    setattr(model, layer_name, torch.nn.Identity())
    return model


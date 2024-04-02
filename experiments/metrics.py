import torch
import torch.nn as nn





def pairwise_query(
    batch_A: torch.Tensor,
    batch_B: torch.Tensor,
    encoder_A: nn.Module=nn.Identity(),
    encoder_B: nn.Module=nn.Identity(),
    topK: int=1,
    ground_truth: torch.Tensor=None,
    inference_device: str='cuda'
):
    rolling_chunk_size = 128
    pass




def count_model_num_params(model: nn.Module, trainable_only=False):
    return 
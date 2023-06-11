from sklearn.manifold import TSNE
import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def display_tnse(extractor: nn.Module, dataloader, num_class=10, unpack: callable=None):
    feature_bank, feature_labels = _get_feature_bank(extractor, dataloader, unpack)
    embeds = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(feature_bank)
    colors = cm.rainbow(np.linspace(0, 1))

    plt.figure(figsize=(10, 10))
    for idx, color in zip(range(num_class), colors):
        indices = np.where(feature_labels == idx)
        plt.scatter(embeds[indices, 0], embeds[indices, 1], color=color, alpha=0.1, label=f'{idx}')
    plt.legend()
    plt.show()

def _to_numpy(t: torch.tensor):
    return t.contiguous().detach().cpu().numpy()

@torch.no_grad()
def _get_feature_bank(extractor: nn.Module , dataloader, unpack: callable =None):
    feature_bank = []
    feature_labels = []
    for inputs in dataloader:
        if unpack is None:
            x, y = inputs
        else:
            x, y = unpack(inputs)
        feature = extractor(x)
        feature = F.normalize(feature, dim=-1)
        feature_bank.append(feature)
        feature_labels.append(y)
    
    feature_bank = _to_numpy(torch.cat(feature_bank, dim=0))
    feature_labels = _to_numpy(torch.cat(feature_labels, dim=0))
    
    return feature_bank, feature_labels



        
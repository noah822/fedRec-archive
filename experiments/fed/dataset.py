from torch.utils.data import Dataset, DataLoader
from typing import List
import torchvision
from fed.utils.sampling import dirichelet_sampling

from experiments.ssl.dataset import get_simclr_transform
from utils.augmentation import Augmentation
import torchvision.transforms as T



def get_default_aug():
    train_transform, test_transform = get_simclr_transform()
    train_aug = Augmentation(train_transform, n_view=2)
    test_aug = Augmentation(test_transform, n_view=2)
    return train_aug, test_aug



def _get_criterion(dataset):
    criterion = []
    for item in dataset:
        _, label = item
        criterion.append(label)
    return criterion

def _get_client_loader(datasets: List[Dataset], config):
    loaders = []
    for dataset in datasets:
        loader = DataLoader(dataset, **config)
        loaders.append(loader)
    return loaders



def prepare_client_loader(num_client, seed):
    NUM_CLIENTS = num_client
    SEED = seed
    dataset = torchvision.datasets.CIFAR10(
        './data', train=True, download=True,
        transform=lambda x: T.ToTensor()(x)
    )
    labels = _get_criterion(dataset)
    client_datasets = dirichelet_sampling(
        dataset,
        labels,
        NUM_CLIENTS,
        alpha=5,
        seed=SEED
    )

    dataloader_config = {
        'batch_size' : 32,
        'shuffle' : True
    }

    client_dataloaders = _get_client_loader(
        client_datasets,
        dataloader_config
    )

    return client_dataloaders

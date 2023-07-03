from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
import torchvision
from fed.utils.sampling import dirichelet_sampling

from experiments.ssl.dataset import get_simclr_transform
from utils.augmentation import Augmentation
import torchvision.transforms as T

from experiments.mmvae.mnist.dataset import get_MNIST_dataloader
from fed.config import STATE

from enum import Enum
import random



def get_default_aug():
    train_transform, test_transform = get_simclr_transform()
    train_aug = Augmentation(train_transform, n_view=2)
    test_aug = Augmentation(test_transform, n_view=2)
    return train_aug, test_aug



def _get_criterion(dataset):
    criterion = []
    for item in dataset:
        label = item[-1]
        criterion.append(label)
    return criterion

def _get_client_loader(datasets: List[Dataset], config):
    loaders = []
    for dataset in datasets:
        loader = DataLoader(dataset, **config)
        loaders.append(loader)
    return loaders

def _seed_everything(seed):
    random.seed(seed)


def prepare_client_loader(
        num_client,
        seed,
        dataset_config: Dict,
        dataloader_config: Dict,
        custom_state: List[STATE]=None,
        keep_intact_ratio: float=None
    ):
    NUM_CLIENTS = num_client
    SEED = seed
    _seed_everything(seed)
    # dataset = torchvision.datasets.CIFAR10(
    #     './data', train=True, download=True,
    #     transform=lambda x: T.ToTensor()(x)
    # )

    # get original multi-modal dataset
    dataset = get_MNIST_dataloader(
        **dataset_config
    )

    client_state: List[STATE] = None

    # sample modality state of each client
    if custom_state is not None:
        assert len(custom_state) == num_client
        assert keep_intact_ratio is None
        client_state = custom_state
    else: 
        # state not explicitly provided, do random sample
        # simulate missing mod scenario
        if keep_intact_ratio is None:
            client_state = [STATE.BOTH for _ in range(num_client)]
        else:
            intact_num_client = int(keep_intact_ratio * num_client)
            missing_num_client = num_client - intact_num_client

            reserved_client_state = [
                STATE.BOTH for _ in range(intact_num_client)
            ]
            missing_client_state = sample_multi_states(
                            missing_num_client,
                            [STATE.AUDIO, STATE.IMAGE],
                            SEED)
            client_state = missing_client_state + reserved_client_state
            random.shuffle(client_state)

    labels = _get_criterion(dataset)
    client_datasets = dirichelet_sampling(
        dataset,
        labels,
        NUM_CLIENTS,
        alpha=5,
        seed=SEED
    )

    # map modality to dataset return value index list
    mod2index = {
        STATE.AUDIO : [0],
        STATE.IMAGE : [1],
        STATE.BOTH : [0, 1]        
    }

    client_datasets = [
        HijackedDataset(dataset, mod2index[state]) \
        for state, dataset in zip(client_state, client_datasets)
    ]
    
    client_dataloaders = _get_client_loader(
        client_datasets,
        dataloader_config
    )

    return client_state, client_dataloaders

# only output instance at given index specified of multimodal dataset
class HijackedDataset(Dataset):
    def __init__(self, origin_dataset, index: List[int]):
        self.origin_dataset = origin_dataset
        self.index = index
    def __len__(self):
        return len(self.origin_dataset)
    def __getitem__(self, index):
        hijacked = []
        sample = self.origin_dataset.__getitem__(index)
        for i in self.index:
            hijacked.append(sample[i])

        # for compatibility, if only one modality sampled, return instance instead of list
        if len(self.index) == 1:
            return hijacked[0]
        else: # if multiple modalities sampled, return list
            return hijacked
        


def sample_multi_states(size: int, states: List[Enum], seed):
    sampled = random.choices(states, k=size)
    return sampled

import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T




def get_simclr_transform(img_size=(32, 32), s=1):
    color_jitter = T.ColorJitter(
        0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
    )
    # ImageNet statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = [
        T.RandomResizedCrop(size=img_size),
        T.RandomHorizontalFlip(),
        T.RandomApply([color_jitter], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)       
    ]
    test_transform = [
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ]
    return train_transform, test_transform

def get_dataloader(
    name,
    trainloader_config,
    testloader_config=None,
    train_transform: callable=None,
    test_transform: callable=None,
    save_path=None
):
    use_testset = testloader_config is not None
    if not use_testset:
        assert test_transform is None
    
    if save_path is None:
        save_path = f'./data/{name}'
    
    trainset, testset = _get_dataset(
        save_path, name, 
        use_testset,
        train_transform, test_transform
    )
    
    trainloader = DataLoader(trainset, **trainloader_config)
    
    testloader = None
    if use_testset:
        testloader = DataLoader(testset, **testloader_config)
    
    return trainloader, testloader

    
    


def _dataset_lookup(name: str):
    _available = ['CIFAR10']
    assert name in _available
    
    dataset = None
    if name == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10
    
    
    return dataset
    


def _get_dataset(
    save_path, 
    name='CIFAR10', 
    use_testset=True,
    train_transform=None,
    test_transform=None

):
    dataset_api = _dataset_lookup(name)
    
    if save_path is None:
        save_path = f'./data/{name}'
    
    trainset = dataset_api(
        save_path, train=True,
        transform=train_transform,
        download=True
    )
    
    testset = None
    if use_testset:
        testset = dataset_api(
            save_path, train=False,
            transform=test_transform,
            download=True
        )
    return trainset, testset
    
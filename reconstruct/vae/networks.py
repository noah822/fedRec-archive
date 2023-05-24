import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import os
from functools import reduce
import numpy as np
from .._utils.nn.autoencoder import (
    get_aligned_decoder
)

def _make_mlp(inplanes, hidden_dim, out_dim):
    return nn.Sequential(
        nn.Linear(inplanes, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim)
    )
    
def _make_conv(
    n_layer, n_channel,
    kernel_size, stride,
    padding=None,
    use_bn=False,
    use_deconv=False,
):
    assert len(n_channel) == n_layer + 1
    assert len(kernel_size) == n_layer
    assert len(stride) == n_layer
    
    if padding is not None:
        assert len(padding) == n_layer
    
    if padding is None:
        padding = [0 for _ in range(n_layer)]
    
    layer_type = nn.Conv2d
    if use_deconv:
        layer_type = nn.ConvTranspose2d

    convs = []
    for i in range(n_layer):
        inplane, outplane = n_channel[i:i+2]
        if use_bn:
            convs += [
                layer_type(inplane, outplane, kernel_size[i], stride[i], padding=padding[i]),
                nn.ReLU(),
                nn.BatchNorm2d(outplane)
            ]
        else:
            convs += [
                layer_type(inplane, outplane, kernel_size[i], stride[i], padding=padding[i]),
                nn.ReLU(),
            ]
            
    return nn.Sequential(*convs)

mnist_img_encoder = nn.Sequential(
    _make_conv(3, [1, 4, 8, 16], kernel_size=[5, 5, 3], stride=[1, 1, 1]),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(),
    _make_mlp(16, 128, 16)
)

mnist_img_decoder = nn.Sequential(
    nn.Linear(16, 3200),
    nn.ReLU(),
    nn.Unflatten(-1, (8, 20, 20)),
    nn.ReLU(),
    _make_conv(2, [8, 4, 1], [5, 5], [1, 1], use_deconv=True)[:-1],
    nn.Sigmoid()
)



mnist_img_linear_encoder = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 8),
)

mnist_img_linear_decoder = nn.Sequential(
    nn.Linear(8, 128),
    nn.ReLU(),
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, 28*28),
    nn.Sigmoid()
)



# do non-square convolutino over mfccs
mnist_audio_encoder = nn.Sequential(
    _make_conv(
        4, [1, 32, 64, 64, 64],
        kernel_size=[4, 3, 3, 3],
        stride=[2, 2, 1, 1],
        use_bn=True
    ),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(),
    _make_mlp(64, 128, 16)
)


mnist_audio_decoder = nn.Sequential(
    nn.Linear(16, 64 * 27 * 3),
    nn.ReLU(),
    nn.Unflatten(-1, (64, 27, 3)),
    _make_conv(
        4, [64, 64, 64, 32, 1],
        kernel_size=[3, 3, 3, 4],
        stride=[1, 1, 2, 2],
        use_bn=True, 
        use_deconv=True
    )
)

mnist_audio_decoder = get_aligned_decoder(
    (32, 1, 128, 32),
    mnist_audio_encoder, 
    mnist_audio_decoder
)

class imageMNIST(Dataset):
    def __init__(self, dir_path, transform=None):
        super().__init__()
        self.dir_path = dir_path

        self.filenames = []
        self.labels = []
        self.transform = transform
        
        # traverse dataset directory
        for dir in os.scandir(dir_path):
            if dir.is_dir():
                for file in os.scandir(dir.path):
                    self.filenames.append(file.path)
                    # parse label 
                    label = int(os.path.basename(file.path).split('_')[0])
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = torchvision.io.read_image(self.filenames[index]) / 255
        if self.transform is not None:
            img = self.transform(img)
            return img, self.labels[index]
        return img.reshape(1, -1), self.labels[index]

class audioMNIST(Dataset):
    def __init__(self, dir_path):
        super().__init__()
        self.dir_path = dir_path

        self.filenames = []
        self.labels = []
        
        
        # traverse dataset directory
        for file in os.scandir(dir_path):
          self.filenames.append(file.path)
          label = int(os.path.basename(file.path).split('_')[0])
          self.labels.append(label)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        with open(self.filenames[index], 'rb') as f:
          waveform = np.load(f)
        return waveform, self.labels[index]
    
class mmMNIST(Dataset):
    pass




def get_MNIST_dataloader(
    path,
    audio_only=False, image_only=False,
    csv_path=None,
    trainloader_config=None,
    valloader_config=None,
    train_val_split_ratio=0.7
):
    assert not (audio_only and image_only)
    if audio_only:
        pass
    if image_only:
        pass
    
    if not (audio_only or image_only):
        assert csv_path is not None
    
    
        
    
    

if __name__ == '__main__':
    x = torch.randn(32, 1, 128, 32)
    net = nn.Sequential(
        mnist_audio_encoder, 
        mnist_audio_decoder
    )
    res = net(x)
    # res = mnist_audio_encoder(x)
    print(res.shape)
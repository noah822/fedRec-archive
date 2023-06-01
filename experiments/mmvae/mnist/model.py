import torch
import torch.nn as nn
import os
from functools import reduce
import numpy as np
from reconstruct._utils.nn.autoencoder import (
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
    _make_mlp(16, 128, 64)
)

mnist_img_decoder = nn.Sequential(
    nn.Linear(32, 3200),
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
        6, [1, 512, 512, 256, 256, 128, 64],
        kernel_size=[4, 4, 3, 3, 3, 3],
        stride=[2, 2, 1, 1, 1, 1],
        padding=[1, 1, 1, 1, 1, 1],
        use_bn=True
    ),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(),
    _make_mlp(64, 256, 64)
)


mnist_audio_decoder = nn.Sequential(
    nn.Linear(32,  64 * 32 * 8),
    nn.ReLU(),
    nn.Unflatten(-1, (64, 32, 8)),
    _make_conv(
        6, [64, 128, 256, 256, 512, 512, 1],
        kernel_size=[3, 3, 3, 3, 4, 4],
        stride=[1, 1, 1, 1, 2, 2],
        padding=[1, 1, 1, 1, 1, 1],
        use_bn=True, 
        use_deconv=True
    )
)


mnist_audio_decoder = get_aligned_decoder(
    (32, 1, 128, 32),
    mnist_audio_encoder, 
    mnist_audio_decoder
)




    
    
    
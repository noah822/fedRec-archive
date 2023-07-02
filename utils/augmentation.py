import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as audioF
import numpy as np

import random
from typing import List, Callable, Dict


def get_default_img_transforms(input_shape, n_channel=3, s=1):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    transform_list = [transforms.RandomResizedCrop(size=input_shape),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomApply([color_jitter], p=0.8)] 
    if n_channel == 3:
        transform_list.append(
            transforms.RandomGrayscale(p=0.2)
        )
    
    data_transforms = transforms.Compose(transforms)
    return data_transforms


def multiview_dataset_wrapper(transforms, n_view=2, keep_original=False):
    def _wrapper(dataset_cls):
        class _multiview_dataset_wrapper(dataset_cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.transforms = transforms
                self.n_view = n_view
                self.keep_original = keep_original
                
                self.multiviewed = True
                
            def __len__(self):
                return super().__len__()
            
            def __getitem__(self, index):
                if not self.multiviewed:
                    return super().__getitem__(index)
                
                data = super().__getitem__(index)
                if len(data) > 1:
                    data = data[0]
                    
                views = [None for _ in range(self.n_view)]
                if self.keep_original:
                    views[0] = data
                    for i in range(1, self.n_view):
                        views[i] = self.transforms(data)
                else:
                    for i in range(self.n_view):
                        views[i] = self.transforms(data)
                return views
            
            def unpack(self):
                self.multiviewed = False
            def pack(self):
                self.multiviewed = True
                
        return _multiview_dataset_wrapper

    return _wrapper

'''
    Augmentations for audio(subclass nn.Module, can be loaded to GPUs)
    1. RandomSpeedChange
    2. 
'''

class RandomSpeedChange(nn.Module):
    def __init__(self, 
                 sr: int,
                 speed_factor: List[float]=[1.4],
                 p: float=0.5
                ):
        super(RandomSpeedChange, self).__init__()
        self.sr = sr
        self.speed_factor = speed_factor
        self._strecher_pool = nn.ModuleList([
            torchaudio.transforms.Speed(orig_freq=sr, factor=k) for k in speed_factor
        ])
        self.p = p

    def forward(self, wave: torch.Tensor):
        if random.random() < self.p:
            strecher = random.choice(self._strecher_pool)
            stretched_wave, _ = strecher(wave)
        else:
            stretched_wave = wave
        return stretched_wave
    
class Reverb(nn.Module):
    def __init__(self, impulse_sample: torch.Tensor, p: float=0.5):
        super(Reverb, self).__init__()
        self._impulse = impulse_sample
        self.p = p
    def forward(self, waveform):
        if random.random() < self.p:
            # align shape of impulse, repeat along batch dimension if necessary
            batched_impulse = self._impulse.repeat(*waveform.shape[:-1],1)
            print(batched_impulse.shape)
            print(waveform.shape)
            rir_augmented = audioF.fftconvolve(waveform, batched_impulse)
        else:
            rir_augmented = waveform
        return rir_augmented
    


from torch_audiomentations import Compose, Gain, PolarityInversion, AddBackgroundNoise

def get_default_waveform_augmentation(sr, noise_path, impulse_path):
    rir_raw, sample_rate = torchaudio.load(impulse_path)
    rir = rir_raw[:, int(sample_rate * 1.01) : int(sample_rate * 1.3)]
    _default_impulse = rir / torch.norm(rir, p=2)

    # Initialize augmentation callable
    base_augmentation = Compose(
        transforms=[
            Gain(
                min_gain_in_db=-15.0,
                max_gain_in_db=5.0,
                p=0.5,
            ),
            PolarityInversion(p=0.6),
            AddBackgroundNoise(background_paths=noise_path,
                            min_snr_in_db=5,
                            max_snr_in_db=35,
                            p=0.6),
        ]
    )
    # thin wrapper of torch_audiomentation callable list 
    base_augmentation = _KeywordFwdWrapper(base_augmentation)
    base_augmentation.register({'sample_rate' : sr})

    pipeline = [
        base_augmentation,
        Reverb(
            impulse_sample=_default_impulse,
            p=1
        ),
        RandomSpeedChange(
            sr=48000,
            speed_factor=[0.8, 1.2, 1.4],
            p=1
        ),
    ]
    pipeline = _CallableChain(pipeline)

    return pipeline 


class _CallableChain(nn.Module):
    def __init__(self, callable_list: List[Callable]):
        super(_CallableChain, self).__init__()

        # register as ModuleList
        self.callable_list = nn.ModuleList(callable_list)
    
    def forward(self, x):
        for f in self.callable_list:
            x = f(x)
        return x
    

class _KeywordFwdWrapper(nn.Module):
    def __init__(self, fn: Callable):
        super(_KeywordFwdWrapper, self).__init__()
        self.fn = fn
    def register(self, kwargs: Dict):
        self._kwargs = kwargs
    def forward(self, x):
        return self.fn(x, **self._kwargs)




class Augmentation(nn.Module):
    def __init__(self, transforms, n_view=2):
        super(Augmentation, self).__init__()
        self.transform = nn.Sequential(*transforms)
        self.n_view = n_view
    def forward(self, x):
        if self.n_view == 1:
            return self.transform(x)
        else:
            return [self.transform(x) for _ in range(self.n_view)]

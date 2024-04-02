import os
import random
from typing import Callable
import torch
import numpy as np
import pandas as pd

from torch.utils.data import (
  Dataset, DataLoader,
  random_split
)
import torchvision
from pathlib import Path


def _recursive_traverse(root_path):
    
  def _impl(path, res):
    if path.is_file():
      res.append(path.path)
      return
    elif path.is_dir():
      for obj in os.scandir(path):
        _impl(obj, res)
    return 

  res = []
  for obj in os.scandir(root_path):
    _impl(obj, res)
  
  return res

def _extract_label(file_iterator):
  res = [[] for _ in range(10)]
  for file in file_iterator:
    label = int(os.path.basename(file).split('_')[0])
    res[label].append(file)
  return res      


def random_pairing(audio_path, image_path, seed=272, save_path='./mmMNIST.csv'):
  pipeline = lambda x: _extract_label(_recursive_traverse(x))
  audio_files = pipeline(audio_path)
  image_files = pipeline(image_path)

  random.seed(seed)
  for audio, image in zip(audio_files, image_files):
    random.shuffle(audio)
    random.shuffle(image)
  audio_files = np.array(audio_files).reshape(1, -1)
  image_files = np.array(image_files).reshape(1, -1)

  df = pd.DataFrame(
      np.concatenate([audio_files, image_files], axis=0).T,
      columns=['audio', 'image']
  )
  df.to_csv(save_path, index=False)


def _get_view_path(orig_path: str, view_folder: str, view_index: int) -> str:
    orig_path: Path = Path(orig_path)
    basename = orig_path.name.split('.')[0]
    extension = orig_path.suffix
    view_filename = f'{basename}_view{view_index}{extension}'
    view_full_path = os.path.join(view_folder, view_filename)
    return view_full_path

def _extract_label(path: str) -> int:
    stem = Path(path).stem.split('_')[0]
    return int(stem)

  
class imageMNIST(Dataset):
    def __init__(self,
                 dir_path: str=None,
                 csv_path: str=None,
                 augmentation: Callable=None,
                 transform: Callable=None
            ):
        super().__init__()
        self.dir_path = dir_path

        self.filenames = []
        self.labels = []
        
        self.transform = transform

        assert not ( (dir_path is None) and (csv_path is None) )
        assert (dir_path is None) or (csv_path is None)
                
        # traverse dataset directory
        if dir_path is not None:
          self.filenames = _recursive_traverse(dir_path)
          for file in self.filenames:
              label = _extract_label(file)
              self.labels.append(label)
        
        if csv_path is not None:
            rowiter = pd.read_csv(csv_path).to_numpy().reshape(-1)
            for file in rowiter:
                self.filenames.append(file)
                label = _extract_label(file)
                self.labels.append(label)
        
        self.augmentation = augmentation
        
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = torchvision.io.read_image(self.filenames[index]) / 255
        

        if self.augmentation is not None:
            augmented_img = self.augmentation(img)
        if self.transform is not None:
            img = self.transform(img)
            
        if self.augmentation is not None:
            return img, augmented_img
        else:
            return img, self.labels[index]

class audioMNIST(Dataset):
    def __init__(
            self,
            dir_path: str=None,
            csv_path: str=None,
            augment_folder: str=None,
            num_view: int=2
          ):
        super().__init__()
        self.dir_path = dir_path

        self.filenames = []
        self.labels = []

        self.augment_folder = augment_folder
        self._offline_augment = augment_folder is not None

        self.num_view = num_view


        # if dir_path is provided, do recursive directory reverse
        # elif csv_path is provided, 
        assert not ( (dir_path is None) and (csv_path is None) )
        assert (dir_path is None) or (csv_path is None)
                
        # traverse dataset directory
        if dir_path is not None:
          self.filenames = _recursive_traverse(dir_path)
          for file in self.filenames:
              label = _extract_label(file)
              self.labels.append(label)
        
        if csv_path is not None:
            rowiter = pd.read_csv(csv_path).to_numpy().reshape(-1)
            for file in rowiter:
                self.filenames.append(file)
                label = _extract_label(file)
                self.labels.append(label)

    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        filename = self.filenames[index]
        label = self.labels[index]
        with open(filename, 'rb') as f:
          waveform = np.load(f)

        if self._offline_augment:
            sampled_view = random.choice(range(self.num_view))
            view_path = _get_view_path(filename, self.augment_folder, sampled_view)
            with open(view_path, 'rb') as f:
                augmented_waveform = np.load(f)
            return waveform, augmented_waveform
        else:
            return waveform, label

import os
  
    
class mmMNIST(Dataset):
    def __init__(
            self, csv_path,
            image_transform=None,
            with_label=False
          ):
          self.csv_path = csv_path
          
          self.files = pd.read_csv(csv_path).to_numpy()
          self.image_transform = image_transform
          self.with_label = with_label

    def __len__(self):
          return len(self.files)
    
        
    def __getitem__(self, index):
          audio_file, image_file = self.files[index]
          label = torch.tensor(int(os.path.basename(audio_file).split('_')[0]))

          with open(audio_file, 'rb') as f:
            waveform = torch.tensor(np.load(f))

          img = torchvision.io.read_image(image_file) / 255
          if self.image_transform is not None:
                img = self.image_transform(img)

          if not self.with_label:
              return waveform, img
          else:
              return waveform, img, label


'''
   Another csv related pipeline,
  - each client read from their corresponding csv file
  - get dataloader according three states specified:
    a. audio-only w/o off-line augmentation
    b. image-only w/o off-line augmentation  
    c. both modalities presented

'''

def get_MNIST_dataloader(
    audio_only=False, image_only=False,
    audio_path=None, image_path=None,
    csv_path=None,
    trainloader_config=None,
    testloader_config=None,
    train_val_split_ratio=None,
    dataset_only=False
):
    assert not (audio_only and image_only)
    dataset = None
    
    if audio_only:
        assert audio_path is not None
        dataset = audioMNIST(audio_path)
    if image_only:
        assert image_path is not None
        dataset = imageMNIST(image_path)
    
    if not (audio_only or image_only):
        assert csv_path is not None
        dataset = mmMNIST(csv_path)
        
    if train_val_split_ratio is not None:
          assert trainloader_config is not None \
            and testloader_config is not None
            
          train_size = len(dataset) * train_val_split_ratio
          test_size = len(dataset) - train_size
          trainset, testset = random_split(dataset, [train_size, test_size])
          if dataset_only:
              return trainset, testset
          
          trainloader = DataLoader(trainset, **trainloader_config)
          testloader = DataLoader(testset, **testloader_config)
          return trainloader, testloader
    else:
          if dataset_only:
              return dataset
          dataloader = DataLoader(dataset, **trainloader_config)
          return dataloader
        

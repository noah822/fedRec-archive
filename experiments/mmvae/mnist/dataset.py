import os
import random
from typing import Any
import torch
import numpy as np
import pandas as pd

from torch.utils.data import (
  Dataset, DataLoader,
  random_split
)
import torchvision


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
    def __init__(self, audio_path, image_path, csv_path):
          self.audio_path = audio_path
          self.image_path = image_path
          self.csv_path = csv_path
          
          self.files = pd.read_csv(csv_path).to_numpy()
    
    def __len__(self):
          return len(self.files)
        
    def __getitem__(self, index):
          audio_file, image_file = self.files[index]
          label = torch.tensor(int(os.path.basename(audio_file).split('_')[0]))
          
          with open(audio_file, 'rb') as f:
            waveform = torch.tensor(np.load(f))
          img = torchvision.io.read_image(image_file) / 255
          
          return waveform, img, label
          




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
        dataset = mmMNIST(
          audio_path, image_path,
          csv_path
        )
        
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
        

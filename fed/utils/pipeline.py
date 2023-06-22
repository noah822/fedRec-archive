import torch
import torch.nn as nn
from torch.utils.data import Dataset

from typing import (
     Dict, Iterable, 
     Tuple
)

from reconstruct.mmvae import DecoupledMMVAE
from utils.train import vanilla_trainer
import torch.optim as optim

from ._internal import _infer_device, _iter_dict

class PairedFeatureBank(Dataset):
    '''
    Args:
    - raw_data: dataset 
    '''
    def __init__(self,
                 raw_dataset,
                 extractors: Tuple[nn.Module]
                 ):
            self.extractors = extractors
            self.raw_dataset = raw_dataset

            self._fb = PairedFeatureBank.construct_fb_array(
                 raw_dataset, 
                 extractors, 
                 to_tensor=False
            )
    
    def __len__(self):
         return len(self._fb)
    def __getitem__(self, index):
         return self._fb[index]

    @staticmethod
    def construct_fb_array(
                     dataset,
                     extractors,
                     unpack: callable=None
                     ):
        
        # device = _infer_device(dataset)
        feature_bank = []
        for inputs in dataset:
            instance_embed = []
            if unpack is not None:
                 inputs = unpack(inputs)
            for (model, x) in zip(extractors, inputs):
                 instance_embed.append(model(x))
            
            feature_bank.append(
                instance_embed
            )
        return feature_bank


class MMVAETrainer:
     '''
     Args:
     - 
     '''
     def __init__(self,
            mmvae,
            dataset: Iterable=None,
            ):
        self.dataset = dataset
        self.mmvae = mmvae

     def train(self):
          epoch = 15
          optimizer = optim.Adam(self.mmvae.parameters(), lr=1e-3, weight_decay=1e-5)
          device = _infer_device(self.mmvae)

          def _unpack_and_forward(model: DecoupledMMVAE, inputs):
               audio_embed, image_embed = inputs
               x = {
                    'audio' : audio_embed,
                    'image' : image_embed
               }
               nelbo, kl, rec, verbose_output = model(
                    x, alpha=0.001,
                    joint_postieror='MoE',
                    iw_cross_mod=False,
                    verbose=True
               )
               prompt = {
                    'nelbo' : nelbo.item(),
                    'kl' : kl.item(),
                    'rec' : rec.item()
               }
               
               for k, v in verbose_output.items():
                    prompt[k] = v.item()

               return nelbo, prompt

          vanilla_trainer(
               self.mmvae, self.dataset,
               optimizer, None,
               epoch, 
               device, 
               _unpack_and_forward,
               do_autoencode=False,
               custom_prompt=True
          )

     def export_mmvae(self):
          return self.mmvae.state_dict()
          

              

          


    
        
        

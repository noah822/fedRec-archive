import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from typing import (
     Dict, Iterable, 
     Tuple
)

from reconstruct.mmvae import DecoupledMMVAE
from utils.train import vanilla_trainer
import torch.optim as optim

from ._internal import _infer_device, _iter_dict
import os

class PairedFeatureBank(Dataset):
    '''
    Args:
    - raw_data: dataset 
    '''
    def __init__(self,
                 raw_dataset,
                 extractors: Tuple[nn.Module],
                 device,
                 dataloader_config=None,
                 file_buffer='./.feature_bank'
                 ):
          self.extractors = extractors
          self.raw_dataset = raw_dataset
          if dataloader_config is None:
               dataloader_config = {
                    'batch_size' : 128,
                    'shuffle' : False,
                    'num_workers' : 2 
               }
          raw_dataloader = DataLoader(raw_dataset, **dataloader_config)

          self.file_buffer = file_buffer
          if not os.path.isdir(file_buffer):
               os.makedirs(file_buffer, exist_ok=True)

          self._fb = PairedFeatureBank.construct_fb_array(
               raw_dataloader, 
               extractors,
               device
          )
    
    def __len__(self):
         return len(self.raw_dataset)
    def __getitem__(self, index):
         tensor_path = f'{self.file_buffer}/{index}.pt'
         embed = torch.load(tensor_path)
         return embed

    @staticmethod
    def construct_fb_array(
                     dataloader,
                     extractors,
                     device,
                     file_buffer: str,
                     unpack: callable=None
                     ):
        
        cnt = 0
        for inputs in dataloader:
            instance_embed = []
            if unpack is not None:
                 inputs = unpack(inputs)
            for (model, x) in zip(extractors, inputs):
                 model.eval()
                 with torch.no_grad():
                    instance_embed.append(model(x.to(device)))
            
            # flatten out batched embeddings and save it into file buffer
            # instance_embed: (mod, batch, latent_dim)
            instance_embed = torch.stack(instance_embed).permute(1, 0, 2)
            for embed in instance_embed:
                 torch.save(embed, f'{file_buffer}/{cnt}.pt')
                 cnt += 1
            
             


class MMVAETrainer:
     '''
     Args:
     - 
     '''
     def __init__(self,
            mmvae,
            dataset: Iterable=None,
            dataloader_config: Dict=None
            ):
          self.dataset = dataset
          self.mmvae = mmvae
          if dataloader_config is None:
               dataloader_config = {
                    'batch_size' : 32,
                    'shuffle' : True,
                    'num_workers' : 2
                    }
          self.dataloader = DataLoader(self.dataset, **dataloader_config)
        

     def train(self):
          epoch = 15
          optimizer = optim.Adam(self.mmvae.parameters(), lr=1e-3, weight_decay=1e-5)
          device = _infer_device(self.mmvae)

          def _unpack_and_forward(model: DecoupledMMVAE, inputs: torch.Tensor):
               inputs = inputs.permute(1,0,2)
               audio_embed = inputs[0]; image_embed = inputs[1]
               x = {
                    'audio' : audio_embed,
                    'image' : image_embed
               }
               nelbo, kl, rec, verbose_output = model(
                    x, alpha=0.001,
                    joint_posterior='MoE',
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
               self.mmvae, self.dataloader,
               optimizer, None,
               epoch, 
               device, 
               _unpack_and_forward,
               do_autoencode=False,
               custom_prompt=True
          )

     def export_mmvae(self):
          return self.mmvae.state_dict()
          

              

          


    
        
        

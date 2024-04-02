import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from typing import (
     Dict, Iterable, 
     Tuple, List
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
                    'batch_size' : 32,
                    'shuffle' : False
               }
          raw_dataloader = DataLoader(raw_dataset, **dataloader_config)

          self.file_buffer = file_buffer
          if not os.path.isdir(file_buffer):
               os.makedirs(file_buffer, exist_ok=True)

          self._fb = PairedFeatureBank.construct_fb_array(
               raw_dataloader, 
               extractors,
               device,
               file_buffer
          )
    
    def __len__(self):
         return len(self.raw_dataset)
    def __getitem__(self, index):
         tensor_path = f'{self.file_buffer}/{index}.pt'
         embed = torch.load(tensor_path)
         return embed[0], embed[1]

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
                 # model.eval()
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
            posterior_opt: str='PoE',
            alpha: float=1.,
            dataset: Iterable=None,
            dataloader_config=None,
            mod_keys: List[str]=None,
            optim_config: Dict[str, float]=None,
            cross_loss_only: bool=False,
            ):
        self.dataset = dataset
        self.mmvae = mmvae
        if dataloader_config is None:
            dataloader_config = {
                'batch_size' : 32,
                'shuffle' : True
            }
        self.dataloader = DataLoader(self.dataset, **dataloader_config)

        if optim_config is None:
            self.optim_config = {
                'lr' : 1e-3,
                'weight_decay' : 1e-5
            }
        else:
            self.optim_config = optim_config

        self.cross_loss_only = cross_loss_only
        if mod_keys is None:
            self._mod_keys = list(mmvae.encoders.keys())
        else:
            self._mod_keys = mod_keys
            
        # mmvae forwarding config
        self._posterior_opt = posterior_opt
        self._alpha = alpha

     def train(self, epoch):
          optimizer = optim.Adam(self.mmvae.parameters(), **self.optim_config)
          device = _infer_device(self.mmvae)

          def _unpack_and_forward(model: DecoupledMMVAE, inputs):
               x = dict()
               # wrap inputs into dict
               for input_x, mod_name in zip(inputs, self._mod_keys):
                   x[mod_name] = input_x
            
               nelbo, kl, rec, verbose_output = model(
                    x, alpha=self._alpha,
                    joint_posterior=self._posterior_opt,
                    iw_cross_mod=False,
                    verbose=True,
                    cross_loss_only=self.cross_loss_only
               )
               prompt = {}
               
               for k, v in verbose_output.items():
                    prompt[k] = v
               prompt['kl'] = kl.item()

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
     

class PriorWeightFitter:
    def __init__(self,
                 priors: torch.Tensor,
                 mapping: nn.Module):
        '''
        Args:
        - priors: (n_center, n_dim), priors, weight of which will be learned
        - mapping: torch Module, generates weights given input for each corresponding
        output for reconstruction
        '''
        self.n_center, self.n_dim = priors.shape
        self._priors = priors
        self._mapping = mapping

        self.criterion = nn.MSELoss()
    
    def fit(self, x: torch.Tensor, y: torch.Tensor):
        self._default_setup()

        # wrap tensor into Dataset
        wrapped_tset = _TensorDatasetWrap([x, y])
        dl = DataLoader(wrapped_tset, **self.dl_config)
        optimizer = optim.Adam(self._mapping, **self.optim_config)

        for _ in range(self.epoch):
            for inputs in dl:
                x, y = inputs
                loss = self._fwd_impl(x, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        return self._mapping

    def _default_setup(self):   
        self.optim_confg = {
            'lr' : 1e-3,
            'weight_decay' : 1e-5
        }

        self.train_config = {
            'epoch' : 15,
        }
        self.dl_config = {
            'batch_size' : 32,
            'shuffle' : True
        }

    def _fwd_impl(self, x, y):
        weights = self._mapping(x) # (batch, n_cluster)
        pred = weights @ self._priors

        loss = self.criterion(pred, y)
        return loss
     


class _TensorDatasetWrap(Dataset):
    def __init__(self, tensor_set: List[torch.Tensor]):
        super(_TensorDatasetWrap, self).__init__()
        self._tensor_set = tensor_set

        tensor_instance = None
        if _TensorDatasetWrap._t_collector_check(tensor_set):
            tensor_instance = next(iter(tensor_set))
        else:
            tensor_instance = tensor_set

        self.num_sample, self.num_dim = tensor_instance.shape

    
    def __len__(self):
        return self.num_sample
    
    def __getitem__(self, idx):
        if not _TensorDatasetWrap._t_collector_check(self._tensor_set):
            return self._tensor_set[idx]
        else:
            return [t[idx] for t in self._tensor_set]

    @staticmethod
    def _t_collector_check(t):
        return isinstance(t, (list, tuple))
    

def fit_cross_mod_rec(
    rec_networks: List[nn.Module],
    dataloader: DataLoader,
    optimizer,
    epoch: int,
    device: str='cuda',
    verbose: bool=False
):
    # by default, the loss function of embedding reconstruction is MSE
    criterion = nn.MSELoss()
    def unpack_and_fwd(model, inputs):
        inputs = inputs.permute(1, 0, 2) # (mod, batch, latent_dim)
        model_y_from_x, model_x_from_y = model

        mod_x, mod_y = inputs
        mod_x = mod_x.to(device); mod_y = mod_y.to(device)
        # approximate embed from one modality from another
        rec_y_from_x = criterion(
            model_y_from_x(mod_x), mod_y
        )
        rec_x_from_y = criterion(
            model_x_from_y(mod_y), mod_x
        )
        loss = rec_x_from_y + rec_y_from_x
        if verbose:
            prompt = {
                '1_from_0' : rec_y_from_x.item(),
                '0_from_1' : rec_x_from_y.item()
            }
            return loss, prompt
        else:
            return loss
    
    vanilla_trainer(
        rec_networks, dataloader,
        optimizer, None,
        epoch, device,
        unpack_and_fwd,
        use_tqdm=verbose
    )
            
    


          

              

          


          


    
        
        

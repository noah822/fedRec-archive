from abc import ABC, abstractmethod

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import trange, tqdm
from typing import (
    List, Dict, Callable
)


from .utils.pipeline import PairedFeatureBank
from .utils._internal import _infer_device, _zippable

from reconstruct.mmvae import DecoupledMMVAE
from experiments.mmvae.mnist.model import _make_mlp



class BaseRecNetTrainer(ABC):
    def __init__(self,
                 extractors: List[nn.Module],
                 raw_dataset: Dataset,
                 embed_pool_dl_conifg: Dict=None):
        super(BaseRecNetTrainer, self).__init__()
        self._extractors = extractors
        self._raw_dataset = raw_dataset

        default_dl_config = {
            'batch_size' : 64, 'shuffle' : True
        }
        self._dl_config = default_dl_config if embed_pool_dl_conifg \
                          else embed_pool_dl_conifg
        self.device = _infer_device(next(iter(extractors)))
    
    def build_embed_pool(self,
                         extractors: List[nn.Module],
                         raw_dataset: Dataset):
        return PairedFeatureBank(
            raw_dataset, extractors,
            self.device
        )
    
    @abstractmethod
    def train_on_batch(self, *args):
        pass
    
    @abstractmethod
    def export_model(self) -> Dict:
        pass
        
    
    def run(self,
            epoch: int,
            dl_config: Dict=None,
            verbose: bool=False):
        dl_config = {'batch_size':64, 'shuffle': True} if dl_config is None else dl_config
        embed_pool = self.build_embed_pool(self._extractors, self._raw_dataset)
        embed_dl = DataLoader(embed_pool, **dl_config)

        epoch_iter = range(epoch) if verbose is False else trange(epoch)
        for _ in epoch_iter:
            data_iter = embed_dl if verbose is False else tqdm(embed_dl)
            for embeds in data_iter:
                embeds = _zippable(embeds)
                
                prompt = self.train_on_batch(*embeds)
                if (prompt is not None) and verbose:
                    data_iter.set_postfix(**prompt)
    

class MLPRecNetTrainer(BaseRecNetTrainer):
    def __init__(self,
                 embed_dim: int,
                 hidden_dim: int,
                 extractors: List[nn.Module],
                 raw_dataset: Dataset,
                 optim_config: Dict=None,
                 criterion: Callable=None,
                 device: str='cuda'):
        super(MLPRecNetTrainer, self).__init__(extractors, raw_dataset)
        self.recnets = nn.ModuleDict({
            '0_to_1' : _make_mlp(embed_dim, hidden_dim, embed_dim),
            '1_to_0' : _make_mlp(embed_dim, hidden_dim, embed_dim)
        }).to(device)
        optim_config = {'lr' : 1e-3, 'weight_decay' : 1e-5} if optim_config is None else optim_config
        self.optimizer = optim.Adam(self.recnets.parameters(), **optim_config)
        self.criterion = nn.MSELoss() if criterion is None else criterion

    def train_on_batch(self,
                       view0: torch.Tensor,
                       view1: torch.Tensor):
        view0 = view0.to(self.device)
        view1 = view1.to(self.device)
        rec_0_to_1 = self.criterion(
            self.recnets['0_to_1'](view0), view1
        )
        rec_1_to_0 = self.criterion(
            self.recnets['1_to_0'](view1), view0
        )
        loss = (rec_0_to_1 + rec_1_to_0) / 2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            '0_to_1' : rec_0_to_1.item(),
            '1_to_0' : rec_1_to_0.item()
        }

    def export_model(self):
        return {
            '0_to_1' : self.recnets['0_to_1'].state_dict(),
            '1_to_0' : self.recnets['1_to_0'].state_dict()
        }

class MMVAERecNetTrainer(BaseRecNetTrainer):
    def __init__(self,
                 embed_dim: int,
                 hidden_dim: int,
                 bottleneck_dim: int,
                 extractors: List[nn.Module],
                 raw_dataset: Dataset,
                 posterior_opt: str='PoE',
                 alpha: float=1.,
                 mod_penalty: Dict[str, float]=None,
                 cross_loss_only: bool=False,
                 self_loss_only: bool=False,
                 optim_config: Dict=None,
                 criterion: Callable=None,
                 device: str='cuda'):
        super(MMVAERecNetTrainer, self).__init__(extractors, raw_dataset)
        criterion = nn.MSELoss() if criterion is None else criterion
        mmvae_config = {
            'encoders' : nn.ModuleDict({
                '0' : _make_mlp(embed_dim, hidden_dim, bottleneck_dim * 2, use_bn=False),
                '1' : _make_mlp(embed_dim, hidden_dim, bottleneck_dim * 2, use_bn=False)
            }).to(device),
            'decoders' : nn.ModuleDict({
                '0' : _make_mlp(bottleneck_dim, hidden_dim, embed_dim, use_bn=False),
                '1' : _make_mlp(bottleneck_dim, hidden_dim, embed_dim, use_bn=False)
            }).to(device),
            'latent_dim' : bottleneck_dim,
            'score_fns' : {'0' : criterion, '1' : criterion}
        }
        self.recnets = DecoupledMMVAE(**mmvae_config, device=device)


        optim_config = {'lr' : 1e-3, 'weight_decay' : 1e-5} if optim_config is None else optim_config
        self.optimizer = optim.Adam(self.recnets.parameters(), **optim_config)

        
        self.mod_penalty = {'0':1., '1':1.} if mod_penalty is None else mod_penalty

        self.cross_loss_only, self.self_loss_only = cross_loss_only, self_loss_only
        self.device = device
        
        self.posterior_opt = posterior_opt
        self.alpha = alpha

    def train_on_batch(self,
                       view0: torch.Tensor,
                       view1: torch.Tensor):
        view0 = view0.to(self.device)
        view1 = view1.to(self.device)
        wrapped_inputs = {
            '0' : view0,
            '1' : view1
        }
        nelbo, kl, rec, verbose_output = self.recnets(
            wrapped_inputs, alpha=self.alpha,
            joint_posterior=self.posterior_opt,
            iw_cross_mod=False,
            verbose=True,
            mod_penalty=self.mod_penalty,
            cross_loss_only=self.cross_loss_only,
            self_loss_only=self.self_loss_only
        )

        self.optimizer.zero_grad()
        nelbo.backward()
        self.optimizer.step()
        verbose_output['kl'] = kl.item()
        return verbose_output

    def export_model(self):
        return {'mmvae' :  self.recnets.state_dict()}


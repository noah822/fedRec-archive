import torch
import torch.nn as nn
from typing import Dict
from abc import abstractmethod
from ..vae import kl_normal

'''
    Standardized jointer posterior inference network interface
    Args for forward[if subclasses nn.Moudle] or callable interface:
    - mu: (mod, batch, latent_dim)
    - logvar: (mod, batch, latent_dim)
    - batch_first: whehter mu/logvar parameters are configured as first dim to be batch
    - return_kl: if set to `True`, kl divergence with prior of latent distribution will be returned
    - mu_prior: default 0 
    - var_prior: default 1 
'''

class JointLatentInfer(nn.Module):
    def __init__(self):
        super(JointLatentInfer, self).__init__()
    
    @abstractmethod
    def sample_latent(self, mu, logvar):
        pass
    
    def compute_kl(self,
        mu, logvar,
        mu_prior, var_prior
    ):
        latent_dim = mu.shape[-1]
        mu = mu.view(-1, latent_dim)
        logvar = logvar.view(-1, latent_dim)
        
        
        if mu_prior is None:
            mu_prior = torch.zeros_like(mu)
        else:
            mu_prior = mu_prior.view(-1, latent_dim)
            
        if var_prior is None:
            var_prior = torch.ones_like(logvar)
        else:
            var_prior = var_prior.view(-1, latent_dim)
        
        return kl_normal(
            mu, torch.exp(logvar),
            mu_prior, var_prior
        )
    
    def forward(
        self,
        mu: torch.tensor,
        logvar: torch.tensor,
        mu_prior, var_prior,
        batch_first=False,
        return_kl=True,
    ):
     if batch_first:
         mu = mu.permute(1, 0, 2)
         logvar = logvar.permute(1, 0, 2)
     z = self.sample_latent(mu, logvar)
     
     if return_kl:
         kl = self.compute_kl(
             mu, logvar, 
             mu_prior, var_prior
         )
         return z, kl
     else:
         return z
     

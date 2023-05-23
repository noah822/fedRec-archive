import torch
import torch.nn as nn
from typing import Dict
from ..vae import (
    reparameterize, 
    kl_normal,
    kl_standard_gaussian
)

def _compute_kl(mu, logvar, mu_prior=None, var_prior=None):
    if mu_prior is None:
        mu_prior = torch.zeros_like(mu)
    if var_prior is None:
        var_prior = torch.ones_like(logvar)
    return kl_normal(
        mu, torch.exp(logvar),
        mu_prior, var_prior
    )



'''
    Product of Experts to aggregate per modality posterior estimate q(z|x_m)
'''
class PoE(nn.Module):
    def __init__(self):
        super(PoE, self).__init__()
    
    def foward(
        self,
        mu, 
        logvar,
        batch_first=True,
        return_kl=True,
        mu_prior=None,
        var_prior=None,
        eps=1e-8
    ):
        '''
        Product of Experts forward implementation:
        Args:
        - mu: (*, n_mod, latent_dim) means of Gaussians from different modalities
        - logvar: (*, n_mod, latent_dim) log of variance of Gaussians from different modalities
        
        Return:
        - joint gaussian posterior
        '''
        joint_mu, joint_logvar = PoE._posterior_param(mu, logvar, eps)
        z = reparameterize(joint_mu, joint_logvar)
        
        
        if return_kl:
            kl = _compute_kl(
                joint_mu, torch.exp(logvar),
                mu_prior, var_prior
            )
            return z, kl
    
        else:
            return z
        
    @staticmethod
    def _posterior_param(mu, logvar, eps=1e-8):
        '''
        Return mean and logvar of posterior of products of Gaussians analytically
        '''
        var = torch.exp(logvar) + eps
        T = 1 / var
        joint_mu = torch.sum(mu * T, dim=-2) / torch.sum(T, dim=-2)
        joint_logvar = torch.log(1 / torch.sum(T, dim=-2))
        return joint_mu, joint_logvar

'''
Mixture of Experts
'''
class MoE(nn.Module):
    def __init__(self):
        super(MoE, self).__init__()
        
    def foward(
        self,
        mu, 
        logvar,
        batch_first=True,
        return_kl=True,
        mu_prior=None,
        var_prior=None,
    ):
        z = reparameterize(mu, logvar).mean(dim=-2)
        
        if return_kl:
            kl = _compute_kl(
                torch.flatten(mu, start_dim=-2, end_dim=-1),
                torch.flatten(logvar, start_dim=-2, end_dim=-1),
                mu_prior, var_prior
            )
            z, kl
        else:
            return z
    
        
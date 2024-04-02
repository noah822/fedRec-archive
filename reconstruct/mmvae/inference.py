import torch
import torch.nn as nn
from typing import Dict
from ..vae import (
    reparameterize, 
)
from .base import JointLatentInfer

'''
 Subclass of JointLatentInfer
 - should impl the abstract method `sample_latent()` defined in JointLatentInfer
 - could overwrite default `compute_kl()` if necessary
'''


'''
    Product of Experts to aggregate per modality posterior estimate q(z|x_m)
'''
class PoE(JointLatentInfer):
    def __init__(self, eps=1e-8):
        super(PoE, self).__init__()
        self.eps = eps
    
    def sample_latent(self, mu, logvar):
        param_list = self._get_param_list(mu, logvar)
        
        latents = []
        for (mu, logvar) in param_list:
            latents.append(reparameterize(mu, logvar))

        return latents
    
    
    def _get_param_list(self, mu, logvar):
        param_list = []
        param_list.append((mu[0,:,:], logvar[0,:,:]))
        param_list.append((mu[1,:,:], logvar[1,:,:]))
        
        joint_mu, joint_logvar = PoE._posterior_param(mu, logvar)
        param_list.append((joint_mu, joint_logvar))
        
        return param_list
    
    def compute_kl(self, mu, logvar, mu_prior, var_prior):
        '''
        Args:
        - mu/logvar: (mod, batch, ...)
        '''
        param_list = self._get_param_list(mu, logvar)
        
        kl = .0
        for (mu, logvar) in param_list:
            mu_prior = torch.zeros_like(mu)
            var_prior = torch.ones_like(logvar)
            
            kl += super(PoE, self).compute_kl(
                mu, logvar,
                mu_prior, var_prior
            )
        
        return kl
        
    @staticmethod
    def _posterior_param(mu, logvar, eps=1e-8):
        '''
        Return mean and logvar of posterior of products of Gaussians analytically
        q(z|x1...m) = p(z)q(z|x_1)...q(z|x_m)
        which has a closed-form solution if the prior and posterior are chosen to be Gaussian
        
        Args:
        - mu: (mod, batch, latent_dim)
        - logvar: (mod, batch, latent_dim)
        '''
        # TODO: integrate customed prior 
        device = mu.device
        
        # expand mod dimension with prior
        prior_mu = torch.zeros(1, *mu.shape[1:]).to(device)
        prior_logvar = torch.zeros(1, *logvar.shape[1:]).to(device)
        
        mu = torch.cat([prior_mu, mu], dim=0)
        logvar = torch.cat([prior_logvar, logvar], dim=0)
        logvar = torch.clip(logvar, 1e-2, 8)
        assert not torch.isnan(logvar).any(), 'catch nan in posterior'
        
        var = torch.exp(logvar) + eps
        T = 1 / var
        joint_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        joint_logvar = torch.log(1 / torch.sum(T, dim=0))
        joint_logvar = torch.clip(joint_logvar, 1e-2, 8)
        return joint_mu, joint_logvar

'''
    Mixture of Experts
    used in Shi(2019): https://arxiv.org/abs/1911.03393
    stratified sampling
'''
class MoE(JointLatentInfer):
    def __init__(self):
        super(MoE, self).__init__()
        
    def sample_latent(self, mu, logvar):
        assert not torch.isnan(logvar).any(), 'catch nan here'
        z = reparameterize(mu, torch.clip(logvar, 1e-2, 1e2))
        return z
        
'''
    Mixture of Product of Experts
'''
class MoPoE(JointLatentInfer):
    def __init__(self):
        super(MoPoE, self).__init__()
    
    def sample_latent(self, mu, logvar):
        ...
    
    def compute_kl(self, mu, logvar, mu_prior, var_prior):
        ...
    
        
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
        joint_mu, joint_logvar = PoE._posterior_param(mu, logvar, self.eps)
        z = reparameterize(joint_mu, joint_logvar)
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
    used in Shi(2019): https://arxiv.org/abs/1911.03393
    stratified sampling
'''
class MoE(JointLatentInfer):
    def __init__(self):
        super(MoE, self).__init__()
        
    def sample_latent(self, mu, logvar):
        z = reparameterize(mu, logvar)
        return z
        
'''
    Mixture of Product of Experts
'''
class MoPoE(JointLatentInfer):
    def __init__(self):
        super(MoPoE, self).__init__()
    
    def sample_latent(self, mu, logvar):
        pass
    
    def compute_kl(self, mu, logvar, mu_prior, var_prior):
        pass
    
        
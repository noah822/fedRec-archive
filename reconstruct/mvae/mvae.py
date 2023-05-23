import torch
import torch.nn as nn
from typing import List

class MVAE(nn.Module):
    def __init__(
        self,
        encoder_list: List[nn.Module],
        decoder_list: List[nn.Module],
        sample_modality=True,
        use_model_parallel=False,
        gpu_ids=None
    ):
        
        if use_model_parallel:
            assert gpu_ids is not None
            self.use_model_parallel = True
            self.gpu_ids = gpu_ids
        
        self.z_prior_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_logvar = torch.log(
            nn.Parameter(torch.ones(1), requires_grad=False)
        )
        self.encoder_list = encoder_list
        self.decoder_list = decoder_list
        self.experts = ProductofExperts()
        
    def forward(self, *inputs):
        '''
        Args:
        - inputs: paired input for batched instance from different modalities
        
        Returns:
        - rec: list of reconstructed object returned by decoder of different modalities
          conditioned on shared latent representation
        - mean: mean of latent repr
        - logvar: log of variance of latent repr
        '''
        # step 1: encoder generates mean and var per modality
        
        z = []
        for x, encoder in zip(inputs, self.encoder_list):
            z.append(encoder(x))
        
        # add prior of latent repr to compute joint posterior
        
        z_prior = torch.normal(
            torch.zeros_like(z[0]), 
            torch.log(torch.ones_like(z[0]))
        )
        z = torch.cat([z_prior, z], dim=0).permute(1, 0, 2)
        
        # step 2: ProductOfExperts aggregation
        mean, logvar = torch.split(z, z.shape[-1] // 2, z.shape[-1])
        mean, logvar = self.experts(mean, logvar)
        
        # step 3: Reparameterization trick 
        epsilon = torch.normal(
            torch.zeros_like(mean), torch.ones_like(logvar)
        )
        
        latent_repr = mean + torch.exp(0.5 * logvar) * epsilon
        
        # step 4: decoders generate reconstruction conditioned on shared latent repr
        rec = []
        for decoder in self.decoder_list:
            rec.append(decoder(latent_repr))
        
        return rec, mean, logvar
    
class ProductofExperts(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, mean, logvar, eps=1e-8):
        '''
        Args:
        Parameters for q(z|x_k)
        - mean: (batch, modality, latent_dim)
        - logvar: (batch, modality, latent_dim)
          Here we assume that the covariance matrix is diagonal, which can be
          parameterized by # latent_dim parameters
        Return:
        - p(z | x_1, x_2, ..., x_M): (batch, latent_dim)
         posterior of latent representation conditioned on all modalities available
        
        '''
        var = torch.exp(logvar) + eps
        T = 1 / var
        experts_mean = torch.sum(mean * T, dim=-2) / torch.sum(T, dim=-2)
        experts_logvar = torch.log(1 / torch.sum(T, dim=-2))
        return experts_mean, experts_logvar
        
        
        
import torch
import torch.nn as nn
from typing import Dict
from ..vae.vae import (
    kl_standard_gaussian,
    reparameterize
)



class MMVAE(nn.Module):
    def __init__(self,
            encoders: Dict[str, nn.Module],
            experts,
            decoders: Dict[str, nn.Module],
            score_fns: Dict[str, callable],
            latent_dim
        ):
        '''
        Pipeline:
        1. Parameterize q(z | x_m) for each modality using encoders
        2. Compute joint posterior with experts model,
        3. Generate reconstruted input for each modelity given shared joint latent representation
        '''
        super(MMVAE, self).__init__()
        self.n_mod = len(encoders)
        self.encoders = encoders
        self.experts = experts
        self.decoders = decoders
        self.score_fns = score_fns
        
        self.latent_dim = latent_dim
    
    def forward(self, inputs: Dict[str, torch.Tensor], alpha=1.):
        latents = []
        for mod in sorted(inputs.keys()):
            x = inputs[mod]; encoder = self.encoders[mod]
            latents.append(encoder(x))
        
        latents = torch.concat(latents).permute(1,0,2)
        
        mu, logvar = torch.split(latents, self.latent_dim, dim=-1)
        joint_z, kl = self.experts(mu, logvar, batch_first=True, return_kl=True)
        
        rec_score = .0
        for mod in sorted(self.decoders.keys()):
            rec = self.decoders[mod](joint_z)
            rec_score += self.score_fns[mod](rec, inputs[mod]) 
        
        kl = kl.mean(); rec_score = rec_score / self.n_mod
        nelbo = rec_score + alpha * kl
        
        return nelbo, kl, rec_score
        
    
    def _sample_powerset(self):
        pass
    

'''
    Follow the work by Shi(2019), specifically for two modalities
    Use K = 1, which refers to time of repetition when sampling z for each modality [to be generalized]
'''
class DecoupledMMVAE(nn.Module):
    def __init__(self,
            encoders: Dict[str, nn.Module],
            decoders: Dict[str, nn.Module],
            latent_dim,
            alpha,
            score_fns: Dict[str, callable],
        ):
        super(DecoupledMMVAE, self).__init__()
        self.encoders, self.decoders = encoders, decoders
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.score_fns = score_fns
        
        
    def forward(self, x):
        kl, rec = self._forward_impl(x)
        nelbo = rec + self.alpha * kl
        return nelbo, kl, rec
        
    def _forward_impl(self, inputs: Dict[str, torch.Tensor]):
        '''
        Pipeline:
        - generate latent repr for each modality: z1 = Enc1(x1); z2 = Enc2(x2)
        - compute kl divergence w.r.t to prior for each z
        - reconstruct each modality conditioned on all zs 
        '''
        latents = []
        mod_names = sorted(inputs.keys())
        kl = .0
        
        # generate latents
        for mod in mod_names:
            x = inputs[mod]
            param = self.encoders[mod](x)
            mu, logvar = torch.split(param, self.latent_dim, dim=-1)
            kl += kl_standard_gaussian(mu, logvar)
            latents.append(reparameterize(mu, logvar))
        latents = torch.cat([z.unsqueeze(0) for z in latents], dim=0) # (mod, batch_size, latent_dim)
        
        # reconstruction
        rec = .0
        for mod_a in mod_names:
            x = inputs[mod_a]
            for z in latents:
                x_hat = self.decoders[mod_a](z)
                rec += self.score_fns[mod_a](x, x_hat)
        
        return kl, rec

        
        
    
        
        
    
        
        
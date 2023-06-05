from collections import OrderedDict
import torch
from torch import Tensor
import torch.nn as nn
from typing import Dict, Scalar
from ..vae.vae import (
    kl_standard_gaussian,
    reparameterize
)
from itertools import chain


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
            score_fns: Dict[str, callable],
        ):
        super(DecoupledMMVAE, self).__init__()
        self.encoders, self.decoders = nn.ModuleDict(encoders), nn.ModuleDict(decoders)
        self.latent_dim = latent_dim
        self.score_fns = score_fns
        
        
        self._config_mod_ae()
        
    @property
    def trainable_components(self):
        return [*self.encoders.values(), *self.decoders.values()]  
    @property
    def _mod_names(self):
        return sorted(self.encoders.keys())
    
    
    def generate(self, x: Dict[str, torch.Tensor], random_latent=False, rsample=1):
        z = self._compute_joint_z(x, random_latent, rsample)
        rec = {}
        for mod in self._mod_names:
            rec[mod] = self.decoders[mod](z)
        return rec
    
    def _compute_joint_z(self, x, random_latent, rsample=1):
        assert isinstance(x, dict)
        param = []
        for mod, mod_x in x.items():
            # batched mu and logvar
            mu, logvar = torch.split(
                self.encoders[mod](mod_x), self.latent_dim, dim=-1
            )
            param.append((mu, logvar))

        latents = []
        if random_latent:
            for mu, logvar in param:
                latents.append(reparameterize(mu, logvar, rsample))
        else:
            for mu, _ in param:
                latents.append(mu)
        z = torch.cat([z.unsqueeze(0) for z in latents], dim=0) # (mod, batch, ...)

        return torch.mean(z, dim=0)
    
    def _set_rec_loss_penalty(self, penalty: Dict[str, float]):
        if penalty is None:
            penalty = {}
            for mod in self._mod_names:
                penalty[mod] = 1.
        else:
            for mod in self._mod_names:
                if mod not in penalty.keys():
                    penalty[mod] = 1.
        return penalty
                
        
    def forward(self, x, alpha, rsample=1, mod_penalty: Dict[str, float]=None, verbose=False):
        mod_penalty = self._set_rec_loss_penalty(mod_penalty)
        
        if verbose:
            kl, rec, verbose_output = self._forward_impl(x, rsample, mod_penalty, verbose)
        else:
            kl, rec = self._forward_impl(x, rsample, mod_penalty, verbose)
        nelbo = rec + alpha * kl
        
        if verbose:
            return nelbo, kl, rec, verbose_output
        else:
            return nelbo, kl, rec
        
    def _forward_impl(self, inputs: Dict[str, torch.Tensor], rsample, mod_penalty, verbose=False):
        '''
        Pipeline:
        - generate latent repr for each modality: z1 = Enc1(x1); z2 = Enc2(x2)
        - compute kl divergence w.r.t to prior for each z
        - reconstruct each modality conditioned on all zs 
        '''
        latents = []
        kl = .0
        
        # generate latents
        for mod in self._mod_names:
            x = inputs[mod]
            param = self.encoders[mod](x)
            mu, logvar = torch.split(param, self.latent_dim, dim=-1)
            kl += kl_standard_gaussian(mu, logvar, reduction='mean')
            latents.append(reparameterize(mu, logvar, rsample))
        
        latents = torch.cat([z.unsqueeze(0) for z in latents], dim=0) # (mod, batch_size, latent_dim)
        
        # reconstruction
        rec = .0
        verbose_rec_output = {}
        for mod_a in self._mod_names:
            x = inputs[mod_a]
            mod_rec = .0
            for mod_b, z in zip(self._mod_names, latents):
                x_hat = self.decoders[mod_a](z)
                loss_a_b = self.score_fns[mod_a](x_hat, x)
                mod_rec += loss_a_b
                if verbose:
                    verbose_rec_output[f'{mod_a}|{mod_b}'] = loss_a_b
            rec += mod_penalty[mod_a] * mod_rec
        
        if verbose:
            return kl, rec, verbose_rec_output
        else:
            return kl, rec
    
    def _config_mod_ae(self):
        for mod in self._mod_names:
            ae = _mod_ae(
                encoder=self.encoders[mod],
                decoder=self.decoders[mod],
                latent_dim =self.latent_dim 
            )
            setattr(self, f'{mod}_ae', ae)

    
class _mod_ae(nn.Module):
    def __init__(self,
        encoder,
        decoder,
        latent_dim    
    ):
        super(_mod_ae, self).__init__()
        self.encoder, self.decoder = encoder, decoder
        self.latent_dim = latent_dim
        
    def forward(self, x, random_latent=False):
        mu, logvar = torch.split(
            self.encoder(x), self.latent_dim, dim=-1
        )
        if random_latent:
            z = reparameterize(mu, logvar)
        else:
            z = mu
        x_hat = self.decoder(z)
        return x_hat
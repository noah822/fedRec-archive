from collections import OrderedDict
import torch
from torch import Tensor
import torch.nn as nn
from typing import Any, Dict
from ..vae.vae import (
    kl_standard_gaussian,
    reparameterize
)
from itertools import chain
import torch.distributions as D


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
    link: https://arxiv.org/abs/1911.03393
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
    @property
    def _num_mod(self):
        return len(self.encoders)
    
    @property
    def latent_prior(self):
        mu = torch.zeros(self.latent_dim)
        std = torch.ones(self.latent_dim)
        _isotropic_normal = D.Normal(mu, std)
        return _isotropic_normal
    


    def reconstruct(self, x: Dict[str, torch.Tensor], random_latent=False, rsample=1):
        '''
            Take averge of q(x|z) if multiple modalities are present
        '''
        # z = self._compute_joint_z(x, random_latent, rsample)
        rec = {mod : [] for mod in self._mod_names}
        with torch.no_grad():
            for mod_i, mod_x in x.items():
                param = self.encoders[mod_i](mod_x)
                mu, logvar = torch.split(param, self.latent_dim, dim=-1)
                if random_latent:
                    latent = reparameterize(mu, logvar,rsample)
                else:
                    latent = mu
                for mod_j in self._mod_names:
                    rec[mod_j].append(
                        self.decoders[mod_j](latent)
                    )
        for mod, x_hat in rec.items():
            x_hat = torch.stack(x_hat).mean(dim=0)
            rec[mod] = x_hat
        return rec
    
    def generate(self, num_sample=1, device='cuda'):
        prior: torch.distributions = self.latent_prior
        z = prior.sample(torch.Size([num_sample])).to(device)
        samples = {}
        
        for mod in self._mod_names:
            samples[mod] = self.decoders[mod](z)
        return samples

    '''
        Potentially to be deprecated
    '''
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
    
    def _compute_z_score_cond_mod(self, z, posteriors: Dict[str, callable]):
        '''
        Compute the explicit value of q(z|x_m) given current parameters
        Args:
        - z: (mod, batch, latent_dim), sorted according to mod name
          current estimate of q(z|x), parameterized by output of encoders
        - posteriors: q(z|x) for each modality
          value for this dict should impl a callable interface, 
          posterior(z) should return the probabilty for z element-wisely
        Return:
        - return as (num_mod, num_mod) matrix, (mod, mod, batch)
        indexing:
          post_score[i,j] = q(z_j | x_i)
        '''
        post_score = [[None for _ in range(self._num_mod)] for _ in range(self._num_mod)]
        for i, mod_i in enumerate(self._mod_names):
            qz_x_i = posteriors[mod_i]
            for j, _ in enumerate(self._mod_names):
                post_score[i][j] = qz_x_i(z[j])
                
        # post_score = torch.cat([torch.cat(y, dim=0) for y in post_score], dim=0)
        return post_score
    
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
        
    def _forward_impl(self,
            inputs: Dict[str, torch.Tensor],
            rsample,
            mod_penalty,
            iw_cross_mod=True,
            verbose=False
        ):
        '''
        Pipeline:
        - generate latent repr for each modality: z1 = Enc1(x1); z2 = Enc2(x2)
        - compute kl divergence w.r.t to prior for each z
        - reconstruct each modality conditioned on each z_m
        - compute importance sampling weight 
          When back prop gradient for encoder of mod i:
             1. sampling latent z_j from all other mod j
             2. compute w_j = q(z_j | x_i) / q(z_j | x_j) -> Estimate of p(x_j | z_i) for all other mod j
             3. penalize rec loss of mod_j by w_j 
        '''
        latents = []
        posteriors = {}
        kl = .0
        
        # generate latents
        for mod in self._mod_names:
            x = inputs[mod]
            param = self.encoders[mod](x)
            mu, logvar = torch.split(param, self.latent_dim, dim=-1)
            posteriors[mod] = _normal_callable_interface(mu, logvar)
            
            kl += kl_standard_gaussian(mu, logvar, reduction='mean')
            latents.append(reparameterize(mu, logvar, rsample))
        
        latents = torch.cat([z.unsqueeze(0) for z in latents], dim=0) # (mod, batch_size, latent_dim)
        
        post_score = None
        if iw_cross_mod:
            post_score = self._compute_z_score_cond_mod(latents, posteriors)
            
        # reconstruction
        rec = .0
        verbose_rec_output = {}

        if iw_cross_mod:
            rec, verbose_rec_output = self._cross_mod_rec_iw(
                inputs, latents, post_score,
                mod_penalty
            )
        else:
            rec, verbose_rec_output = self._cross_mod_rec_vanilla(
                inputs, latents, mod_penalty
            )
        
        if verbose:
            return kl, rec, verbose_rec_output
        else:
            return kl, rec
    
    '''
        possible impl of cross reconstrution loss
        Naive Approach:
        directly calculate p(x_j | z_i), needs extra decoder pass
        
        IW Sampling:
        approximate p(x_j | z_i) by the expectation of 
        p(x_j | z_j) weighted by q(z_i | x_i) / q(z_j | x_j) 
        TODO: possible resample 
        
        Return
        - 
    
    '''
    def _cross_mod_rec_iw(self,
            inputs: Dict[str, torch.tensor],
            latents, post_score,
            mod_penalty
        ):
        verbose_rec_output = {}
        rec = .0
        for i, mod_i in enumerate(self._mod_names):
            mod_rec = .0        # estimate of p(x_j | z_i) for j != i
            for j, (mod_j, z) in enumerate(zip(self._mod_names, latents)):
                x = inputs[mod_j]
                if mod_i == mod_j:
                    latent = z
                else:
                    latent = z.detach()
                x_hat = self.decoders[mod_j](latent)
                
                loss_i_j = self.score_fns[mod_j](x_hat, x)
                if mod_i != mod_j:
                    w = post_score[i][j] / (post_score[j][j].detach())
                else:
                    w = 1.
                if i == j:
                    verbose_rec_output[f'{mod_i}|{mod_j}'] = loss_i_j.mean()
                mod_rec += (w * loss_i_j).mean()
            
            rec += mod_rec * mod_penalty[mod_i]
        return rec, verbose_rec_output
    
    def _cross_mod_rec_vanilla(self, inputs: Dict[str, torch.tensor], latents, mod_penalty):
        
        verbose_rec_output = {}
        rec = .0
        for mod_i in self._mod_names:
            x = inputs[mod_i]
            mod_rec = .0
            for mod_j, z in zip(self._mod_names, latents):
                x_hat = self.decoders[mod_i](z)
                loss_i_j = self.score_fns[mod_i](x_hat, x)
                mod_rec += loss_i_j
            
                verbose_rec_output[f'{mod_i}|{mod_j}'] = loss_i_j.mean()
            rec += mod_penalty[mod_i] * mod_rec

        return rec, verbose_rec_output
    
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

class _normal_callable_interface:
    def __init__(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        self._normal = D.Normal(mu, std)
    def __call__(self, x):
        log_prob = self._normal.log_prob(x).sum(dim=-1)
        return torch.exp(log_prob)
        
        
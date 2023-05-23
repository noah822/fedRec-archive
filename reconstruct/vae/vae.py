import torch 
import torch.nn as nn


def kl_normal(qm, qv, pm, pv):
    """
    Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
    sum over the last dimension

    Args:
        qm: tensor: (batch, dim): q mean
        qv: tensor: (batch, dim): q variance
        pm: tensor: (batch, dim): p mean
        pv: tensor: (batch, dim): p variance

    Return:
        kl: tensor: (batch,): kl between each sample
    """
    
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    return kl

def kl_standard_gaussian(mu, logvar):
    return kl_normal(
        mu, torch.exp(logvar),
        torch.zeros_like(mu), torch.ones_like(logvar)
    )

def reparameterize(mu, logvar):
    '''
    Reparameterization trick to produce differentiable posterior
    mu: (batch, latent_dim)
    logvar: (batch, latent_dim)
    '''
    
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(mu)
    
    return mu + std * eps



class VAE(nn.Module):
    def __init__(
        self, encoder, decoder,
        latent_dim,
        prior_mean=None, prior_logvar=None,
        rec_loss_fn='mse'
        ):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        self.latent_dim = latent_dim
        
        if prior_mean is None:
            prior_mean = nn.Parameter()
        self.prior = {
            'mean' : prior_mean, 
            'logvar' : prior_logvar
        }
        
        self.criterion = None
        self.loss = rec_loss_fn
        
        if rec_loss_fn == 'mse':
            self.criterion = nn.MSELoss()
        elif rec_loss_fn == 'bce':
            self.criterion = nn.BCELoss(reduction='sum')
            
    def sample(self, batch_size=10):
        z = torch.randn(batch_size, self.latent_dim)
        recon = torch.sigmoid(self.decoder(z))
        return recon
    
    def forward(self, x, alpha=1.0):
        kl, rec = self._foward_impl(x)
        nelbo = rec + alpha * kl
        return nelbo, kl, rec
    
    def _reparameterize(self, mu, logvar):
        '''
            mu: (batch, latent_dim)
            logvar: (batch, latent_dim)
        '''
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu) 
        return mu + std * eps
    
    
    def _foward_impl(self, x):
        
        # generate params for p(z|x), approximated by q(z|x) -> KL divergence
        mu, logvar = torch.split(self.encoder(x), self.latent_dim, dim=-1)
        z = self._reparameterize(mu, logvar)
        
        kl = kl_normal(
            mu, torch.exp(logvar),
            torch.zeros_like(mu), torch.ones_like(logvar)
        ).sum()
        
        # compute p(x | z) -> reconstruction loss
        x_hat = self.decoder(z)
        
        if self.loss == 'bce':
            x_hat = torch.sigmoid(x_hat)
        
        rec = self.criterion(x_hat, x)
        
        return kl, rec
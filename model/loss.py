from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCE(nn.Module):
    '''
        main reference: https://github.com/sthalles/SimCLR/blob/master/simclr.py
        
        pipleline:
        >>> criterion = InfoNCE(reduction='mean')
        >>> view1_embed, view2_embed = model(x) # embeds before normalize
        >>> loss = criterion(view1_embed, view2_embed)
    
    '''
    
    def __init__(self,
                 temperature: float=1.,
                 alpha: float=1.,
                 reduction='mean'):
        super(InfoNCE, self).__init__()

        self.reduction = reduction
        self._alpha = alpha
        self._overriding_alpha = None

        self._temperature = temperature
        self._overriding_temp = None

    @property
    def alpha(self):
        if self._overriding_alpha is not None:
            return self._overriding_alpha
        elif self._alpha is not None:
            return self._alpha
        else:
            return None
    
    @property
    def temperature(self):
        if self._overriding_temp is not None:
            return self._overriding_temp
        elif self._temperature is not None:
            return self._temperature
        else:
            return None

        
    def forward(self,
                view1: torch.Tensor,
                view2: torch.Tensor,
                negative_aug1: torch.Tensor=None,
                negative_aug2: torch.Tensor=None,
                alpha: float=None,
                temperature: float=None,
                cross_view_only: bool=True
                ):
        '''
            - negative_aug: extra negative samples used to compute contrastive loss 
            Logit is computed as following:
            if only one negative augmentation is specified:
                [abused-notations] p -> postive; n -> negative; u -> augmented negative
                    logit = <p, p> / (<p, n> + eta * <p, u>)
            if two negative augmentation is specified:
                [abused-notations]
                p1, p2 -> paired postive from view1 & view2
                n1, n2 -> negatives from view1 & view2
                u1, u2 -> augmented negatives for view1 & view2
                
                logit[for p1] = <p1, p2> / (<p1, n2> + eta * <p1, u2>)
                logit[for p2] = <p2, p1> / (<p2, n1> + eta * <p2, u1>)
            - cross_view_only:
              if set to true,
              negative sample will only come from the other batch of sample, i.e other view
              
              usually used in cross-modal setting,
              see CLIP paper https://arxiv.org/pdf/2103.00020.pdf

        '''
        self._overriding_alpha = alpha
        self._overriding_temp = temperature

        if cross_view_only:
            loss = InfoNCE._inter_batch_fwd_impl(
                view1, view2,
                self.temperature,
                negative_aug1, 
                negative_aug2,
                self.alpha
            )
        else: # intra-batch + inter-batch
            loss = InfoNCE._default_forward_impl(
                        view1, view2,
                        self.temperature,
                        negative_aug1,
                        negative_aug2,
                        self.alpha
                    )
        return loss
    
    
    @staticmethod
    def _default_forward_impl(
            view1: torch.Tensor,
            view2: torch.Tensor,
            temperature: float,
            negative_aug1: torch.Tensor=None,
            negative_aug2: torch.Tensor=None,
            alpha: float= .0
        ):
        n_view = 2
        batch_size = view1.shape[0]
        device = view1.device
        
        view1 = F.normalize(view1, dim=-1)
        view2 = F.normalize(view2, dim=-1)
        
        
        labels = torch.cat([torch.arange(batch_size) for _ in range(n_view)]).to(device) # (bzs, n_view * embed_dim)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1))
        
        sample_size = labels.shape[0]
        
        mask = torch.eye(sample_size ,dtype=torch.bool).to(device)
        
        labels = labels[~mask]
        
        feature = torch.concat([view1, view2], dim=0)
        logits_matrix= (feature @ feature.T)[~mask]
        
        positive = logits_matrix[labels.bool()].view(sample_size, -1) # (2N, ...)
        negative = logits_matrix[~labels.bool()].view(sample_size, -1)

        # use negative augmentation
        if negative_aug1 is not None:
            if negative_aug2 is None:
                negative_aug2 = negative_aug1
            
            extra_penalty1 = view1 @ negative_aug2.T # (N, M)
            extra_penalty2 = view2 @ negative_aug1.T # (N, M)

            extra_penalty = torch.concat([extra_penalty1, extra_penalty2], dim=0)
            negative = torch.concat([negative, alpha*extra_penalty], dim=-1)
            
        logits_matrix = torch.concat([positive, negative], dim=-1) / temperature
        labels = torch.zeros(sample_size, dtype=torch.long).to(device)

        loss = F.cross_entropy(logits_matrix + 1e-6, labels)
        
        return loss


    
    @staticmethod
    def _inter_batch_fwd_impl(
                    view1: torch.Tensor,
                    view2: torch.Tensor,
                    temperature: float=1.,
                    negative_aug1: torch.Tensor=None,
                    negative_aug2: torch.Tensor=None,
                    alpha: float=1.
                    ):
            
            device = view1.device
            batch_size = view1.shape[0]
            target = torch.arange(batch_size).to(device)
            
            normed_view1 = F.normalize(view1, dim=-1)
            normed_view2 = F.normalize(view2, dim=-1)

            # compute view1 logits  
            _view2 = view2
            if negative_aug2 is not None:
                _view2 = torch.concat([view2, alpha * negative_aug2], dim=0)
            view1_logits = (normed_view1 @ F.normalize(_view2, dim=-1).T) / temperature
            view1_loss = F.cross_entropy(view1_logits + 1e-6, target)

            # compute view2 logits
            _view1 = view1
            if negative_aug1 is not None:
                _view1 = torch.concat([view1, alpha * negative_aug1], dim=0)
            view2_logits = (normed_view2 @ F.normalize(_view1, dim=-1).T) / temperature
            view2_loss = F.cross_entropy(view2_logits + 1e-6, target)

            loss = (view1_loss + view2_loss) / 2

            return loss
    
    
def _infer_device(x):
    if isinstance(x, torch.Tensor):
        return x.device
    elif isinstance(x, nn.Module):
        return next(x.parameters()).device

def _off_diagnoal(x: torch.Tensor, reduction='sum'):
    device = _infer_device(x)
    h, w = x.shape[-2:]
    assert h == w
    masked_x = x * (1 - torch.eye(h).to(device))
    masked_x = masked_x.view(*x.shape[:-2], -1)
    
    if reduction == 'sum':
        return masked_x.sum(-1)
    elif reduction == 'mean':
        return masked_x.mean(-1)

def _on_diagnoal(x: torch.Tensor, reduction='sum'):
    device = _infer_device(x)
    h, w = x.shape[-2:]
    assert h == w
    masked_x = x * torch.eye(h).to(device)
    masked_x = masked_x.view(*x.shape[:-2], -1)
    
    if reduction == 'sum':
        return masked_x.sum(-1)
    elif reduction == 'mean':
        return masked_x.mean(-1)

'''
    BarlowTwins
    view1 \       / proj1
           encoder         -> compute correlation matrix of feature
    view2 /       \ proj2
    
    Objective:  
        push off-diagnol elements to 0 -> decorrelated
        push on-diagonal elements to 1 
'''
class BarlowTwins(nn.Module):
    def __init__(self, reduction='sum'):
        super(BarlowTwins, self).__init__()
        assert reduction in ['sum', 'mean', 'none']
        self.reduction = reduction
    
    def forward(self, view1: torch.Tensor, view2: torch.Tensor, eps):
        '''
        Args:
        - view1/view2: (batch, embed_dim)
        '''
        B, D = view1.shape
        view1 = (view1 - view1.mean(0)) / view1.std(0)
        view2 = (view2 - view2.mean(0)) / view2.std(0)
        
        feature_corr = (1/B) * view1.T @ view2 # (embed_dim, embed_dim)
        
        device = feature_corr.device
        
        feature_diff = (feature_corr - torch.eye(D).to(device)).pow(2)
        
        on_diag = _on_diagnoal(feature_diff, device, reduction=self.reduction).sum()
        off_diag = _off_diagnoal(feature_diff, device, reduction=self.reduction).sum()
        
        return on_diag + eps * off_diag
        
        

class VicReg(nn.Module):
    def __init__(self, 
        sim_coef,
        std_coef,
        cov_coef,
        eps=1e-3             
    ):
        super(VicReg, self).__init__()
        self.sim_coef, self.std_coef, self.cov_coef = sim_coef, std_coef, cov_coef
        self.eps = eps
    
    def forward(self, x, y):
        '''
        main reference: https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
    
        Args:
        - view1/view2: (batch, latent_dim)
        '''
        B, D = x.shape
        sim_loss = F.mse_loss(x, y)
        
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        
        std_x = torch.sqrt(x.var(dim=0) + self.eps)
        std_y = torch.sqrt(y.var(dim=0) + self.eps)
        std_loss = torch.mean(F.relu(1-std_x)) / 2 + \
                   torch.mean(F.relu(1-std_y)) / 2
                   
        cov_x = (x.T @ x) / B
        cov_y = (y.T @ y) / B
        cov_loss = (_off_diagnoal(cov_x.pow(2)) + _off_diagnoal(cov_y.pow(2))) / (2 * D)
        
        loss = self.sim_coef * sim_loss + \
               self.std_coef * std_loss + \
               self.cov_coef * cov_loss 
        return loss 
        
        
        
        
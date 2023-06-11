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
    
    def __init__(self, temperature, reduction='mean'):
        super(InfoNCE, self).__init__()
        self.reduction = reduction
        self.temperature = temperature
        
    def forward(self, view1, view2):
        logits, labels = InfoNCE._forward_impl(view1, view2, self.temperature)
        return F.cross_entropy(logits, labels)
    
    
    @staticmethod
    def _forward_impl(view1, view2, temperature):
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
        
        positive = logits_matrix[labels.bool()].view(sample_size, -1)
        negative = logits_matrix[~labels.bool()].view(sample_size, -1)
        
        logits_matrix = torch.concat([positive, negative], dim=-1) / temperature
        labels = torch.zeros(sample_size, dtype=torch.long).to(device)
        
        return logits_matrix, labels

def _off_diagnoal(x: torch.tensor, device, reduction='sum'):
    h, w = x.shape[-2:]
    assert h == w
    masked_x = x * (1 - torch.eye(h).to(device))
    masked_x = masked_x.view(*x.shape[:-2], -1)
    
    if reduction == 'sum':
        return masked_x.sum(-1)
    elif reduction == 'mean':
        return masked_x.mean(-1)

def _on_diagnoal(x: torch.tensor, device, reduction='sum'):
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
class BarlowTwins:
    def __init__(self, reduction='sum'):
        assert reduction in ['sum', 'mean', 'none']
        self.reduction = reduction
    
    def __call__(self, view1: torch.tensor, view2: torch.tensor, eps):
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
        
        

class VicReg:
    pass
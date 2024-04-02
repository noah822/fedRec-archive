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

    

        
        
class StandardPipeline(nn.Module):
    def __init__(self, 
                 backbone,
                 proj_head):
        super(StandardPipeline, self).__init__()
        self.backbone, self.proj_head = backbone, proj_head
    def forward(self, x, return_embed=False):
        embed = self.backbone(x)
        projection = self.proj_head(embed)
        if not return_embed:
            return projection
        else:
            return projection, embed
        
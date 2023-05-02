import torch
import torch.nn as nn
import torch.nn.functional as F

class BYOL:
    def __init__(self,
        online_network,
        target_network,
        predictor,
        epsilon,
        device='cpu'
    ):
        self.online_network = online_network.to(device)
        self.target_network = target_network.to(device)
        self.predictor = predictor.to(device)
        self.epsilon = epsilon
        self.device = device
    
    
    @torch.no_grad()
    def _update_target_network_params(self):
        for online_param, target_param in zip(
            self.online_network.parameters(), self.target_network.parameters()
        ):
            target_param.data = online_param.data * self.epsilon + target_param.data * (1 - self.epsilon)
    
    
    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)
    
        
    def train(self, dataloader, optimizer, n_epoch):
        
        for _ in range(n_epoch):
            for (view1, view2) in dataloader:
                loss = self._update(view1, view2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                self._update_target_network_params()
                
                
    
    def _update(self, batch_view_1, batch_view_2):
        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_1)
            targets_to_view_1 = self.target_network(batch_view_2)

        loss = BYOL.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += BYOL.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()
            
        
        
        
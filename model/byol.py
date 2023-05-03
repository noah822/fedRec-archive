import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class BYOL:
    def __init__(self,
        backbone,
        inplanes,
        pred_hidden_dim,
        epsilon,
        proj_dim=None,
        proj_hidden_dim=None,
        user_proj=True,
        head_layer_name='fc',
        prune_backbone=True,
        use_1d_bn=True,
        device='cpu'
    ):
        '''
        
        online:  backbone -> projection -> predictor
        target:  backbone -> projection               
        
        projection: MLP with one hidden layer
        Args:
        - inplanes: output dimension of bachbone
        - proj_hidden_dim: hidden dimension of projection layer
        
        predictor: MLP with one hidden layer (proj_dim -> hidden -> proj_dim)
        Args:
        - pred_hidden_dim: hidden dimension of predictor layer
        
        '''
        if prune_backbone:
            assert head_layer_name is not None
            backbone = BYOL._remove_classifier(backbone, head_layer_name)
        if user_proj:
            assert proj_hidden_dim is not None and proj_dim is not None
            projection = self._make_mlp(
                inplanes, proj_hidden_dim, 
            )
            online_network = nn.Sequential(
                backbone, projection
            )
        else:
            online_network = backbone
        

        self.online_network = online_network.to(device)
        self.target_network = copy.deepcopy(online_network).to(device)
        self.predictor = self._make_mlp(
            proj_dim, pred_hidden_dim, proj_dim, use_1d_bn
        )
        
        self.epsilon = epsilon
        self.device = device
        
        
    @staticmethod
    def _remove_classifier(model, layer_name):
        setattr(model, layer_name, nn.Identity())
        return model
        
    def _make_mlp(self, in_dim, hidden_dim, out_dim, use_1d_bn=True):
        if use_1d_bn:
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
    
    @torch.no_grad()
    def _update_target_network_params(self):
        for online_param, target_param in zip(
            self.online_network.parameters(), self.target_network.parameters()
        ):
            target_param.data = online_param.data * (1-self.epsilon) + target_param.data * self.epsilon
    
    
    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)
    
        
    def train(self, dataloader, optimizer, optimizer_config, n_epoch):
        _optimizer = optimizer(
            list(self.online_network.parameters()) + list(self.predictor.parameters()),
            **optimizer_config
        )
        
        self.optimizer = _optimizer
        
        
        for _ in range(n_epoch):
            for (view1, view2) in dataloader:
                view1 = view1.to(self.device); view2 = view2.to(self.device)
                loss = self._update(view1, view2)
                _optimizer.zero_grad()
                loss.backward()
                _optimizer.step()
                
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
    
    @property
    def feature_extractor(self):
        return self.online_network
    
    def save_state_dict(self, save_path='./byol.pt'):
        torch.save({
            'online_network': self.online_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'predictor': self.predictor.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, save_path)
    
    def load_state_dict(self, load_path='./byol.pt'):
        ckp = torch.load(load_path)
        self.online_network.load_state_dict(ckp['online_network'])
        self.target_network.load_state_dict(ckp['target_network'])
        self.predictor.load_state_dict(ckp['predictor'])
        self.optimizer.load_state_dict(ckp['optimizer'])
            
        
        
        
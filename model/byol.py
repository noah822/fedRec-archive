import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm.notebook import tqdm, trange
from torch.utils.tensorboard import SummaryWriter

from ._utils import (
    _remove_component
)

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
            backbone = _remove_component(backbone, head_layer_name)
        
        if user_proj:
            assert proj_hidden_dim is not None and proj_dim is not None
            projection = self._make_mlp(
                inplanes, proj_hidden_dim, proj_dim
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
        ).to(device)
        
        self.epsilon = epsilon
        self.device = device
        
        
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
    
        
    def train(
            self, augmentation, dataloader,
            optimizer, optimizer_config,
            scheduler, scheduler_config,
            n_epoch,
            save_path,
            use_tqdm=False,
            use_tensorboard=False,
            tensorboard_path=None
        ):
        
        writer = None
        if use_tensorboard:
            assert tensorboard_path is not None
            writer = SummaryWriter(log_dir=tensorboard_path)
        _optimizer = optimizer(
            list(self.online_network.parameters()) + list(self.predictor.parameters()),
            **optimizer_config
        )
        _scheduler = scheduler(
            _optimizer, 
            **scheduler_config
        )
        
        self.optimizer = _optimizer
        
        epoch_iterator = range(n_epoch)
        if use_tqdm:
            epoch_iterator = trange(n_epoch)
        
        
        for epoch in epoch_iterator:
            running_loss = .0
            data_iterator = dataloader
            if use_tqdm:
                data_iterator = tqdm(dataloader)
            for inputs, _ in data_iterator:
                
                view1, view2 = augmentation(inputs.to(self.device))
                loss = self._update(view1, view2)
                _optimizer.zero_grad()
                loss.backward()
                _optimizer.step()
                
                self._update_target_network_params()
                if use_tqdm:
                    data_iterator.set_postfix(loss=loss.item())
                running_loss += loss.item()
            
            _scheduler.step()
            
            avg_loss = running_loss / len(dataloader)
            if use_tqdm:
                epoch_iterator.set_postfix(avg_loss=avg_loss)
                
            if writer is not None:
                writer.add_scalar("Loss/train", avg_loss, epoch)
        
        self.save_state_dict(save_path)
    

        writer.close()
                
                
    
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
            'predictor': self.predictor.state_dict()
        }, save_path)
    
    def load_state_dict(self, load_path='./byol.pt'):
        ckp = torch.load(load_path)
        self.online_network.load_state_dict(ckp['online_network'])
        self.target_network.load_state_dict(ckp['target_network'])
        self.predictor.load_state_dict(ckp['predictor'])
            
        
        
        
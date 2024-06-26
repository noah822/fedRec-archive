from typing import Any
from tqdm.notebook import trange, tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List

def train(
    model, dataloader,
    optimizer, scheduler, 
    criterion,
    augmentation,
    n_epoch,
    device,
    merge_batch=True,
    use_mp=False,
    use_tqdm=False,
    use_tensorboard=False,
    tensorboard_path=None,
    save_path=None
):
    def _forward_impl(model, view1, view2, merge_batch):
        if merge_batch:
            x = torch.concat([view1, view2], dim=0)
            x = model(x)
            view1_embed, view2_embed = torch.split(x, x.shape[0] // 2, dim=0)
        else:
            view1_embed = model(view1)
            view2_embed = model(view2)
        return view1_embed, view2_embed
            
    epoch_iterator = range(n_epoch)
    
    writer = None
    if use_tensorboard:
        assert tensorboard_path is not None
        writer = SummaryWriter(log_dir=tensorboard_path)
    
    if use_tqdm:
        epoch_iterator = trange(n_epoch)
        
        
    for epoch in epoch_iterator:
        data_iterator = dataloader
        running_loss = .0
        if use_tqdm:
            data_iterator = tqdm(dataloader)
        
        for inputs, _ in data_iterator:
            inputs = inputs.to(device)
            view1, view2 = augmentation(inputs)
            if use_mp:
                with torch.cuda.amp.autocast():
                    view1_embed, view2_embed = _forward_impl(
                        model, view1, view2, merge_batch
                    )
            else:
                view1_embed, view2_embed = _forward_impl(
                        model, view1, view2, merge_batch
                    )
            
            loss = criterion(view1_embed, view2_embed)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if use_tqdm:
                data_iterator.set_postfix(loss=loss.item())
        scheduler.step()
        
        avg_loss = running_loss / len(dataloader)
        if use_tqdm:
            epoch_iterator.set_postfix(avg_loss=avg_loss)
            
        if writer is not None:
            writer.add_scalar("Loss/train", avg_loss, epoch)
                
    torch.save({
        'model' : model.state_dict(),
        'optimizer' : optimizer.state_dict()
    }, save_path)

def val(model, dataloader, device, unpack_and_forward=None):
    acc = .0
    for inputs in dataloader:
        if unpack_and_forward is None:  
            x, label = inputs
            x = x.to(device); label = label.to(device)
            pred = model(x)
        else:
            pred = unpack_and_forward(model, inputs)
            label = inputs[-1].to(device) 
        acc += (torch.argmax(pred, dim=-1) == label).sum()
    
    return acc / len(dataloader.dataset)


import torch.nn.functional as F

def linear_prob(
        backbone, head,
        dataloader,
        optimizer, criterion,
        device,
        n_epoch,
        unpack_and_forward=None,
        use_tqdm=False,
        normalize=False
    ):
    
    
    epoch_iterator = range(n_epoch)
    if use_tqdm:
        epoch_iterator = trange(n_epoch)
    
    for _ in epoch_iterator:
        
        data_iterator = dataloader
        if use_tqdm:
            data_iterator = tqdm(dataloader)
            
        running_loss = .0
        for inputs in data_iterator:
            if unpack_and_forward is None:
                x, y = inputs
                x = x.to(device); y = y.to(device)
                with torch.no_grad():
                    feature = backbone(x).detach()
                    if normalize:
                        feature = F.normalize(feature, dim=-1)
                pred = head(feature)
                loss = criterion(pred, y)
            else:
                loss = unpack_and_forward(backbone, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if use_tqdm:
                data_iterator.set_postfix(loss=loss.item())
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
    
        if use_tqdm:
            epoch_iterator.set_postfix(avg_loss=avg_loss)
            
           
class Prompt:
    def __init__(self, keys):
        self.prompt = OrderedDict()
        self._keys = keys
        for k in self._keys:
            self.prompt[k] = .0
    def __add__(self, rhs):
        for k in self._keys:
            self.prompt[k] += rhs[k]
        return self
    def __iadd__(self, rhs):
        self = self + rhs
        return self
    def __truediv__(self, scalar):
        for k in self._keys:
            self.prompt[k] /= scalar
        return self
    
    def __idiv__(self, scalar):
        self = self / scalar
        return self
    def __repr__(self):
        return self.prompt.__repr__()
    
    def __getitem__(self, key):
        return self.prompt[key]

# TODO: custom writer
def vanilla_trainer(
    model, dataloader,
    optimizer,
    criterion,
    n_epoch,
    device, 
    unpack_and_forward: callable = None,
    do_autoencode=True,
    scheduler=None,
    post_bp_operation: callable = None,
    use_tqdm=True,
    custom_prompt=False,
    use_tensorboard=False,
    tb_write_list: callable = None,
    tensorboard_path='./runs',
    do_val=False,
    val_freq=5,
    valloader=None,
    save_path=None
    ):
    
    epoch_iterator = range(n_epoch)
    if use_tqdm:
        epoch_iterator = trange(n_epoch)
    if do_val:
        assert valloader is not None
    if custom_prompt:
        assert unpack_and_forward is not None
    
    writer = None
    if use_tensorboard:
        writer = SummaryWriter(log_dir=tensorboard_path)
    
    for epoch in epoch_iterator:
        data_iterator = dataloader
        if use_tqdm:
            data_iterator = tqdm(data_iterator)
        epoch_prompt = None
            
        for inputs in data_iterator:
            
            if unpack_and_forward is not None:
                prompt = None
                if not custom_prompt:
                    loss = unpack_and_forward(model, inputs)
                    prompt = OrderedDict({'loss' : loss.item()})
                else:
                    loss, prompt = unpack_and_forward(model, inputs)
                    
            else:
                x, label = inputs
                x = x.to(device); label = label.to(device)
                target = x if do_autoencode else label
                
                pred = model(x)
                if criterion is not None:
                    loss = criterion(pred, target)
                else:
                    loss = pred
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if post_bp_operation is not None:
                post_bp_operation(model, loss)
            
            if epoch_prompt is None:
                epoch_prompt = Prompt(prompt.keys())
            
            epoch_prompt += prompt
            
            if use_tqdm:
                if do_val and (epoch + 1) % val_freq == 0:
                    val_acc = val(model, valloader, device)
                    data_iterator.set_postfix(**prompt, val_acc=val_acc)
                    if writer is not None:
                        writer.add_scalar('Loss/val', val_acc, epoch)
                else:
                    data_iterator.set_postfix(**prompt)
        
        epoch_prompt = epoch_prompt / len(dataloader)
        
        if scheduler is not None:
            scheduler.step()
        
        if writer is not None:
            if tb_write_list is not None:
                assert unpack_and_forward is not None
                assert use_tensorboard is True
                for write_tuple in tb_write_list(epoch_prompt):
                    title, value = write_tuple
                    writer.add_scalar(title, value, epoch)
            else:
                writer.add_scalar('Loss/train', epoch_prompt['loss'], epoch)
        if use_tqdm:
            epoch_iterator.set_postfix(**epoch_prompt.prompt)
        
    if save_path is not None:
        torch.save({
            'model' : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict() if scheduler is not None else None 
        }, save_path)

import random
import numpy as np
class CustomBatchSampler:
    def __init__(self,
                 num_sample: int,
                 batch_size: int,
                 drop_last: bool=True):
        self.num_sample = num_sample
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.src_idcs = np.array(range(num_sample))

        # build shuffled batch idcs at init time
        self.shuffle()
    
    def shuffle(self) -> List[int]:
        # inplace shuffle
        random.shuffle(self.src_idcs)
        self.cur_idcs = self._proper_chunk(self.src_idcs)
    
    def _proper_chunk(self, idcs):
        if self.drop_last:
            integer_batch = int(self.num_sample / self.batch_size)
        else:
            integer_batch = int((self.num_sample + self.batch_size - 1) / self.batch_size)
        batched_idcs = [
            idcs[i*self.batch_size: (i+1)*self.batch_size] for i in range(integer_batch)
        ]
        return batched_idcs
    @property
    def drawed_batch(self):
        return self.cur_idcs
    def __iter__(self):
        return iter(self.cur_idcs)
    def __len__(self):
        return len(self.cur_idcs)
    
                
        
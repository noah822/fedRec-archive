from tqdm.notebook import trange, tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn

def train(
    model, dataloader,
    optimizer, scheduler, 
    criterion,
    augmentation,
    n_epoch,
    device,
    use_tqdm=False,
    use_tensorboard=False,
    tensorboard_path=None,
    save_path=None
):
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
            x = torch.concat([view1, view2], dim=0)
            x = model(x)
            view1_embed, view2_embed = torch.split(x, x.shape[0] // 2, dim=0)
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

def val(model, dataloader, device):
    acc = .0
    for input, label in dataloader:
        input = input.to(device); label = label.to(device)
        pred = model(input)
        acc += (torch.argmax(pred, dim=-1) == label).sum()
    
    return acc / len(dataloader.dataset)

def linear_prob(
        backbone, head,
        dataloader,
        optimizer, criterion,
        device,
        n_epoch,
        use_tqdm=False         
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
            x, y = inputs
            x = x.to(device); y = y.to(device)
            with torch.no_grad():
                feature = backbone(x).detach()
            pred = head(feature)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            data_iterator.set_postfix(loss=loss.item())
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        epoch_iterator.set_postfix(avg_loss=avg_loss)
            
            
        
            
        
import torchvision
import torch.nn as nn


def get_backbone(
    model_name='resnet18',
    resize_first_conv=True,
    remove_last_linear=True
):
    _available = ['resnet18', 'resnet34', 'resnet50']
    assert model_name in _available
    
    model = getattr(torchvision.models, model_name)(pretrained=False)
    
    if resize_first_conv:
        _default_first_conv_name = 'conv1'
        _default_first_mp_name = 'maxpool'
        '''
            follow the architecture choice in papar SimCLR
            Replace 7x7 conv with 3x3 conv, kernel_size=3, stride=1
            Reomove first maxpool layer
        '''
        _default_conv_3x3 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1,1), padding=1)
        setattr(model, _default_first_conv_name, _default_conv_3x3)
        setattr(model, _default_first_mp_name, nn.Identity())
    if remove_last_linear:
        _default_last_linear_name = 'fc'
        setattr(model, _default_last_linear_name, nn.Identity())
    return model
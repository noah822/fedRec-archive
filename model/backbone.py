import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class residual(nn.Module):
    def __init__(self, input_channels, output_channels,
                 kernel_size=3, stride=1,
                 use_1x1_conv=False):
        super(residual, self).__init__()
        
        padding = int((kernel_size-1)/2)
        self.conv1 = nn.Conv2d(input_channels, output_channels,
                               kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm2d(output_channels)
        
        self.conv2 = nn.Conv2d(output_channels, output_channels,
                               kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(output_channels)
        
        self.conv3 = None
        if use_1x1_conv:
            self.conv3 = nn.Conv2d(input_channels, output_channels,
                                   kernel_size=1, stride=stride)
    
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        
        if self.conv3 is not None:
            y += self.conv3(x)
        else: y += x
        
        return F.relu(y)
    
class res_block(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(res_block, self).__init__()
        use_1x1_conv = (input_channels != output_channels) or (stride != 1)
        self.layer1 = residual(input_channels, output_channels, 
                               kernel_size=3, stride=stride, 
                               use_1x1_conv=use_1x1_conv)
        
        self.layer2 = residual(output_channels, output_channels, 
                               kernel_size=3)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    
    
class TinyNet(nn.Module):
    def __init__(self, input_channel=1, output_dim=32):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channel, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3,stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(64*7*7, 512),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        out = self.classifier(self.flatten(x))
        return out
    
    
class AudioTinyNet(nn.Module):
    def __init__(self, input_channel=1, n_class=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channel, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3,stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(64*8*15, 512),
            nn.Dropout(0.25),
            nn.Linear(512, n_class)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        out = self.classifier(self.flatten(x))
        
        return out
    


class ResNet_18(nn.Module):
    def __init__(self, input_channels, n_class):
        super(ResNet_18, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, padding=3,
                               stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.res_blk1 = res_block(64, 128, stride=2)
        self.res_blk2 = res_block(128, 256, stride=2)
        self.res_blk3 = res_block(256, 512, stride=2)
        self.res_blk4 = res_block(512, 512, stride=2)
        self.ada_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, n_class)
            
            
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.res_blk1(x)
        x = self.res_blk2(x)
        x = self.res_blk3(x)
        x = self.res_blk4(x)
        x = self.flatten(self.ada_pool(x))
        x = self.fc(x)
        
        return x

    
    
# class AVNet(nn.Module):
#     def __init__(self, output_dim):
#         super(AVNet, self).__init__()
        
#         self.audio_net = ResNet_18(input_channels=1)
#         self.visual_net = ResNet_18(input_channels=3)
        
#         self.ada_pool_a = nn.AdaptiveAvgPool2d((1,1))
#         self.ada_pool_v = nn.AdaptiveAvgPool2d((1,1))
#         self.flatten_a = nn.Flatten()
#         self.flatten_v = nn.Flatten()
        
#         self.fc_a = nn.Linear(512, 128)
#         self.fc_v = nn.Linear(512, 128)
        
#         self.fc_ = nn.Linear(256, output_dim)
        
        
        
#     def forward(self, a, v):
#         a = self.audio_net(a)
#         v = self.visual_net(v)
        
#         a = self.flatten_a(self.ada_pool_a(a))
#         v = self.flatten_v(self.ada_pool_v(v))
        
#         embedding_a = self.fc_a(a); embedding_v = self.fc_v(v)
        
#         concat_embedding = torch.concat([
#             embedding_a, embedding_v
#         ], dim=1)
        
#         pred = self.fc_(concat_embedding)
        
        
        
#         return pred, embedding_a, embedding_v
    
import torch.nn as nn
import torch
from copy import deepcopy
from typing import List
# from networks import *


class _aligned_decoder(nn.Module):
    def __init__(
        self, input_shape,
        encoder: nn.Module,
        decoder: nn.Module
    ):  
        super(_aligned_decoder, self).__init__()
        self.requested_conv_shape = _aligned_decoder._register_conv_hook(
            input_shape, deepcopy(encoder)
        )
        self.requested_conv_shape.reverse()
        self.requested_conv_shape = \
            self.requested_conv_shape[1:] + [torch.Size(input_shape)]
        self.decoder = decoder
    
        
        
    def forward(self, x):
        return self._requested_fwd(x)
    
    
    '''
        current implementation only supports one-depth nn.Sequential unfolding
    '''
    def _requested_fwd(self, x):
        conv_layer_cnt = 0
        
        for mod in self.decoder.children():
            if isinstance(mod, nn.ConvTranspose2d):
                # get rid of batch and channel dimension
                x = mod(x, output_size=self.requested_conv_shape[conv_layer_cnt][-2:])
                conv_layer_cnt += 1
                
            # flatten out Sequential layer 
            elif isinstance(mod, nn.Sequential):
                for submod in mod.children():
                    if isinstance(submod, nn.ConvTranspose2d):
                        x = submod(x, output_size=self.requested_conv_shape[conv_layer_cnt][-2:])
                        conv_layer_cnt += 1
                    else:
                        x = submod(x)
            else:
                x = mod(x)
        return x
    
            
    @staticmethod    
    def _register_conv_hook(input_shape, encoder: nn.Module):
        conv_output_shape = []
        
        def hook(model, input, output):
            conv_output_shape.append(output.shape)
            
        for mod in encoder.children():
            if isinstance(mod, nn.Conv2d):
                mod.register_forward_hook(hook)
                
            if isinstance(mod, nn.Sequential):
                for submod in mod.children():
                    if isinstance(submod, nn.Conv2d):
                        submod.register_forward_hook(hook)
            
        
        # one time forward 
        encoder(torch.ones(input_shape))
        return conv_output_shape
                

def get_aligned_decoder(input_shape, encoder, decoder):
    return _aligned_decoder(input_shape, encoder, decoder)

# if __name__ == '__main__':
    
    # decoder = get_aligned_decoder(
    #     (32, 1, 256, 64),
    #     mnist_audio_encoder, 
    #     mnist_audio_decoder
    # )
    
    # print(decoder.requested_conv_shape)
    
    # # x = torch.randn(32, 1, 256, 64)
    # latent = torch.randn(32, 16)
    # rec = decoder(latent)
    # print(rec.shape)
    

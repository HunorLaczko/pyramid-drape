import torch
from torch import nn
from models.convnext_orig import ConvNeXt, CNBlock, CNBlockConfig
from typing import List


class VariationalEncoder(nn.Module):
    
    def __init__(self, block_setting: List[CNBlockConfig], kernel_size: int = 7, input_features: int = 16, device: str = 'cuda'):
        super(VariationalEncoder, self).__init__()

        self.encoder = ConvNeXt(block_setting=block_setting, kernel_size=kernel_size, input_features=input_features, device=device)

        layers = [CNBlock(block_setting[-1].input_channels, kernel_size=kernel_size) for _ in range(5)]
        self.mean_branch = nn.Sequential(*layers).to(device)
        layers = [CNBlock(block_setting[-1].input_channels, kernel_size=kernel_size) for _ in range(5)]
        self.var_branch = nn.Sequential(*layers).to(device)

        self.flatten_mean = nn.Flatten()
        self.flatten_var = nn.Flatten()


    def forward(self, inputs):
        x = inputs

        x = self.encoder(x)
        mean = self.mean_branch(x)
        log_var = self.var_branch(x)

        mean = self.flatten_mean(mean)
        log_var = self.flatten_var(log_var)

        return mean, log_var
        

class Encoder(nn.Module):
    
    def __init__(self, block_setting: CNBlockConfig, kernel_size: int = 7, activation='tanh', is_label=False, nr_of_labels=None, device='cuda'):
        super(Encoder, self).__init__()

        self.encoder = ConvNeXt(block_setting=block_setting, kernel_size=kernel_size, is_label=is_label, nr_of_labels=nr_of_labels, device=device)
        if activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise ValueError('Activation function not supported')


    def forward(self, inputs):
        if len(inputs) == 2:
            x, _ = inputs
        else:
            x = inputs

        x = self.encoder(x)
        x = self.act(x)

        if len(inputs) == 2:
            return x, inputs[1]
        else:
            return x


class Decoder(nn.Module):
    
    def __init__(self, block_setting: CNBlockConfig, kernel_size=7, is_label=False, nr_of_classes=None, device='cuda'):
        super(Decoder, self).__init__()

        self.is_label = is_label

        self.decoder = ConvNeXt(block_setting=block_setting, kernel_size=kernel_size, is_label=is_label, nr_of_labels=nr_of_classes, device=device, reversed=True)
            
        if self.is_label:
            self.softmax = nn.Softmax(dim=1)


    def forward(self, inputs):
        if len(inputs) == 2:
            x, mask = inputs
        else:
            x = inputs

        x = self.decoder(x)

        if self.is_label:
            x = x * mask[:,None,:,:]

            background = torch.ones((mask.shape[0],1,512,256), device=mask.get_device()) * (1 - mask[:,None,:,:])
            x = torch.concat((background, x), dim=1)
            x = self.softmax(x)

        if len(inputs) == 2:
            if len(mask.shape) == 2:
                x = x * mask[None,None,:,:]
            else:
                x = x * mask[:,None,:,:]

        return x
    

class AutoEncoder(nn.Module):
    
    def __init__(self, config):
        super(AutoEncoder, self).__init__()

        self.encoder : Encoder = Encoder(config['encoder_block_settings'], config['kernel_size'], config['activation'], is_label=config['is_label'], device=config['device'], nr_of_labels=7)
        self.decoder : Decoder = Decoder(config['decoder_block_settings'], kernel_size=config['kernel_size'], is_label=config['is_label'], device=config['device'], nr_of_classes=6)


    def forward(self, input):
        x = input

        x = self.encoder(x)
        x = self.decoder(x)

        return x
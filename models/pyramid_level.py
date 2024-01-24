import torch
from torch import nn
from models.base import VariationalEncoder, Decoder
from models.upscaler import Upscaler


class PyramidLevel(nn.Module):
    
    def __init__(self, config):
        super(PyramidLevel, self).__init__()
        self.size = config['resolution']

        self.encoder: VariationalEncoder = VariationalEncoder(config['encoder_block_settings'], input_features=config['input_features'],  kernel_size=config['kernel_size'])
        self.decoder: Decoder = Decoder(config['decoder_block_settings'], config['kernel_size'])

        if config['resolution'] != 512:
            self.upscale_enhancer = Upscaler(config['resolution'], kernel_size=config['kernel_size']).to('cuda')
            for param in self.upscale_enhancer.parameters(): param.requires_grad = False

        self.scaling = 1.0 if 'scaling' not in config else config['scaling']


    def enhance(self, low, mask):
        x = self.upscale_enhancer([low, mask])
        return x


    def sample(self, z_cond, uv_mask, noise=None):
        if noise is None:
            eps = torch.randn(z_cond.shape).to(uv_mask.device)
        else:
            eps = noise

        z = torch.cat([eps, z_cond], dim=1)
        x = self.decoder(z)
        x = x * uv_mask[:,None,:,:]

        return x


    def reparameterize(self, mean, logvar):
        if self.training == True:
            eps = torch.normal(mean=0.0, std=1.0, size=mean.shape).to(mean.device)
            out = eps * torch.exp(logvar * .5) + mean
        else:
            out = torch.exp(logvar * .5) + mean
        return out


    def forward(self, inputs, z_cond):
        uv_unposed, uv_template, uv_body_posed, uv_normals, uv_mask, _ = inputs

        x = torch.concat([uv_unposed, uv_template, uv_body_posed, uv_normals], dim=1)
            
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        z = torch.reshape(z, (mean.shape[0], -1, 8, 4))

        z = torch.concat([z, z_cond], dim=1)
        x = self.decoder(z)

        x = x * uv_mask[:,None,:,:]
        x = x * self.scaling

        return x, mean, logvar
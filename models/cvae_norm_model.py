import torch
from torch import nn
from models.base import Encoder, VariationalEncoder, Decoder


class CVAENorm(nn.Module):
    
    def __init__(self, config):
        super(CVAENorm, self).__init__()
        self.size = config['resolution']

        self.static_enc : Encoder = Encoder(config['cond_ae_static']['encoder_block_settings'], config['cond_ae_static']['kernel_size'], config['cond_ae_static']['activation'], device=config['device'])
        self.body_enc : Encoder = Encoder(config['cond_ae_body']['encoder_block_settings'], config['cond_ae_body']['kernel_size'], config['cond_ae_body']['activation'], device=config['device'])

        for param in self.static_enc.parameters(): param.requires_grad = False
        for param in self.body_enc.parameters(): param.requires_grad = False

        self.encoder = VariationalEncoder(config['cvae']['encoder_block_settings'], config['cvae']['kernel_size'], config['cvae']['input_features'], device=config['device'])
        self.decoder : Decoder = Decoder(config['cvae']['decoder_block_settings'], kernel_size=config['cvae']['kernel_size'], device=config['device'])

        self.latent_shape = (config['cvae']['latent_dim'], 8, 4)


    def sample(self, uv_template, uv_body_posed, uv_mask, noise=None):
        if noise is None:
            eps = torch.randn(uv_template.shape[0], self.latent_shape[0], self.latent_shape[1], self.latent_shape[2]).to(uv_template.device)
        else:
            eps = noise

        z_cond = self.encode_conds(uv_template, uv_body_posed)
        z = torch.cat([eps, z_cond], dim=1)
        x = self.decoder(z)
        x = x * uv_mask[:,None,:,:]

        return x


    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor, training: bool = True):
        if training == True:
            eps = torch.randn(size=mean.shape).to(mean.device)
            out = eps * torch.exp(logvar * 0.5) + mean
        else:
            out = torch.exp(logvar * 0.5) + mean
        return out


    def encode_conds(self, uv_template, uv_body_posed):
        z_body = self.body_enc(uv_body_posed)
        z_static = self.static_enc(uv_template)
        z_cond = torch.concat([z_static, z_body], dim=1)
        return z_cond


    def forward(self, inputs):
        uv_unposed, uv_template, uv_body_posed, uv_mask = inputs
        x = torch.concat([uv_unposed, uv_template, uv_body_posed], dim=1)
        
        mean, logvar = self.encoder(x)
        z: torch.Tensor = self.reparameterize(mean, logvar)
        z = z.view((mean.shape[0], *self.latent_shape))
        z_cond = self.encode_conds(uv_template, uv_body_posed)

        z = torch.concat([z, z_cond], dim=1)
        x = self.decoder(z)
        x = x * uv_mask[:,None,:,:]

        return x, mean, logvar
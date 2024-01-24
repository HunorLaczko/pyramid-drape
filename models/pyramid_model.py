import torch
from torch import nn
from utils.utils import load_model
from models.base import Encoder
from models.pyramid_level import PyramidLevel
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode


class Pyramid(nn.Module):
    
    def __init__(self, config):
        super(Pyramid, self).__init__()

        self.static_enc : Encoder = Encoder(config['cond_ae_static']['encoder_block_settings'], kernel_size=config['cond_ae_static']['kernel_size'], activation=config['cond_ae_static']['activation'])
        self.body_enc : Encoder = Encoder(config['cond_ae_body']['encoder_block_settings'], kernel_size=config['cond_ae_body']['kernel_size'], activation=config['cond_ae_body']['activation'])
        self.norm_enc : Encoder = Encoder(config['cond_ae_norm']['encoder_block_settings'], config['cond_ae_norm']['kernel_size'], config['cond_ae_norm']['activation'], device=config['device'])

        for param in self.static_enc.parameters(): param.requires_grad = False
        for param in self.body_enc.parameters(): param.requires_grad = False
        for param in self.norm_enc.parameters(): param.requires_grad = False

        self.levels = nn.ModuleList([PyramidLevel({ **config['pyramid']['levels'][i]}) for i in range(len(config['pyramid']['levels']))])

        self.resizers = { res: Resize(size=(res, int(res/2)), interpolation=InterpolationMode.BILINEAR).to(config['device']) for res in config['pyramid']['resolutions'] }
        self.resizers[512] = Resize(size=(512, 256), interpolation=InterpolationMode.BILINEAR).to(config['device'])

    
    def sample(self, uv_template, uv_body_posed, uv_normals, uv_mask, noise=None):

        z_cond = self.encode_conds(uv_template, uv_body_posed, uv_normals)
    
        assert len(self.levels) == 4
        out = torch.zeros((uv_template.shape[0], 3, 64, 32)).to(uv_template.device)

        for i in range(len(self.levels)):
            level_out = self.levels[i].sample(z_cond, uv_mask[self.levels[i].size], noise=noise)
            if i != len(self.levels) -  1:
                out = out + level_out 
                out = self.levels[i].enhance(out, uv_mask[self.levels[i].size])
                out = self.resizers[self.levels[i+1].size](out)
                out = out * uv_mask[self.levels[i+1].size][:,None,:,:]
            else:
                out = out + level_out

        out = out * uv_mask[512][:,None,:,:]

        return out
    

    def encode_conds(self, uv_template, uv_body_posed, uv_norm):
        z_static = self.static_enc(uv_template)
        z_body = self.body_enc(uv_body_posed)
        z_norm = self.norm_enc(uv_norm)
        z_cond = torch.concat([z_static, z_body, z_norm], dim=1)

        return z_cond


    def forward(self, inputs):
        levels_input, conds = inputs
        uv_static, uv_body, uv_norm = conds[0], conds[1], conds[2]

        z_cond = self.encode_conds(uv_static, uv_body, uv_norm)

        out_levels = []
        pred, mean, logvar = self.levels[0](levels_input[0], z_cond)
        if 1 != len(self.levels):
            pred = self.levels[0].enhance(pred, levels_input[0][-2])  # levels_input[i][-2]=uv_mask_res
            out_levels.append([pred, mean, logvar])
            pred = self.resizers[self.levels[1].size](pred)
            pred = pred * levels_input[1][-2][:,None,:,:]
        else:
            pred = self.levels[0].enhance(pred, levels_input[0][-2])  # levels_input[i][-2]=uv_mask_res
            pred = self.resizers[512](pred)
            pred = pred * levels_input[-1][-1][:,None,:,:] # full res uv mask
            out_levels.append([pred, mean, logvar])

        for i in range(1, len(self.levels)):
            input = levels_input[i]
            input[0] = input[0] - pred
            pred_level, mean, logvar = self.levels[i](input, z_cond)
            pred = pred + pred_level
            if i != len(self.levels) - 1:
                pred = self.levels[i].enhance(pred, levels_input[i][-2])
                out_levels.append([pred, mean, logvar])
                pred = self.resizers[self.levels[i+1].size](pred)
                pred = pred * levels_input[i+1][-2][:,None,:,:]
            else:
                if self.levels[-1].size != 512:
                    pred = self.levels[-1].enhance(pred, levels_input[-1][-2])
                    pred = self.resizers[512](pred)
                pred = pred * levels_input[-1][-1][:,None,:,:] # full res uv mask
                out_levels.append([pred, mean, logvar])

        return out_levels
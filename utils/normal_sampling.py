import torch
import numpy as np
import scipy.interpolate as spi

from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode

from models.cvae_norm_model import CVAENorm
from utils.utils import load_model


# The decoder of the cvae is tanh bounded and comes out systematically smaller than
# the stored uv normals, so its output is rescaled the same way the original inference
# code did. Fitting the scale by least squares on the test set gives 1.39.
NORMAL_SCALE = 1.4


def fill(img, mask):
    # same as data/downscale.py, duplicated because that module parses command line
    # arguments when it is imported
    coords = np.transpose(np.stack(np.nonzero((1 - mask))), (1,0))
    img_idx = np.transpose(np.stack(np.nonzero((mask))), (1,0))
    img_data = img[img_idx[:,0], img_idx[:,1], :]

    interpolator = spi.NearestNDInterpolator(x=img_idx, y=img_data)

    img_2 = interpolator(coords)
    res = np.zeros((512,256,3), dtype=np.float32)
    res[coords[:,0], coords[:,1], :] = img_2
    res[img_idx[:,0], img_idx[:,1], :] = img_data

    return res


class NormalSampler():
    """Replaces the ground truth uv normals of a batch with sampled ones."""

    def __init__(self, config):
        sampler_config = config['norm_sampler']

        self.model: CVAENorm = CVAENorm(sampler_config)
        load_model(self.model, sampler_config['checkpoint'])
        self.model.eval()

        self.resolutions = [res for res in config['pyramid']['resolutions'] if res != 512]
        self.resizers = { res: Resize(size=(res, int(res/2)), interpolation=InterpolationMode.BICUBIC).to(config['device']) for res in self.resolutions }


    def __call__(self, batch, noise=None):
        with torch.inference_mode():
            uv_normals = self.model.sample(batch['uv_static'], batch['uv_body_posed'], batch['uv_mask'], noise=noise) * NORMAL_SCALE

        batch['uv_normals'] = uv_normals

        # the lower resolutions are built the same way data/downscale.py built the
        # ground truth ones: fill the masked out region, resize, re-apply the mask
        mask = batch['uv_mask'].cpu().numpy()
        downscaled = { res: [] for res in self.resolutions }
        for i in range(uv_normals.shape[0]):
            filled = fill(uv_normals[i].cpu().numpy().transpose(1,2,0), mask[i])
            filled = torch.from_numpy(filled.transpose(2,0,1)).to(uv_normals.device)
            for res in self.resolutions:
                downscaled[res].append(self.resizers[res](filled) * batch['uv_mask_' + str(res)][i])

        for res in self.resolutions:
            batch['uv_normals_' + str(res)] = torch.stack(downscaled[res], dim=0)

        return batch

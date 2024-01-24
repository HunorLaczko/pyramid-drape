import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

import importlib
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--name', type=str)
parser.add_argument('--config', type=str, default='configs.pyr')
parser.add_argument('--debug', action='store_true')
parser.add_argument('-g', '--gpu', type=str, default='3')
args = vars(parser.parse_args())

if args['debug']:
    from configs.pyr import config
else:
    config_module = importlib.import_module(args['config'])
    config = config_module.config

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu']
if args['debug']:
    os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

import tqdm
import numpy as np
import torch

from tqdm import tqdm
from data.dataset import Dataset
from torch.utils.data import DataLoader
from utils.utils import UV2Mesh, group_inputs_for_levels, load_model
from smpl.smpl_torch import SMPL
from models.pyramid_model import Pyramid


uv2mesh = UV2Mesh(res=256, is_skirt=config['is_skirt'], is_smpl_orig=config['is_cape'])


def step(batch):
    with torch.inference_mode():
        levels_input, conds = group_inputs_for_levels(batch, config)

        pred = model([levels_input, conds])[-1][0]
        mesh_pred = uv2mesh(pred)

        shape = batch['pose_shape'][:, 72:]
        gender = torch.greater_equal(shape[:, 10] , 0.5).to(dtype=torch.int64)
        shape_offsets = torch.stack([torch.tensordot(shape[i,:10][None, :], smpl.shapedirs[gender[i]], dims=[[1], [2]])[0,smpl.to_keep,:] for i in range(shape.shape[0])])

        mesh_pred_shaped = mesh_pred + shape_offsets * batch['mesh_mask'][:,:,None]
        if not config['is_cape']:
            mesh_pred_posed = smpl(batch['pose_shape'][:, :72], batch['pose_shape'][:, 72:], mesh_pred_shaped)[0]
        else:
            mesh_pred_posed = mesh_pred_shaped

    return mesh_pred_posed


def generate():
    for batch in tqdm(test_dataloader):
        with torch.inference_mode():
            mesh_pred = step(batch)

        ids = batch['id']
        frame_nrs = batch['frame_nr']
        garments = batch['garment']

        for i in range(len(ids)):
            np.savez_compressed(f'{out_folder}/{ids[i]}_{frame_nrs[i]}_{garments[i]}', mesh_pred[i].cpu().numpy())


if __name__ == '__main__':
    smpl = SMPL(orig_smpl=config['is_cape'], path='smpl', device=config['device'])

    dataset_test = Dataset(config['data_dir'], config['signature'], device=config['device'], frames_file=config['frames'], use_cache=config['dataset_in_memory'], resolutions=config['pyramid']['resolutions'], is_cape=config['is_cape'])
    test_dataloader = DataLoader(dataset_test, batch_size=4, drop_last=True, shuffle=False)

    model: Pyramid = Pyramid(config)
    load_model(model, 'checkpoints/' + config['name'], False)
    model.eval()

    out_folder = f'{config["out_dir"]}/{config["name"]}'
    os.makedirs(out_folder, exist_ok=True)

    generate()
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

import importlib
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--name', type=str)
parser.add_argument('--config', type=str, default='configs.pyr_skirt')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--iteration', type=str, default='')
parser.add_argument('-g', '--gpu', type=str, default='1')
args = vars(parser.parse_args())

config_module = importlib.import_module(args['config'])
config = config_module.config

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']
if args['debug']:
    os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

import argparse
import tqdm
import numpy as np

from tqdm import tqdm
from utils.utils import Frame
from data.cloth3d_io import readPC2Frame


def calculate():
    errors = []
    errors_category = dict()

    for frame in tqdm(frames):
        try:
            pred = np.load(f'{out_folder}/{frame}.npz', allow_pickle=True)['arr_0']
        except:
            print(f'Could not load {frame}. This might happen if the number of frames was not divisible by the batch size.')
            continue
        gt = readPC2Frame(f'{config["data_dir"]}/mesh_posed_no_orient/{frame.id}_{frame.garment}.pc16', frame.frame_nr, True)
        
        fix_data = np.load(f'{config["data_dir"]}/mesh_fixed/{frame.id}_{frame.garment}.npz')
        try:
            mask_convert = fix_data['mask_convert']
        except:
            mask_convert = fix_data['mas_convert']
        new_mask = fix_data['mask']
        gt = gt[mask_convert]

        pred = pred[new_mask]
        error = np.linalg.norm(pred - gt, axis=1).mean()
        errors.append((str(frame), error))

        faces = fix_data['faces']

        if args['debug']:
            # create a plotly plot with pred and gt as meshes
            import plotly.graph_objects as go
            fig = go.Figure(data=[
                go.Mesh3d(
                    x=pred[:, 0], y=pred[:, 2], z=pred[:, 1],
                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                    color='lightpink', opacity=1,
                    name='pred'
                ),
                go.Mesh3d(
                    x=gt[:, 0], y=gt[:, 2], z=gt[:, 1],
                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                    color='lightblue', opacity=1,
                    name='gt'
                )
            ])
            fig.update_layout(
                scene = dict(
                    xaxis = dict(nticks=4, range=[-1,1],),
                    yaxis = dict(nticks=4, range=[-1,1],),
                    zaxis = dict(nticks=4, range=[-1,1],),
                ),
                scene_aspectmode='manual',
                scene_aspectratio=dict(x=1, y=1, z=1)
            )
            fig.show()
        

        if frame.garment not in errors_category.keys():
            errors_category[frame.garment] = []
        errors_category[frame.garment].append(error)

    errors_mean = dict()
    for category in errors_category:
        errors_mean[category] = round(np.mean(errors_category[category]), 5)
    print(errors_mean) 

    mean = round(np.mean(list(errors_mean.values())), 5)
    print(f'Final error: {mean}')

    # save errors to a csv file
    with open(f'{config["out_dir"]}/{config["name"]}.csv', 'w') as f:
        for frame, error in errors:
            frame_str = ','.join(str(frame).split('_'))
            f.write(f'{frame_str},{error}\n')


if __name__ == '__main__':
    out_folder = f'{config["out_dir"]}/{config["name"]}'
    os.makedirs(out_folder, exist_ok=True)

    frames_file = config['frames']
    with open(os.path.join(config['data_dir'], frames_file), 'r') as f:
        frames = f.readlines()
    frames = [Frame(frames.split()[0], int(frames.split()[1]), frames.split()[2]) for frames in frames]

    calculate()
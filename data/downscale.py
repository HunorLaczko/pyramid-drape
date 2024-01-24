import os
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", help = "data folder", default='/data/cloth3d_processed/test')
parser.add_argument("-s", "--start", help = "start of range", default=0)
parser.add_argument("-e", "--end", help = "end of range", default=1000000)
parser.add_argument("-n", "--name", help = "name", default='downscale')
parser.add_argument("-g", "--gpu", help = "gpu id", default='0')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import tqdm
import cv2
import torch
import scipy.interpolate as spi
import numpy as np
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from utils.utils import Frame


def fill(img, mask):
    coords = np.transpose(np.stack(np.nonzero((1 - mask))), (1,0))
    img_idx = np.transpose(np.stack(np.nonzero((mask))), (1,0))
    img_data = img[img_idx[:,0], img_idx[:,1], :]

    interpolator =  spi.NearestNDInterpolator(x=img_idx, y=img_data)

    img_2 = interpolator(coords)
    res = np.zeros((512,256,3), dtype=np.float32)
    res[coords[:,0], coords[:,1], :] = img_2
    res[img_idx[:,0], img_idx[:,1], :] = img_data

    return res


def generate_downscaled_uv_map(folder, frame: Frame, size):
    img = np.load(os.path.join(folder, 'uv_mask', f'{frame.id}_{frame.garment}.npz')); img = img['uv_mask'].astype(np.float32)
    down = resizers_nearest[size](torch.from_numpy(img[None,:,:]).to(device))
    down = np.squeeze(down.cpu().numpy().astype(bool))
    if not os.path.exists(os.path.join(folder, 'uv_mask' + '_' + str(size))):
        os.makedirs(os.path.join(folder, 'uv_mask' + '_' + str(size)), exist_ok=True)
    np.savez_compressed(os.path.join(folder, 'uv_mask' + '_' + str(size), f'{frame.id}_{frame.garment}.npz'), uv_mask = down)


class UVMaskStore():
    def __init__(self, folder, size):
        self.store = dict()
        self.folder = folder
        self.size = size

    def read(self, frame: Frame):
        path = os.path.join(self.folder, 'uv_mask' + ('_' + str(self.size) if self.size != 512 else ''), f'{frame.id}_{frame.garment}.npz')
        if not os.path.exists(path):
            generate_downscaled_uv_map(self.folder, frame, self.size)

        mask = np.load(path)
        mask = mask['uv_mask']
        return mask  

    def __getitem__(self, frame: Frame) -> dict:
        if (frame.id, frame.garment) not in self.store:
            self.store[(frame.id, frame.garment)] = self.read(frame)

        return self.store[(frame.id, frame.garment)]


def downscale_uv_maps_posed_body(folder, type, sizes):
    for size in sizes:
        os.makedirs(os.path.join(folder, type + str(size)), exist_ok=True)

    frames = sorted(os.listdir(os.path.join(folder, type)))[int(args.start):int(args.end)]
    frames = [Frame(frame.split('_')[0], frame.split('_')[1].split('.')[0], '') for frame in frames]

    for frame in tqdm.tqdm(frames):
        img = np.load(os.path.join(folder, type, f'{frame.id}_{frame.frame_nr}.npz'))['uv_body_posed']
        img = fill(img, uv_masks_body[512])
        for size in sizes:
            mask = uv_masks_body[size]
            down = resizers[size](torch.from_numpy(img.transpose((2,0,1))).to(device)).cpu().numpy()
            down = down * mask[None,:,:]
            np.savez_compressed(os.path.join(folder, type + str(size), f'{frame.id}_{frame.frame_nr}.npz'), uv_body_posed = down)


def downscale_uv_maps_posed_no_orient(folder, type, sizes):
    for size in sizes:
        os.makedirs(os.path.join(folder, type + '_' + str(size)), exist_ok=True)

    frames = sorted(os.listdir(os.path.join(folder, type)))[int(args.start):int(args.end)]
    frames = [Frame(frame.split('_')[0], frame.split('_')[1], frame.split('_')[2].split('.')[0]) for frame in frames]

    for frame in tqdm.tqdm(frames):
        img = np.load(os.path.join(folder, type, f'{frame.id}_{frame.frame_nr}_{frame.garment}.npz'))['uv_posed']
        img = fill(img, uv_mask_store[512][frame])
        for size in sizes:
            mask = uv_mask_store[size][frame]
            down = resizers[size](torch.from_numpy(img.transpose((2,0,1))).to(device)).cpu().numpy()
            down = down * mask[None,:,:]
            np.savez_compressed(os.path.join(folder, type + '_' + str(size), f'{frame.id}_{frame.frame_nr}_{frame.garment}.npz'), uv_posed = down)


def downscale_uv_maps_normals(folder, type, sizes):
    for size in sizes:
        os.makedirs(os.path.join(folder, type + '_' + str(size)), exist_ok=True)

    frames = sorted(os.listdir(os.path.join(folder, type)))[int(args.start):int(args.end)]
    frames = [Frame(frame.split('_')[0], frame.split('_')[1], frame.split('_')[2].split('.')[0]) for frame in frames]

    for frame in tqdm.tqdm(frames):
        img = np.load(os.path.join(folder, type, f'{frame.id}_{frame.frame_nr}_{frame.garment}.npz'))['uv_normals']
        img = fill(img, uv_mask_store[512][frame])
        for size in sizes:
            mask = uv_mask_store[size][frame]
            down = resizers[size](torch.from_numpy(img.transpose((2,0,1))).to(device)).cpu().numpy()
            down = down * mask[None,:,:]
            np.savez_compressed(os.path.join(folder, type + '_' + str(size), f'{frame.id}_{frame.frame_nr}_{frame.garment}.npz'), uv_normals = down)


def downscale_uv_maps_unposed(folder, type, sizes):
    for size in sizes:
        os.makedirs(os.path.join(folder, type + '_' + str(size)), exist_ok=True)

    frames = sorted(os.listdir(os.path.join(folder, type)))[int(args.start):int(args.end)]
    frames = [Frame(frame.split('_')[0], frame.split('_')[1], frame.split('_')[2].split('.')[0]) for frame in frames]

    for frame in tqdm.tqdm(frames):
        img = np.load(os.path.join(folder, type, f'{frame.id}_{frame.frame_nr}_{frame.garment}.npz'))['uv_unposed']
        img = fill(img, uv_mask_store[512][frame])
        for size in sizes:
            mask = uv_mask_store[size][frame]
            down = resizers[size](torch.from_numpy(img.transpose((2,0,1))).to(device)).cpu().numpy()
            down = down * mask[None,:,:]
            np.savez_compressed(os.path.join(folder, type + '_' + str(size), f'{frame.id}_{frame.frame_nr}_{frame.garment}.npz'), uv_unposed = down)


def downscale_uv_maps_static(folder, type, sizes):
    for size in sizes:
        os.makedirs(os.path.join(folder, type + '_' + str(size)), exist_ok=True)

    frames = sorted(os.listdir(os.path.join(folder, type)))
    import random
    random.shuffle(frames)
    frames = [(frame.split('_')[0], frame.split('_')[1].split('.')[0]) for frame in frames]

    for frame in tqdm.tqdm(frames):
        img = np.load(os.path.join(folder, type, f'{frame[0]}_{frame[1]}.npz'))['uv_static']
        img = fill(img, uv_mask_store[512][Frame(frame[0], 0, frame[1])])
        for size in sizes:
            mask = uv_mask_store[size][Frame(frame[0], 0, frame[1])]
            down = resizers[size](torch.from_numpy(img.transpose((2,0,1))).to(device)).cpu().numpy()
            down = down * mask[None,:,:]
            np.savez_compressed(os.path.join(folder, type + '_' + str(size), f'{frame[0]}_{frame[1]}.npz'), uv_static = down)



if __name__ == '__main__':


    device = 'cuda'
    sizes = [64, 128, 256]
    folder = args.folder
    resizers = { res: Resize(size=(res, int(res/2)), interpolation=InterpolationMode.BICUBIC).to(device) for res in sizes }
    resizers_nearest = { res: Resize(size=(res, int(res/2)), interpolation=InterpolationMode.NEAREST_EXACT).to(device) for res in sizes }

    uv_masks_body = dict()
    uv_mask_body = np.load('utils_data/mask_256x512.npy')
    uv_mask_body = cv2.dilate(uv_mask_body.astype(np.float32), cv2.getStructuringElement(cv2.MORPH_RECT, (2 * 2 + 1, 2 * 2 + 1), (2, 2)), iterations=2)
    uv_masks_body[512] = uv_mask_body.astype(bool)
    for res in [64, 128, 256]:
        down = resizers_nearest[res](torch.from_numpy(uv_mask_body[None,:,:].astype(np.int32)).to(device))
        down = np.squeeze(down.cpu().numpy().astype(bool))
        uv_masks_body[res] = down
        
    uv_mask_store = { size: UVMaskStore(folder, size=size) for size in sizes }
    uv_mask_store[512] = UVMaskStore(folder, size=512)

    downscale_uv_maps_static(folder=folder, type='uv_static', sizes=sizes)
    downscale_uv_maps_unposed(folder=folder, type='uv_unposed_no_shape', sizes=sizes)
    downscale_uv_maps_posed_no_orient(folder=folder, type='uv_posed_no_orient', sizes=sizes)
    downscale_uv_maps_normals(folder=folder, type='uv_normals', sizes=sizes)
    downscale_uv_maps_posed_body(folder=folder, type='uv_body_posed_no_orient', sizes=sizes)

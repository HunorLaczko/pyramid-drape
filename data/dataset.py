import os
import sys
import torch
import random
import torchvision as tv
import numpy as np

from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from utils.utils import Frame
from data.cloth3d_io import readPC2, readPC2Frame
# from data.dataset_cape import DatasetCape


def read_frames(folder, shuffle=True, is_cape=False, frames_file=None):
    if frames_file is None:
        uv_static_folder = os.path.join(folder, 'pose_shape') 
        frames = sorted(os.listdir(uv_static_folder))

        if is_cape:
            frames = ['.'.join(frame.split('.')[:-1]) for frame in frames]
            frames = [Frame(frame.split('.')[0], frame.split('.')[1]) for frame in frames]
        else:
            frames = [frame.split('.')[0] for frame in frames]
            frames = [Frame(frame.split('_')[0], frame.split('_')[1]) for frame in frames]

    if frames_file is not None:
        frames = []
        with open(os.path.join(folder, frames_file), 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        frames = [Frame(line.split('_')[0], line.split('_')[1]) for line in lines]

    if shuffle:
        random.shuffle(frames)

    return frames


class StaticCache():
    def __init__(self, folder, signature, resolutions=[512], is_cape=False, device='cuda') -> None:
        self.store = dict()
        self.signature = signature
        self.folder = folder
        assert set(resolutions).issubset(set([32, 64, 128, 256, 512]))
        self.resolutions = resolutions
        self.is_cape = is_cape
        self.device = device
        
        if is_cape:
            self.uv_masks = dict()
            uv_mask = torch.tensor(np.load('custom_data/garment_uv_mask_full.npy').astype(np.int32), device=device)
            self.uv_masks[512] = uv_mask.to(torch.float32).cpu().numpy()
            for res in [32, 64, 128, 256]:
                down = Resize(size=(res, int(res/2)), interpolation=InterpolationMode.NEAREST_EXACT)(uv_mask[None,:,:])
                down = torch.squeeze(down).to(torch.float32)
                self.uv_masks[res] = down.cpu().numpy()


    def read(self, id, garment):
        data = dict()

        mesh_mask = np.load(os.path.join(self.folder, 'mesh_mask_orig', f'{id}_{garment}.npy')); mesh_mask = mesh_mask.astype(bool)
        data['mesh_mask'] = mesh_mask

        mesh_fix = np.load(os.path.join(self.folder, 'mesh_fixed', f'{id}_{garment}.npz'))
        data['mesh_fix'] = dict()
        data['mesh_fix']['mesh_mask'] = torch.from_numpy(mesh_fix['mask'])

        for res in self.resolutions:
            res_string = ''
            if res != 512:
                res_string = '_' + str(res)

            if 'uv_static' + res_string in self.signature:
                uv_static = np.load(os.path.join(self.folder, f'uv_static{res_string}', f'{id}_{garment}.npz')); uv_static = uv_static['uv_static']; data['uv_static' + res_string] = uv_static
            if 'uv_mask' + res_string in self.signature:
                if self.is_cape:
                    data['uv_mask' + res_string] = self.uv_masks[res]
                else:
                    uv_mask = np.load(os.path.join(self.folder, f'uv_mask{res_string}', f'{id}_{garment}.npz')); uv_mask=uv_mask['uv_mask']; uv_mask = uv_mask.astype(np.float32); data['uv_mask' + res_string] = uv_mask

        if 'mesh_static' in self.signature:
            mesh_static = np.load(os.path.join(self.folder, 'mesh_static', f'{id}_{garment}.npy'))
            data['mesh_static'] = np.zeros((14475, 3), dtype=np.float32)
            data['mesh_static'][mesh_mask] = mesh_static.astype(np.float32)

        return data


    def __getitem__(self, frame: Frame) -> dict:
        if (frame.id, frame.garment) not in self.store:
            item = self.read(frame.id, frame.garment)
            stored_data = { key: torch.from_numpy(value).to('cpu') for key, value in item.items()  if type(value) == np.ndarray}
            stored_data['mesh_fix'] = item['mesh_fix']
            for key, value in stored_data.items():
                if key.startswith('uv_') and value.shape[-1] == 3:
                    stored_data[key] = torch.permute(value, [2,0,1])
            self.store[(frame.id, frame.garment)] = stored_data
        
        data = { key: value.to(self.device) for key, value in self.store[(frame.id, frame.garment)].items() if type(value) == torch.Tensor }
        data['mesh_fix'] = { key: value.to(self.device) for key, value in self.store[(frame.id, frame.garment)]['mesh_fix'].items() }
        return data



def read_frames_file(frames_file, shuffle=False):
    frames = []
    with open(frames_file, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    frames = [Frame(frame_str=line) for line in lines]

    if shuffle:
        random.shuffle(frames)

    return frames


class DatasetCloth3D(torch.utils.data.Dataset):
    def __init__(self, folder, signature, resolutions=[512], frames_file='', device='cuda', use_cache=False, is_cape=False):
        self.folder = folder
        self.device = device
        self.is_cape = is_cape
        self.signature = signature
        self.resolutions = resolutions

        self.transforms = tv.transforms.ToTensor()
        self.frames = read_frames_file(os.path.join(folder, frames_file), shuffle=False)

        self.static_cache = StaticCache(folder, signature, is_cape=is_cape, resolutions=resolutions, device=device)

        self.cache = dict() if use_cache else None

        ids = set([frame.id for frame in self.frames])
        self.pose_cache = dict()
        for id in ids:
            self.pose_cache[id] = np.load(os.path.join(self.folder, 'pose', f'{id}.npy'))
        self.shape_cache = dict()
        for id in ids:
            self.shape_cache[id] = np.load(os.path.join(self.folder, 'shape', f'{id}.npy'))


    def __len__(self):
        return len(self.frames)


    def __getitem__(self, idx):
        frame : Frame = self.frames[idx]
        if self.cache is not None:
            if frame in self.cache.keys():
                data = { key: (value.to(self.device) if type(value) == torch.Tensor else value) for key, value in self.cache[frame].items()}
                data = { **data, **self.static_cache[frame] }
                return data

        item = dict()
        mesh_mask = self.static_cache[frame]['mesh_mask'].cpu().numpy().astype(bool)

        item['id'] = frame.id
        item['frame_nr'] = frame.frame_nr
        item['garment'] = frame.garment

        if 'pose_shape' in self.signature:
            pose = self.pose_cache[frame.id][frame.frame_nr]
            pose[:3] = 0
            item['pose_shape'] = np.concatenate((pose, self.shape_cache[frame.id]), axis=0)
        if 'mesh_body_posed' in self.signature:
            mesh_body_posed_no_orient = readPC2Frame(os.path.join(self.folder, 'mesh_body_posed_no_orient', f'{frame.id}.pc16'), frame.frame_nr, True)
            item['mesh_body_posed'] = mesh_body_posed_no_orient
        if 'mesh_posed' in self.signature:
            mesh_posed = readPC2Frame(os.path.join(self.folder, 'mesh_posed_no_orient', f'{frame.id}_{frame.garment}.pc16'), frame.frame_nr, True)
            item['mesh_posed'] = np.zeros((14475, 3), dtype=np.float32)
            item['mesh_posed'][mesh_mask] = mesh_posed
        if 'mesh_unposed' in self.signature:
            mesh_unposed_no_shape = readPC2Frame(os.path.join(self.folder, 'mesh_unposed_no_shape', f'{frame.id}_{frame.garment}.pc16'), frame.frame_nr, True)
            item['mesh_unposed'] = np.zeros((14475, 3), dtype=np.float32)
            item['mesh_unposed'][mesh_mask] = mesh_unposed_no_shape
        if 'mesh_normals' in self.signature:
            mesh_normals = readPC2Frame(os.path.join(self.folder, 'mesh_normals', f'{frame.id}_{frame.garment}.pc16'), frame.frame_nr, True)
            item['mesh_normals'] = np.zeros((14475, 3), dtype=np.float32)
            item['mesh_normals'][mesh_mask] = mesh_normals

        for res in self.resolutions:
            res_string = ''
            if res != 512:
                res_string = '_' + str(res)

            if 'uv_normals' + res_string in self.signature:
                uv_normals = np.load(os.path.join(self.folder, 'uv_normals' + res_string, f'{frame.id}_{frame.frame_nr}_{frame.garment}.npz')); uv_normals = uv_normals['uv_normals']
                item['uv_normals' + res_string] = uv_normals

            if 'uv_unposed' + res_string in self.signature:
                uv_unposed_no_shape = np.load(os.path.join(self.folder, 'uv_unposed_no_shape' + res_string, f'{frame.id}_{frame.frame_nr}_{frame.garment}.npz')); uv_unposed_no_shape = uv_unposed_no_shape['uv_unposed']
                item['uv_unposed' + res_string] = uv_unposed_no_shape

            if 'uv_body_posed' + res_string in self.signature:
                uv_body_posed_no_orient = np.load(os.path.join(self.folder, 'uv_body_posed_no_orient' + res_string[1:], f'{frame.id}_{frame.frame_nr}.npz')); uv_body_posed_no_orient = uv_body_posed_no_orient['uv_body_posed']
                item['uv_body_posed' + res_string] = uv_body_posed_no_orient

            if 'uv_posed' + res_string in self.signature:
                uv_posed = np.load(os.path.join(self.folder, 'uv_posed_no_orient' + res_string, f'{frame.id}_{frame.frame_nr}_{frame.garment}.npz')); uv_posed = uv_posed['uv_posed']
                item['uv_posed' + res_string] = uv_posed


        data = { key: (torch.from_numpy(value) if not (type(value) == str or type(value) == torch.Tensor or type(value) == int) else value) for key, value in item.items()}

        for key, value in data.items():
            if key.startswith('uv_') and (value.shape[-1] == 3 or value.shape[-1] == 7):
                data[key] = torch.permute(value, [2,0,1])

        if self.cache is not None:
            self.cache[frame] = data
        
        data = { key: (value.to(self.device) if type(value) == torch.Tensor else value) for key, value in data.items()}
        data = { **data, **self.static_cache[frame] }
        return data


def Dataset(folder, signature, resolutions=[512], frames_file='', device='cuda', use_cache=False, is_cape=False):
    if not is_cape:
        return DatasetCloth3D(folder, signature, resolutions=resolutions, frames_file=frames_file, device=device, use_cache=use_cache, is_cape=False)
    else:
        return DatasetCape(folder, signature, resolutions=resolutions, frames_file=frames_file, device=device, use_cache=use_cache, is_cape=True)


import os
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--start", help = "start of range", type=int, default=0)
parser.add_argument("-e", "--end", help = "end of range", type=int, default=3)
parser.add_argument("-g", "--gpu", help = "gpu id", default='0')
parser.add_argument("-d", "--debug", help = "debug mode", default=False, action='store_true')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import random
import pyvista as pv
import numpy as np
import scipy.io as sio
import trimesh as tm
from tqdm import tqdm

import plotly.graph_objects as go

import torch
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from data.cloth3d_io import readPC2, loadInfo, writePC2, readOBJ, quads2tris, readPC2Frame
from smpl.smpl_torch import SMPL
from utils.utils import _maskFaces, remove_long_edges, Frame


def deorient(verts, pose):
    rodr = smpl.rodrigues(torch.tensor(pose.T, device=smpl.device).reshape(-1, 1, 3)).reshape(verts.shape[0], -1, 3, 3).cpu().numpy()
    rotated = np.zeros_like(verts)
    for frame in range(verts.shape[0]):
        r = R.from_matrix(rodr[frame, 0, :, :])
        rotated[frame] = r.apply(verts[frame], inverse=True)

    return rotated


def create_mesh_posed_no_orient_and_mask(folder, folder_reg, out_folder):
    ids = sorted(os.listdir(folder))
    os.makedirs(os.path.join(out_folder, 'mesh_posed_no_orient'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'mesh_mask_orig'), exist_ok=True)

    for id in tqdm(ids):
        garment_dir = os.path.join(folder, id)
        if not os.path.isdir(garment_dir):
            continue

        garments = [f.replace('.obj','') for f in os.listdir(garment_dir) if f.endswith('.obj')]
        info = loadInfo(os.path.join(folder, id, 'info.mat'))
        
        for garment in garments:
            verts = readPC2(os.path.join(garment_dir, garment + '.pc16'), True)['V']
            reg_data = sio.loadmat(os.path.join(folder_reg, id, garment + '_weights_fixed.mat'))
            verts = (verts[:, reg_data['F'], :] * reg_data['W'][None, :, :, None]).sum(axis=2)
            mask_idx = sio.loadmat(os.path.join(folder_reg, id, garment + '_mask.mat'))['mask_idx'].reshape((-1)) - 1
            mask = np.zeros((14475), dtype=bool)
            mask[mask_idx] = True

            assert info['poses'].shape[1] == verts.shape[0], 'The shape of the pose and the number of frames in the mesh do not match!'

            verts = deorient(verts, info['poses'])

            writePC2(os.path.join(out_folder, 'mesh_posed_no_orient', f'{id}_{garment}.pc16'), verts, True)
            np.save(os.path.join(out_folder, 'mesh_mask_orig', f'{id}_{garment}.npy'), mask)


def create_pose_shape(folder, out_folder):
    ids = sorted(os.listdir(folder))
    os.makedirs(os.path.join(out_folder, 'pose'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'shape'), exist_ok=True)

    for id in tqdm(ids):
        garment_dir = os.path.join(folder, id)
        if not os.path.isdir(garment_dir):
            continue

        info = loadInfo(os.path.join(folder, id, 'info.mat'))
        
        shape = np.concatenate([info['shape'], np.asarray([info['gender']])]).astype(np.float32)
        np.save(os.path.join(out_folder, 'pose', f'{id}.npy'), info['poses'].T.astype(np.float32))
        np.save(os.path.join(out_folder, 'shape', f'{id}.npy'), shape)


def create_mesh_static(folder, out_folder):
    ids = sorted(os.listdir(folder))
    os.makedirs(os.path.join(out_folder, 'mesh_static'), exist_ok=True)

    for id in tqdm(ids):
        garment_dir = os.path.join(folder, id)
        if not os.path.isdir(garment_dir):
            continue

        garments = [f.replace('.obj','') for f in os.listdir(garment_dir) if f.endswith('.obj')]
        
        for garment in garments:
            if os.path.exists(os.path.join(out_folder, f'{id}_{garment}.pc16')):
                continue

            verts_static, _, _, _ = readOBJ(os.path.join(garment_dir, f'{garment}.obj'))
            reg_data = sio.loadmat(os.path.join(folder_reg, id, garment + '_weights_fixed.mat'))
            verts_static = (verts_static[reg_data['F'], :] * reg_data['W'][:, :, None]).sum(axis=1).astype(np.float32)

            # rotate 90 degrees around x axis
            verts_static = np.matmul(verts_static, np.array([[1,0,0],[0,0,-1],[0,1,0]]))

            np.save(os.path.join(out_folder, 'mesh_static', f'{id}_{garment}.npy'), verts_static.astype(np.float32))


def create_mesh_body_posed_no_orient(folder, out_folder):
    ids = sorted(os.listdir(folder))
    os.makedirs(os.path.join(out_folder, 'mesh_body_posed_no_orient'), exist_ok=True)

    for id in tqdm(ids):
        garment_dir = os.path.join(folder, id)
        if not os.path.isdir(garment_dir):
            continue

        info = sio.loadmat(os.path.join(folder, id, 'info.mat'))
        pose, shape, gender = info['poses'], info['shape'], info['gender']
        pose = np.transpose(pose, [1, 0]).astype(np.float32)

        pose[:, :3] = 0
        beta = np.squeeze(np.concatenate((shape, gender), axis=1)).astype(np.float32)

        smpl = SMPL(path='smpl', device='cuda', batch_size=pose.shape[0])

        T = smpl(torch.tensor(pose, device=smpl.device), torch.tensor(np.tile(beta[np.newaxis, :], [pose.shape[0], 1]), device=smpl.device))[0]
        T = T.cpu().numpy()

        writePC2(os.path.join(out_folder, 'mesh_body_posed_no_orient', f'{id}.pc16'), T, True)


def calculate_normals(vertices, faces):
    normals = []
    for i in range(vertices.shape[0]):
        mesh = tm.Trimesh(vertices=vertices[i], faces=faces, process=False, validate=False)
        normals.append(mesh.vertex_normals)
    return np.stack(normals)


def create_mesh_normals(folder, out_folder):
    ids = sorted(os.listdir(folder))
    os.makedirs(os.path.join(out_folder, 'mesh_normals'), exist_ok=True)

    for id in tqdm(ids):
        garment_dir = os.path.join(folder, id)
        if not os.path.isdir(garment_dir):
            continue

        garments = [f.replace('.obj','') for f in os.listdir(garment_dir) if f.endswith('.obj')]
        info = sio.loadmat(os.path.join(folder, id, 'info.mat'))
        
        for garment in garments:
            if os.path.exists(os.path.join(out_folder, f'{id}_{garment}.pc16')):
                continue

            verts = readPC2(os.path.join(garment_dir, garment + '.pc16'), True)['V']
            verts = deorient(verts, info['poses'][:, :verts.shape[0]])
            _, faces, _, _ = readOBJ(os.path.join(garment_dir, garment + '.obj'))
            faces = quads2tris(faces)
            normals = calculate_normals(verts, faces.astype(np.int32))

            reg_data = sio.loadmat(os.path.join(folder_reg, id, garment + '_weights_fixed.mat'))
            normals = (normals[:, reg_data['F'], :] * reg_data['W'][None, :, :, None]).sum(axis=2)

            writePC2(os.path.join(out_folder, 'mesh_normals', f'{id}_{garment}.pc16'), normals, True)


def create_mesh_normals_cape(folder, out_folder):
    frames = os.listdir(os.path.join(folder, 'mesh_posed'))
    os.makedirs(os.path.join(folder, 'mesh_normals'), exist_ok=True)
    F = np.load('smpl/smpl_faces_orig.npy')

    for frame in tqdm(frames):
        id, frame_nr = frame.split('.')[:2]

        verts = np.load(os.path.join(folder, 'mesh_posed', f'{id}.{frame_nr}.npz'))['mesh_posed']
        normals = calculate_normals(verts[None,:,:], F)[0]

        np.savez_compressed(os.path.join(folder, 'mesh_normals', f'{id}.{frame_nr}.npz'), mesh_normals=normals)


def verify_registration_and_mask(folder, out_folder):
    ids = sorted(os.listdir(folder))

    for id in ids:
        garment_dir = os.path.join(folder, id)
        if not os.path.isdir(garment_dir):
            continue

        garments = [f.replace('.obj','') for f in os.listdir(garment_dir) if f.endswith('.obj')]
        
        for garment in garments:
            vertices = readPC2(os.path.join(out_folder, 'mesh_posed_no_orient', f'{id}_{garment}.pc16'), True)['V']
            mask = np.load(os.path.join(out_folder, 'mesh_mask', f'{id}_{garment}.npy'))

            if mask.sum() == vertices.shape[1]:
                print(id, garment)

            pass


def create_mesh_unposed_no_shape(folder, out_folder, start=0, end=1):
    ids = sorted(os.listdir(folder))[start:end]
    os.makedirs(os.path.join(out_folder, 'mesh_unposed_no_shape'), exist_ok=True)
    F = np.load('smpl/smpl_faces.npy')

    for id in tqdm(ids):
        garment_dir = os.path.join(folder, id)
        if not os.path.isdir(garment_dir):
            continue

        garments = [f.replace('.obj','') for f in os.listdir(garment_dir) if f.endswith('.obj')]
        pose = np.load(os.path.join(out_folder, 'pose', f'{id}.npy'))
        shape = np.tile(np.load(os.path.join(out_folder, 'shape', f'{id}.npy'))[None,:], [pose.shape[0],1])

        pose = torch.tensor(pose, dtype=torch.float32, device='cuda')
        shape = torch.tensor(shape, dtype=torch.float32, device='cuda') 

        smpl = SMPL(path='smpl', device='cuda', batch_size=pose.shape[0])

        for garment in garments:
            if os.path.exists(os.path.join(out_folder, f'{id}_{garment}.pc16')):
                continue    

            mesh_static = np.load(os.path.join(out_folder, 'mesh_static', f'{id}_{garment}.npy'))
            mesh_posed = readPC2(os.path.join(out_folder, 'mesh_posed_no_orient', f'{id}_{garment}.pc16'), True)['V']
            mask = np.load(os.path.join(out_folder, 'mesh_mask_orig', f'{id}_{garment}.npy'))

            # remove global orientation
            pose[:,:3] = 0

            unposed = np.zeros((14475, 3), dtype=np.float32)
            unposed[mask] = mesh_static

            unposed = torch.tensor(np.tile(unposed[None,:,:], [mesh_posed.shape[0],1,1]), dtype=torch.float32, device='cuda')
            unposed.requires_grad = True
            mesh_posed = torch.tensor(mesh_posed, dtype=torch.float32, device='cuda')

            optimizer = torch.optim.Adam([unposed], lr=0.01)

            for i in range(5000):
                    optimizer.zero_grad()

                    pred = smpl(pose, shape, unposed)[0]
                    loss : torch.Tensor = ((pred[:,mask,:] - mesh_posed)**2).sum()

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    with torch.inference_mode():
                        euclidean_error = torch.sqrt(((pred[:,mask,:] - mesh_posed)**2).sum(-1)).mean()

                    # Print the loss
                    # if i % 50 == 0:
                    #     print(f"it: {i} loss: {loss:.8f} euclidean_error: {euclidean_error:.8f}")
 
                    if euclidean_error < 0.0000001:
                        break

                    if euclidean_error < 0.0000005 and i > 2000:
                        break

                    if euclidean_error < 0.000001 and i > 4000:
                        break

            unposed = unposed.detach().cpu().numpy()

            gender = int(shape[0].cpu().numpy()[-1] > 0)
            shape_offsets = torch.tensordot(shape[:,:10], smpl.shapedirs[gender], dims=[[1], [2]])
            shape_offsets = shape_offsets.cpu().numpy()[:,smpl.to_keep,:]

            unshaped = (unposed - shape_offsets)[:,mask,:]

            writePC2(os.path.join(out_folder, 'mesh_unposed_no_shape', f'{id}_{garment}.pc16'), unshaped, True)


def create_frames_file(folder, out_folder):
    ids = os.listdir(folder)

    frames_non_skirt = []
    frames_skirt = []
    ids_non_skirt = set()
    ids_skirt = set()

    for id in tqdm(ids):
        garment_dir = os.path.join(folder, id)
        if not os.path.isdir(garment_dir):
            continue

        info = sio.loadmat(os.path.join(folder, id, 'info.mat'))
        garments = [f.replace('.obj','') for f in os.listdir(garment_dir) if f.endswith('.obj')]
        frames = info['poses'].shape[1]

        assert info['poses'].shape[0] == 72

        for garment in garments:
            for i in range(frames):
                if garment == 'Skirt' or garment == 'Dress':
                    frames_skirt.append(Frame(id, i, garment))
                    ids_skirt.add(id)
                else:
                    frames_non_skirt.append(Frame(id, i, garment))
                    ids_non_skirt.add(id)

    ids_skirt = list(ids_skirt)
    ids_non_skirt = list(ids_non_skirt)

    with open(os.path.join(out_folder, 'frames.txt'), 'w') as f:
        for frame in frames_non_skirt:
            if frame.id in ids_non_skirt:
                f.write(f'{frame.id} {frame.frame_nr} {frame.garment}\n')

    with open(os.path.join(out_folder, 'frames_skirt.txt'), 'w') as f:
        for frame in frames_skirt:
            if frame.id in ids_skirt:
                f.write(f'{frame.id} {frame.frame_nr} {frame.garment}\n')


def create_fixed_masks_faces(folder, out_folder):
    ids = sorted(os.listdir(folder))
    os.makedirs(os.path.join(out_folder, 'mesh_fixed'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'mesh_mask'), exist_ok=True)
    F = np.load('smpl/smpl_faces.npy')
    F_skirt = np.load('smpl/smpl_faces_skirt.npy')
    random.shuffle(ids)

    for id in tqdm(ids):
        garment_dir = os.path.join(folder, id)
        if not os.path.isdir(garment_dir):
            continue

        garments = [f.replace('.obj','') for f in os.listdir(garment_dir) if f.endswith('.obj')]

        for garment in garments:
            if os.path.exists(os.path.join(out_folder, f'{id}_{garment}.pc16')):
                continue

            is_skirt = garment in ['Dress', 'Skirt']

            mesh_static = np.load(os.path.join(out_folder, 'mesh_static', f'{id}_{garment}.npy'))
            mask = np.load(os.path.join(out_folder, 'mesh_mask_orig', f'{id}_{garment}.npy'))
            _, faces = _maskFaces(np.zeros((14475, 3)), F_skirt if is_skirt else F, mask)
            _, new_mask = remove_long_edges(mesh_static, faces.astype(np.int32), 0.2 if is_skirt else 0.1)
            new_mask_full = np.zeros((14475), dtype=bool)
            new_mask_full[mask] = new_mask
            _, new_faces = _maskFaces(np.zeros((14475, 3)), F_skirt if is_skirt else F, new_mask_full)

            mesh_new = mesh_static[new_mask]

            flattened_faces = np.asarray([np.concatenate((np.array([3]), f)) for f in new_faces.astype(np.int32)]).flatten()
            m = pv.PolyData(mesh_new, flattened_faces)
            m.fill_holes(0.03, inplace=True, progress_bar=False)

            mesh_new = np.asarray(m.points)
            new_faces = np.asarray(m.faces).reshape([-1, 4])[:, 1:]

            m = tm.Trimesh(vertices=mesh_new, faces=new_faces, process=False, validate=False)
            tm.repair.fix_winding(m)

            mesh_new = np.asarray(m.vertices)
            new_faces = np.asarray(m.faces)

            np.savez_compressed(os.path.join(out_folder, 'mesh_fixed', f'{id}_{garment}.npz'), mask=new_mask_full, mask_convert=new_mask, faces=new_faces)
            np.save(os.path.join(out_folder, 'mesh_mask', f'{id}_{garment}.npy'), new_mask_full)



if __name__ == '__main__':
    folder = '/data/cloth3d/test'
    folder_reg = '/data/cloth3d_registration/test'
    out_folder = '/data/cloth3d_processed/test'

    smpl = SMPL(path='smpl', device='cuda')

    create_mesh_posed_no_orient_and_mask(folder, folder_reg, out_folder)
    create_pose_shape(folder, out_folder)
    create_mesh_static(folder, out_folder)
    create_mesh_body_posed_no_orient(folder, out_folder)
    create_mesh_normals(folder, out_folder)
    create_mesh_unposed_no_shape(folder, out_folder, start=args.start, end=args.end)
    create_frames_file(folder, out_folder)
    create_fixed_masks_faces(folder, out_folder)

import os
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", help = "end of range", default='/data/cloth3d_processed/test')
parser.add_argument("-s", "--start", help = "start of range", default=0, type=int)
parser.add_argument("-e", "--end", help = "end of range", default=1000000, type=int)
parser.add_argument("-n", "--name", help = "name", default='convert')
parser.add_argument("-g", "--gpu", help = "gpu id", default='0')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import glob
import tqdm
import time
import pickle
import cv2
import numpy as np
import trimesh as tm
import scipy.io as sio
import scipy.interpolate as spi
import plotly.express as ex
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from data.cloth3d_io import readPC2

def get_aux_variables(res, m2uv, uv_map, mask):
    garment_mask = np.zeros((uv_map.shape[0]), dtype=bool)
    garment_mask[m2uv[:, 0]] = mask
    garment_mask[m2uv[:, 1]] += mask
    garment_mask = garment_mask[uv_map_mask[type]]

    uv_map_discreet_unique_unique = uv_map_discreet_unique[type][garment_mask, :]

    garment_uv_mask = np.zeros((res * 2, res))
    garment_uv_mask[uv_map_discreet_unique_unique[:, 0], uv_map_discreet_unique_unique[:, 1]] = 255
    garment_uv_mask = cv2.morphologyEx(garment_uv_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (4 * 2 + 1, 4 * 2 + 1), (4, 4)), iterations=4)
    garment_uv_mask_border = garment_uv_mask.astype(bool)
    garment_uv_mask = cv2.dilate(garment_uv_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * 2 + 1, 2 * 2 + 1), (2, 2)), iterations=2).astype(bool)
    garment_uv_mask_border = ~garment_uv_mask_border & garment_uv_mask
    garment_uv_mask = garment_uv_mask * uv_mask[type][:,:,0]

    return garment_uv_mask, garment_uv_mask_border



def convert_to_uv_batch_body(res, m2uv, uv_map, verts, neighbours=15, mask=None, garment_uv_mask=None, garment_uv_mask_border=None):
    batch = verts.shape[0]
    uv_values = np.zeros((batch, uv_map.shape[0], 3))
    uv_values[:, m2uv[:, 0]] = verts
    uv_values[:, m2uv[:, 1]] += verts
    uv_values = uv_values[:, uv_map_mask['non-skirt'], :]
    uv_values /= uv_counter['non-skirt'][np.newaxis, :, np.newaxis]
    return_garment_uv_mask = False

    if mask is not None:
        garment_mask = np.zeros((uv_map.shape[0]), dtype=bool)
        garment_mask[m2uv[:, 0]] = mask
        garment_mask[m2uv[:, 1]] += mask
        garment_mask = garment_mask[uv_map_mask['non-skirt']] #   add garment mask
    else:
        garment_mask = np.sum(uv_values[0,:,:], axis=-1) != 0

    unique_uv_values = uv_values[:, garment_mask, :]
    uv_map_discreet_unique_unique = uv_map_discreet_unique['non-skirt'][garment_mask, :]
    rbf_dest_current = rbf_dest['non-skirt']

    # add back mesh vertices that are not part of the garment
    uv_map_discreet_unique_unique_inv = uv_map_discreet_unique['non-skirt'][np.invert(garment_mask), :]
    rbf_dest_fill = np.zeros((res * 2, res), dtype=bool)
    rbf_dest_fill[uv_map_discreet_unique_unique_inv[:, 0], uv_map_discreet_unique_unique_inv[:, 1]] = True
    
    if garment_uv_mask is None and mask is not None:
        garment_uv_mask = np.zeros((res * 2, res))
        garment_uv_mask[uv_map_discreet_unique_unique[:, 0], uv_map_discreet_unique_unique[:, 1]] = 255
        garment_uv_mask = cv2.morphologyEx(garment_uv_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * 2 + 1, 2 * 2 + 1), (2, 2)), iterations=4)
        garment_uv_mask = cv2.dilate(garment_uv_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * 2 + 1, 2 * 2 + 1), (2, 2)), iterations=2).astype(bool)
        return_garment_uv_mask = True

    if garment_uv_mask is not None:
        rbf_dest_current = (rbf_dest['non-skirt'] + rbf_dest_fill) * garment_uv_mask + garment_uv_mask_border

    rbf_dest_current = np.nonzero(rbf_dest_current)
    rbf_dest_current = np.array(rbf_dest_current).T

    final = []
    for i in range(batch):
        interpolator =  spi.RBFInterpolator(y=uv_map_discreet_unique_unique, d=unique_uv_values[i,:,:], neighbors=neighbours, kernel='linear')

        result = np.zeros((res * 2, res, 3), np.float32)
        result[rbf_dest_current[:, 0], rbf_dest_current[:, 1], :] = interpolator(rbf_dest_current)
        result[uv_map_discreet_unique_unique[:, 0], uv_map_discreet_unique_unique[:, 1], :] = unique_uv_values[i,:,:]

        if garment_uv_mask is not None:
            result *= garment_uv_mask[:,:,np.newaxis]
        
        final.append(result)

    if return_garment_uv_mask:
        return final, garment_uv_mask
    else:
        return final 
    

def convert_to_uv_batch(res, m2uv, uv_map, verts, neighbours=15, mask=None, garment_uv_mask=None, garment_uv_mask_border=None):
    batch = len(verts)

    assert mask is not None
    garment_mask = np.zeros((uv_map.shape[0]), dtype=bool)
    garment_mask[m2uv[:, 0]] = mask
    garment_mask[m2uv[:, 1]] += mask
    garment_mask = garment_mask[uv_map_mask[type]] #   add garment mask

    rbf_dest_current = rbf_dest[type]

    # add back mesh vertices that are not part of the garment
    uv_map_discreet_unique_unique_inv = uv_map_discreet_unique[type][np.invert(garment_mask), :]
    rbf_dest_fill = np.zeros((res * 2, res), dtype=bool)
    rbf_dest_fill[uv_map_discreet_unique_unique_inv[:, 0], uv_map_discreet_unique_unique_inv[:, 1]] = True

    assert not (garment_uv_mask is None and mask is not None)
    assert garment_uv_mask is not None
    rbf_dest_current = (rbf_dest[type] + rbf_dest_fill) * garment_uv_mask + garment_uv_mask_border

    rbf_dest_current = np.nonzero(rbf_dest_current)
    rbf_dest_current = np.array(rbf_dest_current).T

    out = []

    for i in range(batch):
        print('processing frame: ' + str(i))

        uv_values = np.zeros((uv_map.shape[0], 3))
        uv_values[m2uv[:, 0]] = verts[i]
        uv_values[m2uv[:, 1]] += verts[i]
        uv_values = uv_values[uv_map_mask[type], :]
        uv_values /= uv_counter[type][:, np.newaxis]

        unique_uv_values = uv_values[garment_mask, :]
        uv_map_discreet_unique_unique = uv_map_discreet_unique[type][garment_mask, :]

        interpolator =  spi.RBFInterpolator(y=uv_map_discreet_unique_unique, d=unique_uv_values[:,:], neighbors=neighbours, kernel='linear')
        result = np.zeros((res * 2, res, 3), np.float32)
        result[rbf_dest_current[:, 0], rbf_dest_current[:, 1], :] = interpolator(rbf_dest_current)
        result[uv_map_discreet_unique_unique[:, 0], uv_map_discreet_unique_unique[:, 1], :] = unique_uv_values[:,:]

        if garment_uv_mask is not None:
            result *= garment_uv_mask[:,:,np.newaxis]
        
        out.append(result)
    
    return out


def m2uv_array(m2uv):
    result = np.zeros((14475, 2), dtype=np.int32)
    for k, v in m2uv.items():
        if len(v) == 2:
            result[k] = list(v)
        elif len(v) == 1:
            x = v.pop()
            result[k] = [x, x]
        else:
            assert False

    return result


def unique_uv_map_mask(uv_map_discreet):
    mask = np.zeros((uv_map_discreet.shape[0]), dtype=bool)
    temp = np.zeros((uv_res * 2, uv_res))
    for p in uv_map_discreet:
        temp[p[0], p[1]] += 1
    # temp = temp == 1
    for i, p in enumerate(uv_map_discreet):
        if temp[p[0], p[1]] == 1:
            mask[i] = True
        elif temp[p[0], p[1]] == 2:
            mask[i] = True
            temp[p[0], p[1]] = 0
    return mask


if __name__ == '__main__':


    # load necessary files 
    F = {'non-skirt': np.load("smpl/smpl_faces.npy"), 'skirt': np.load("smpl/smpl_faces_skirt.npy")}
    uv_map = {'non-skirt': np.load('utils_data/uv_map.npy'), 'skirt': np.load('utils_data/uv_map_skirt.npy')}
    m2uv = {'non-skirt': np.load('utils_data/m2uv.npy', allow_pickle=True).item(), 'skirt': np.load('utils_data/m2uv_skirt.npy', allow_pickle=True).item()}
    m2uv = {k: m2uv_array(v) for k, v in m2uv.items()}
    uv_mask = {'non-skirt': np.load('utils_data/mask_256x512.npy'), 'skirt': np.load('utils_data/mask_256x512_skirt.npy')}

    #  prepare helper data structures
    uv_res = 256
    uv_map_discreet = dict()
    uv_map_discreet_unique = dict()
    uv_counter = dict()
    rbf_dest = dict()
    uv_map_mask = dict()
    for k,v in uv_mask.items():
        # dilate uv_mask to avoid problems along the borders
        uv_mask[k] = (cv2.resize(v * 255.0, (uv_res, uv_res * 2)) / 255).astype(bool)[:, :, np.newaxis]
        uv_mask[k] = cv2.dilate(v.astype(np.float32), cv2.getStructuringElement(cv2.MORPH_RECT, (2 * 2 + 1, 2 * 2 + 1), (2, 2)), iterations=2).astype(bool)[:, :, np.newaxis]

        # discretize the uv mapping from th econtinous 0,1 space to the discrete 0,uv_res space
        uv_map_discreet[k] = np.floor(uv_map[k] * uv_res).astype(np.int32)
        uv_map_mask[k] = unique_uv_map_mask(uv_map_discreet[k])

        # count how many times each vertex is mapped to a pixel, the verrtices along the seam are mapped twice
        uv_counter[k] = np.zeros((uv_map[k].shape[0]))
        uv_counter[k][m2uv[k][:, 0]] = 1
        uv_counter[k][m2uv[k][:, 1]] += 1
        uv_counter[k] = uv_counter[k][uv_map_mask[k]]

        uv_map_discreet_unique[k] = uv_map_discreet[k][uv_map_mask[k], :]

        # determine the pixels that need to be interpolated
        rbf_dest[k] = np.ones((uv_res * 2, uv_res))
        rbf_dest[k][uv_map_discreet_unique[k][:, 0], uv_map_discreet_unique[k][:, 1]] = 0
        rbf_dest[k][np.invert(uv_mask[k][:,:,0])] = 0


    folder = args.folder

    # find the ids of the meshes that will be processed
    ids = sorted(os.listdir(folder + '/mesh_body_posed_no_orient'))[args.start:args.end]
    ids = [id.split('.')[0].split('_')[0] for id in ids]

    # create the output folders
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(folder, 'uv_posed_no_orient'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'uv_static'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'uv_normals'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'uv_unposed_no_shape'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'uv_body_posed_no_orient'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'uv_mask'), exist_ok=True)

    for id in tqdm.tqdm(ids):
        garments = glob.glob(os.path.join(folder, 'mesh_mask_orig', f'{id}_*'))
        garments = [garment.split('/')[-1].split('.')[0].split('_')[-1] for garment in garments]

        # loop through each garment of the outfit and process it
        for k, garment in enumerate(garments):
            # load all the meshes that will be converted
            mask = np.load(os.path.join(folder, 'mesh_mask_orig', f'{id}_{garment}.npy'))
            x = readPC2(os.path.join(folder, 'mesh_posed_no_orient', f'{id}_{garment}.pc16'), True)['V']
            mesh_posed = np.zeros((x.shape[0], 14475, 3), dtype=np.float32)
            mesh_posed[:,mask, :] = x
            mesh_unposed = np.zeros((x.shape[0], 14475, 3), dtype=np.float32)
            mesh_unposed[:,mask, :] = readPC2(os.path.join(folder, 'mesh_unposed_no_shape' ,f'{id}_{garment}.pc16'), True)['V']
            mesh_static = np.zeros((14475, 3), dtype=np.float32)
            mesh_static[mask, :] = np.load(os.path.join(folder, 'mesh_static', f'{id}_{garment}.npy'))
            mesh_normals = np.zeros((x.shape[0], 14475, 3), dtype=np.float32)
            mesh_normals[:,mask, :] = readPC2(os.path.join(folder, 'mesh_normals', f'{id}_{garment}.pc16'), True)['V']
            if k == 0: # only need to convert body once per outfit
                body_posed = readPC2(os.path.join(folder, 'mesh_body_posed_no_orient', f'{id}.pc16'), True)['V']

            type = 'skirt' if garment in ['Dress', 'Skirt'] else 'non-skirt'

            # convert the meshes to uv space
            uv_mask_local, garment_uv_mask_border = get_aux_variables(uv_res, m2uv[type], uv_map[type], mask)
            uv_static = convert_to_uv_batch(uv_res, m2uv[type], uv_map[type], mesh_static[None, :, :], neighbours=3, mask=mask, garment_uv_mask=uv_mask_local, garment_uv_mask_border=garment_uv_mask_border)
            uv_static = uv_static[0].astype(np.float32)

            uv_posed = convert_to_uv_batch(uv_res, m2uv[type], uv_map[type], mesh_posed, neighbours=3, mask=mask, garment_uv_mask=uv_mask_local, garment_uv_mask_border=garment_uv_mask_border)
            uv_unposed = convert_to_uv_batch(uv_res, m2uv[type], uv_map[type], mesh_unposed, neighbours=3, mask=mask, garment_uv_mask=uv_mask_local, garment_uv_mask_border=garment_uv_mask_border)
            uv_normals = convert_to_uv_batch(uv_res, m2uv[type], uv_map[type], mesh_normals, neighbours=3, mask=mask, garment_uv_mask=uv_mask_local, garment_uv_mask_border=garment_uv_mask_border)
            if k == 0: # only need to convert body once per outfit
                uv_body_posed = convert_to_uv_batch_body(uv_res, m2uv['non-skirt'], uv_map['non-skirt'], body_posed, neighbours=3, mask=None, garment_uv_mask=None, garment_uv_mask_border=garment_uv_mask_border)

            # save the uv maps
            np.savez_compressed(os.path.join(folder, 'uv_static',  f'{id}_{garment}.npz'), uv_static=uv_static)
            np.savez_compressed(os.path.join(folder, 'uv_mask',  f'{id}_{garment}.npz'), uv_mask=uv_mask_local)
            for i in range(len(uv_posed)):
                np.savez_compressed(os.path.join(folder, 'uv_posed_no_orient',  f'{id}_{i}_{garment}.npz'), uv_posed=uv_posed[i])
            for i in range(len(uv_unposed)):
                np.savez_compressed(os.path.join(folder, 'uv_unposed_no_shape',  f'{id}_{i}_{garment}.npz'), uv_unposed=uv_unposed[i])
            for i in range(len(uv_normals)):
                np.savez_compressed(os.path.join(folder, 'uv_normals',  f'{id}_{i}_{garment}.npz'), uv_normals=uv_normals[i])
            if k == 0: # only need to convert body once per outfit
                for i in range(len(uv_body_posed)):
                    np.savez_compressed(os.path.join(folder, 'uv_body_posed_no_orient',  f'{id}_{i}.npz'), uv_body_posed=uv_body_posed[i])


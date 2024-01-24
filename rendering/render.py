#!/usr/bin/env python
import sys 
import os
import glob

sys.path.append(".")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

from rendering.renderer import GarmentRenderer

# try:
#     index = sys.argv.index("--path")
#     path = sys.argv[index + 1]
# except ValueError:
#     print("Usage: blender --background rendering/scene.blend --python rendering/render.py --path <path_to_meshes>")

# import scipy.io as sio
# import numpy as np
# import trimesh as tm
# from stl import mesh as mesh_stl
# from utils.utils import _maskFaces
# from data.cloth3d_io import readPC2, readPC2Frame

id = '00001'
garment_type = 'Top'
path = '/code/temp'
# folder_reg = '/data/cloth3d_registration/train'
# folder = '/data/cloth3d_processed'

# garments = sorted(glob.glob(f'/output/pyr/{id}*{garment_type}.npz'))
# mask_idx = sio.loadmat(os.path.join(folder_reg, id, garment_type + '_mask.mat'))['mask_idx'].reshape((-1)) - 1
# faces = sio.loadmat(os.path.join(folder_reg, id, garment_type + '.mat'))['F']
# mask = np.zeros((14475), dtype=bool)
# mask[mask_idx] = True
# fix_data = np.load(f'{folder}/mesh_fixed/{id}_{garment_type}.npz')
# F = np.load("smpl/smpl_faces.npy")
# _, garment_faces = _maskFaces(np.zeros((14475, 3), dtype=np.float32), F, mask)
# garment_faces = garment_faces.astype(np.uint32)


# def save_stl(file, mesh, faces):
#     from stl import mesh as mesh_stl
#     obj = mesh_stl.Mesh(np.zeros(faces.shape[0], dtype=mesh_stl.Mesh.dtype))
#     for i, f in enumerate(faces):
#         for j in range(3):
#             obj.vectors[i][j] = mesh[f[j],:]    

#     obj.save(file)

# for i, garment in enumerate(garments):
#     garment = np.load(garment)['arr_0'][mask,:]
#     mesh_body_posed_no_orient = readPC2Frame(os.path.join(folder, 'mesh_body_posed_no_orient', f'{id}.pc16'), i, True)
#     mesh = tm.Trimesh(vertices=garment, faces=garment_faces, process=False, validate=False)
#     mesh.export(os.path.join('temp', f'{id}_{i:04}_{garment_type}.obj'))
#     mesh_body = tm.Trimesh(vertices=mesh_body_posed_no_orient, faces=F, process=False, validate=False)
#     mesh_body.export(os.path.join('temp', f'{id}_{i:04}_body.obj'))
#     # save_stl(os.path.join('temp', f'{id}_{i}_{garment_type}.stl'), garment, garment_faces)

#     pass

renderer = GarmentRenderer(
    cloth_paths=sorted(glob.glob(os.path.join(path, f"{id}*{garment_type}.obj"))),
    body_paths=sorted(glob.glob(os.path.join(path, f"{id}*body.obj"))),
    cloth_material="ClothMaterialYellow",
    body_material="MannequinMaterialDark",
    export_path=os.path.join(path, "render")
)

renderer.render(resolution_percentage=100, fov=50, start_frame=0, end_frame=None)
renderer.generate_video()


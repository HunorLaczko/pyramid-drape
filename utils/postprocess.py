import torch
import numpy as np
import scipy.io as sio
from pytorch3d.structures import Meshes

from utils.utils import _nearest_neighbour

# based on hbertiche/PBNS
to_keep = sio.loadmat('smpl/keep_remove.mat')['keep'].reshape(-1)
F_fix_coll = torch.tensor(np.tile(np.load("smpl/smpl_faces.npy")[None,:,:], [4,1,1]), device='cuda')
smpl_to_keep_orig_post = sio.loadmat('smpl/keep_remove.mat')['keep'].reshape(-1)
smpl_to_keep_orig_post = smpl_to_keep_orig_post[smpl_to_keep_orig_post < 6890]
F_fix_coll_orig = torch.tensor(np.tile(np.load("smpl/smpl_faces_orig.npy")[:6890,:][None,:,:], [4,1,1]), device='cuda')[:,smpl_to_keep_orig_post,:]


def fix_collision(V: torch.Tensor, B: torch.Tensor, mask: torch.Tensor, thr=.001):
    if V.shape[1] == 14475:
        F_fix_coll_local = F_fix_coll
    elif V.shape[1] == 3638:
        F_fix_coll_local = F_fix_coll_orig
    else:
        raise ValueError("V must have 14475 or 3638 vertices")
    if V.shape[0] == 1:
        F_fix_coll_local = F_fix_coll_local[:1]

    mesh = Meshes(B, F_fix_coll_local)
    B_vn = torch.stack(mesh.verts_normals_list())
   
    idx = _nearest_neighbour(V, B)
    B_idx = torch.stack([B[i,idx[i],:] for i in range(V.shape[0])])
    D = V - B_idx
    b_vn = torch.stack([B_vn[i,idx[i],:] for i in range(V.shape[0])])
    dot = torch.einsum('abc,abc->ab', D, b_vn)
    loss = torch.minimum(dot - thr, torch.zeros_like(dot))  * mask
    B_offset_closest = B_idx + (b_vn / 2e2)
    output = torch.where(loss[:,:,None] < 0, B_offset_closest, V)

    return output
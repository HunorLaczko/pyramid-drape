import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
from torch.utils.data import DataLoader
from stl import mesh as mesh_stl
from pathlib import Path
from stl import mesh as mesh_stl
from plotly.subplots import make_subplots
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode
from torch_cluster import knn


class Accumulator():
    def __init__(self):
        self.dict = dict()

    def update(self, dict):
        for key in dict.keys():
            if key in self.dict.keys():
                self.dict[key].append(dict[key])
            else:
                self.dict[key] = [dict[key]]

    def get(self):
        result = dict()
        for key in self.dict.keys():
            result[key] = np.mean(self.dict[key])
        return result

    def reset(self):
        self.dict = dict()


class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch


class Frame:
    def __init__(self, id: str ='', frame_nr: int= 0, garment: str='', frame = None, frame_str: str = ''):
        if frame is not None:
            self.id: str = frame.id
            self.frame_nr: int = frame.frame_nr
            self.garment: str = frame.garment
            assert id == '' and frame_nr == 0 and garment == '' and frame_str == ''
        elif frame_str != '':
            if ' ' in frame_str:
                self.id: str = frame_str.split()[0]
                self.frame_nr: int = int(frame_str.split()[1])
                self.garment: str = frame_str.split()[2]
            else:
                self.id: str = frame_str.split('_')[0]
                self.frame_nr: int = int(frame_str.split('_')[1])
                self.garment: str = frame_str.split('_')[2]
            assert id == '' and frame_nr == 0 and garment == '' and frame is None
        else:
            self.id: str = id
            self.frame_nr: int = frame_nr
            self.garment: str = garment

    def __hash__(self):
        return hash((self.id, self.frame_nr, self.garment))

    def __eq__(self, other):
        return (self.id, self.frame_nr, self.garment) == (other.id, other.frame_nr, self.garment)
    
    def __str__(self):
            return f'{self.id}_{self.frame_nr}_{self.garment}'
    

def text_to_html(lines):  
    html = ''
    for line in lines:
        html += "<pre>" + line + "</pre>"
    return html


def save_stl(file, mesh, faces):
    obj = mesh_stl.Mesh(np.zeros(faces.shape[0], dtype=mesh_stl.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            obj.vectors[i][j] = mesh[f[j],:]    

    obj.save(file)


def create_temp_frames(folder: str, frame: Frame):
    with open(Path(folder) / 'frames_temp.txt', 'w') as f:
        f.writelines([f'{frame}\n' for _ in range(4)])


def pickle_load(file):
    """
    Load a pickle file.
    """
    with open(file, 'rb') as f:
        loadout = pickle.load(f)

    return loadout


def pickle_dump(loadout, file):
    """
    Dump a pickle file. Create the directory if it does not exist.
    """
    os.makedirs(os.path.dirname(str(file)), exist_ok=True)

    with open(file, 'wb') as f:
        pickle.dump(loadout, f)



def m2uv_array(m2uv, is_smpl_orig=False):
    result = np.zeros((14475 if not is_smpl_orig else 3638, 2), dtype=np.int32)
    for k, v in m2uv.items():
        if len(v) == 2:
            result[k] = list(v)
        elif len(v) == 1:
            x = v.pop()
            result[k] = [x, x]
        else:
            assert False

    return result


class UV2Mesh(nn.Module):
    def __init__(self, res=256, is_skirt=False, is_smpl_orig=False):
        super(UV2Mesh, self).__init__()
        self.res = res
        uv_map = np.load('utils_data/uv_map' + ('_skirt' if is_skirt else '') + ('_orig' if is_smpl_orig else '') + '.npy')
        self.uv_map_discreet = np.floor(uv_map * self.res).astype(np.int32)
        m2uv = np.load('utils_data/m2uv' + ('_skirt' if is_skirt else '') + ('_orig' if is_smpl_orig else '') + '.npy', allow_pickle=True).item()
        m2uv = m2uv_array(m2uv, is_smpl_orig=is_smpl_orig)
        self.uv_pixels = self.uv_map_discreet[m2uv]
        self.smpl_res = 14475 if not is_smpl_orig else 3638

        self.resizer = Resize(size=(512, 256), interpolation=InterpolationMode.BILINEAR)


    def uv_to_mesh(self, uv):
        if uv.shape[1] == 3 or uv.shape[1] == 7:
            uv = torch.permute(uv, [0,2,3,1])
        if uv.shape[1] != 512:
            uv = torch.permute(self.resizer(torch.permute(uv, [0,3,1,2])), [0,2,3,1])

        mesh = torch.mean(uv[:, self.uv_pixels[:,:,0], self.uv_pixels[:,:,1]].float(), dim=2)
        return mesh


    def __call__(self, uv):
        x = self.uv_to_mesh(uv)
        return x


def _maskFaces(V, F, mask, return_unfixed=False):
    V_mask = mask
    F_mask = np.take(V_mask, F.reshape(-1), axis=0)
    F_mask = F_mask.reshape((-1, 3)).sum(axis=1) == 3

    V_ind = np.zeros(V_mask.shape)
    V_ind[V_mask] = np.arange(V_mask.sum())
    F_new = np.take(V_ind, F[F_mask, :].reshape(-1), axis=0).reshape((-1, 3))
    V_new = V[V_mask, :]

    if return_unfixed:
        return V_new, F_new, F[F_mask, :]
    else:
        return V_new, F_new


def maskFacesBatch(V, F, mask):
    Vs = []
    Fs = []
    for i in range(mask.shape[0]):
        v, f = _maskFaces(V[i], F, mask[i])
        Vs.append(v)
        Fs.append(f)
    return Vs, Fs


def plot_3d(mesh_gt, mesh_pred_unposed, faces, collision, collision_faces, collision_color):
    fig = make_subplots(rows=1, cols=3, start_cell="top-left", specs = [ [{ 'type': 'scene' } for _ in range(3)] ])

    fig.add_trace(go.Mesh3d(
                # 8 vertices of a cube
                x=mesh_gt[:,0],
                y=mesh_gt[:,2],
                z=mesh_gt[:,1],
                # i, j and k give the vertices of triangles
                i = faces[:,0],
                j = faces[:,1],
                k = faces[:,2]
            ),
            row=1, col=1)
    
    fig.add_trace(go.Mesh3d(
                # 8 vertices of a cube
                x=mesh_pred_unposed[:,0],
                y=mesh_pred_unposed[:,2],
                z=mesh_pred_unposed[:,1],
                # i, j and k give the vertices of triangles
                i = faces[:,0],
                j = faces[:,1],
                k = faces[:,2]
            ),
            row=1, col=2)

    fig.add_trace(go.Mesh3d(
                # 8 vertices of a cube
                x=collision[:,0],
                y=collision[:,2],
                z=collision[:,1],
                # i, j and k give the vertices of triangles
                i = collision_faces[:,0],
                j = collision_faces[:,1],
                k = collision_faces[:,2],
                vertexcolor=collision_color
            ),
            row=1, col=3)


    fig.update_layout(scene_aspectmode='manual', scene_aspectratio=dict(x=1, y=1, z=1),
                    scene2_aspectmode='manual', scene2_aspectratio=dict(x=1, y=1, z=1),
                    scene3_aspectmode='manual', scene3_aspectratio=dict(x=1, y=1, z=1),
                    )
    helper = dict(xaxis = dict(nticks=4, range=[-1,1],),
                yaxis = dict(nticks=4, range=[-1,1],),
                zaxis = dict(nticks=4, range=[-1,1],),)
    fig.update_layout(scene = helper,
                        scene2 = helper,
                        scene3 = helper,
                        )

    return fig


def plot_3d_collision(gt, collision_1, collision_2, collision_faces, collision_color):
    fig = make_subplots(rows=1, cols=3, start_cell="top-left", specs = [ [{ 'type': 'scene' } for _ in range(3)] ])

    fig.add_trace(go.Mesh3d(
                x=gt[:,0],
                y=gt[:,2],
                z=gt[:,1],
                i = collision_faces[:,0],
                j = collision_faces[:,1],
                k = collision_faces[:,2],
                vertexcolor=collision_color
            ),
            row=1, col=1)

    fig.add_trace(go.Mesh3d(
                x=collision_1[:,0],
                y=collision_1[:,2],
                z=collision_1[:,1],
                i = collision_faces[:,0],
                j = collision_faces[:,1],
                k = collision_faces[:,2],
                vertexcolor=collision_color
            ),
            row=1, col=2)

    fig.add_trace(go.Mesh3d(
                x=collision_2[:,0],
                y=collision_2[:,2],
                z=collision_2[:,1],
                i = collision_faces[:,0],
                j = collision_faces[:,1],
                k = collision_faces[:,2],
                vertexcolor=collision_color
            ),
            row=1, col=3)


    fig.update_layout(scene_aspectmode='manual', scene_aspectratio=dict(x=1, y=1, z=1),
                    scene2_aspectmode='manual', scene2_aspectratio=dict(x=1, y=1, z=1),
                    scene3_aspectmode='manual', scene3_aspectratio=dict(x=1, y=1, z=1),
                    )
    helper = dict(xaxis = dict(nticks=4, range=[-1,1],),
                yaxis = dict(nticks=4, range=[-1,1],),
                zaxis = dict(nticks=4, range=[-1,1],),)
    fig.update_layout(scene = helper,
                        scene2 = helper,
                        scene3 = helper,
                        )

    return fig



def get_summary_str(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())

    return [f'all_params: {total_params}, trainable: {trainable_params}']


def get_summary_html(model):
    lines = get_summary_str(model)
    html = ''
    for line in lines:
        html += "<pre>" + line + "</pre>"
    return html


def save_stl(file, mesh, faces):
    obj = mesh_stl.Mesh(np.zeros(faces.shape[0], dtype=mesh_stl.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            obj.vectors[i][j] = mesh[f[j],:]    

    obj.save(file)


def write_obj(file, vertices, faces, colors):
    with open(file, 'w') as file:
        for v, c in zip(vertices, colors):
            line = 'v ' + ' '.join([str(_) for _ in v]) + ' ' + ' '.join([str(_) for _ in c]) + '\n'
            file.write(line)
        for f in faces:
            line = 'f ' + ' '.join([str(_) for _ in f]) + '\n'
            file.write(line)


def group_inputs_for_levels(batch, config):
    levels_input = []
    for res in config['pyramid']['resolutions']:
        res_modifier = '_' + str(res) if res != 512 else ''
        uv_orig = batch['uv_unposed' + res_modifier]
        levels_input.append([uv_orig, batch['uv_static' + res_modifier], batch['uv_body_posed' + res_modifier], batch['uv_normals' + res_modifier], batch['uv_mask' + res_modifier], batch['uv_mask']])
    conds = [batch['uv_static'], batch['uv_body_posed'], batch['uv_normals']]
    return levels_input, conds


def remove_long_edges(vertices, faces, limit=0.1):
    # Compute edge lengths
    edges = np.vstack((faces[:,[0,1]], faces[:,[1,2]], faces[:,[2,0]]))
    edge_lengths = np.linalg.norm(vertices[edges[:,0]] - vertices[edges[:,1]], axis=1)

    # Find edges longer than limit
    long_edges = edges[edge_lengths > limit]

    # Find vertices that are part of long edges
    vertices_to_remove = np.unique(long_edges)

    # Remove vertices and update faces
    mask = np.ones(vertices.shape[0], dtype=bool)
    mask[vertices_to_remove] = False
    new_vertices = vertices[mask]


    return new_vertices, mask


def load_model(model: torch.nn.Module, checkpoint: str, partial: bool = False):
    data = torch.load(checkpoint + '.pt')['model']
    model.load_state_dict(data, strict=not partial)
    print(f'Loaded model from {checkpoint}')


def _nearest_neighbour(V: torch.Tensor, B: torch.Tensor):
    # flatten V and B along the batch dimension, create a batch_x and batch_y variable to restore the batch dimension after the nearest neighbour search using pytorch_geomtry knn function
    batch_size = V.shape[0]
    vertices_count = V.shape[1]
    batch_x = torch.arange(batch_size, device=V.device).repeat_interleave(V.shape[1])
    V = V.reshape(-1, V.shape[-1])
    B = B.reshape(-1, B.shape[-1])
    idx = knn(B.to(V.dtype), V, k=1, batch_x=batch_x, batch_y=batch_x)[1]
    idx = (idx - batch_x * vertices_count).reshape(batch_size, vertices_count)
    return idx
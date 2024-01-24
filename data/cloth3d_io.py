# from cloth3d starterkit
import os
import scipy
import numpy as np
import scipy.io as sio
from struct import pack, unpack


"""
Reads PC2 files, and proposed format PC16 files
Inputs:
- file: path to .pc2/.pc16 file
- float16: False for PC2 files, True for PC16
Output:
- data: dictionary with .pc2/.pc16 file data
NOTE: 16-bit floats lose precision with high values (positive or negative),
      we do not recommend using this format for data outside range [-2, 2]
"""
def readPC2(file, float16=False):
    assert file.endswith('.pc2') and not float16 or file.endswith('.pc16') and float16, 'File format not consistent with specified input format'
    data = {}
    bytes = 2 if float16 else 4
    dtype = np.float16 if float16 else np.float32
    with open(file, 'rb') as f:
        # Header
        data['sign'] = f.read(12)
        # data['version'] = int.from_bytes(f.read(4), 'little')
        data['version'] = unpack('<i', f.read(4))[0]
        # Num points
        # data['nPoints'] = int.from_bytes(f.read(4), 'little')
        data['nPoints'] = unpack('<i', f.read(4))[0]
        # Start frame
        data['startFrame'] = unpack('f', f.read(4))
        # Sample rate
        data['sampleRate'] = unpack('f', f.read(4))
        # Number of samples
        # data['nSamples'] = int.from_bytes(f.read(4), 'little')
        data['nSamples'] = unpack('<i', f.read(4))[0]
        # Animation data
        size = data['nPoints']*data['nSamples']*3*bytes
        data['V'] = np.frombuffer(f.read(size), dtype=dtype).astype(np.float32)
        data['V'] = data['V'].reshape(data['nSamples'], data['nPoints'], 3)
        
    return data
    
"""
Reads an specific frame of PC2/PC16 files
Inputs:
- file: path to .pc2/.pc16 file
- frame: number of the frame to read
- float16: False for PC2 files, True for PC16
Output:
- T: mesh vertex data at specified frame
"""
def readPC2Frame(file, frame, float16=False):
    assert file.endswith('.pc2') and not float16 or file.endswith('.pc16') and float16, 'File format not consistent with specified input format'
    assert frame >= 0 and isinstance(frame,int), 'Frame must be a positive integer'
    bytes = 2 if float16 else 4
    dtype = np.float16 if float16 else np.float32
    with open(file,'rb') as f:
        # Num points
        f.seek(16)
        # nPoints = int.from_bytes(f.read(4), 'little')
        nPoints = unpack('<i', f.read(4))[0]
        # Number of samples
        f.seek(28)
        # nSamples = int.from_bytes(f.read(4), 'little')
        nSamples = unpack('<i', f.read(4))[0]
        if frame > nSamples:
            print("Frame index outside size")
            print("\tN. frame: " + str(frame))
            print("\tN. samples: " + str(nSamples))
            return
        # Read frame
        size = nPoints * 3 * bytes
        f.seek(size * frame, 1) # offset from current '1'
        T = np.frombuffer(f.read(size), dtype=dtype).astype(np.float32)
    return T.reshape(nPoints, 3)

"""
Writes PC2 and PC16 files
Inputs:
- file: path to file (overwrites if exists)
- V: 3D animation data as a three dimensional array (N. Frames x N. Vertices x 3)
- float16: False for writing as PC2 file, True for PC16
This function assumes 'startFrame' to be 0 and 'sampleRate' to be 1
NOTE: 16-bit floats lose precision with high values (positive or negative),
      we do not recommend using this format for data outside range [-2, 2]
"""
def writePC2(file, V, float16=False):
    assert file.endswith('.pc2') and not float16 or file.endswith('.pc16') and float16, 'File format not consistent with specified input format'
    if float16: V = V.astype(np.float16)
    else: V = V.astype(np.float32)
    with open(file, 'wb') as f:
        # Create the header
        headerFormat='<12siiffi'
        headerStr = pack(headerFormat, b'POINTCACHE2\0',
                        1, V.shape[1], 0, 1, V.shape[0])
        f.write(headerStr)
        # Write vertices
        f.write(V.tobytes())

"""
Reads OBJ files
Only handles vertices, faces and UV maps
Input:
- file: path to .obj file
Outputs:
- V: 3D vertices
- F: 3D faces
- Vt: UV vertices
- Ft: UV faces
Correspondence between mesh and UV map is implicit in F to Ft correspondences
If no UV map data in .obj file, it shall return Vt=None and Ft=None
"""
def readOBJ(file):
    V, Vt, F, Ft = [], [], [], []
    with open(file, 'r') as f:
        T = f.readlines()
    for t in T:
        # 3D vertex
        if t.startswith('v '):
            v = [float(n) for n in t.replace('v ','').split(' ')]
            V += [v]
        # UV vertex
        elif t.startswith('vt '):
            v = [float(n) for n in t.replace('vt ','').split(' ')]
            Vt += [v]
        # Face
        elif t.startswith('f '):
            idx = [n.split('/') for n in t.replace('f ','').split(' ')]
            f = [int(n[0]) - 1 for n in idx]
            F += [f]
            # UV face
            if '/' in t:
                f = [int(n[1]) - 1 for n in idx]
                Ft += [f]
    V = np.array(V, np.float32)
    Vt = np.array(Vt, np.float32)
    if Ft: assert len(F) == len(Ft), 'Inconsistent .obj file, mesh and UV map do not have the same number of faces' 
    else: Vt, Ft = None, None
    return V, F, Vt, Ft



"""
Writes OBJ files
Only handles vertices, faces and UV maps
Inputs:
- file: path to .obj file (overwrites if exists)
- V: 3D vertices
- F: 3D faces
- Vt: UV vertices
- Ft: UV faces
Correspondence between mesh and UV map is implicit in F to Ft correspondences
If no UV map data as input, it will write only 3D data in .obj file
"""
def writeOBJ(file, V, F, Vt=None, Ft=None):
    if not Vt is None:
        assert len(F) == len(Ft), 'Inconsistent data, mesh and UV map do not have the same number of faces'
        
    with open(file, 'w') as file:
        # Vertices
        for v in V:
            line = 'v ' + ' '.join([str(_) for _ in v]) + '\n'
            file.write(line)
        # UV verts
        if not Vt is None:
            for v in Vt:
                line = 'vt ' + ' '.join([str(_) for _ in v]) + '\n'
                file.write(line)
        # 3D Faces / UV faces
        if Ft:
            F = [[str(i+1)+'/'+str(j+1) for i,j in zip(f,ft)] for f,ft in zip(F,Ft)]
        else:
            F = [[str(i + 1) for i in f] for f in F]		
        for f in F:
            line = 'f ' + ' '.join(f) + '\n'
            file.write(line)


def loadInfo(filename):
    '''
    this function should be called instead of direct sio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    del data['__globals__']
    del data['__header__']
    del data['__version__']
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        elif isinstance(elem, np.ndarray) and np.any([isinstance(item, sio.matlab.mio5_params.mat_struct) for item in elem]):
            dict[strg] = [None] * len(elem)
            for i,item in enumerate(elem):
                if isinstance(item, sio.matlab.mio5_params.mat_struct):
                    dict[strg][i] = _todict(item)
                else:
                    dict[strg][i] = item
        else:
            dict[strg] = elem
    return dict


# can i cache map?
def read_posed_garment_frame(folder, id, garment, frame, map=None, mask=None, shape=None, shapedirs=None, pose=None, unshape=False, deorient=False):
    pc16_path = os.path.join(folder, id, garment + '.pc16')
    verts = readPC2Frame(pc16_path, frame, True)

    if map is None:
        map = sio.loadmat(os.path.join(folder, id, garment + '_map.mat'))['map'].reshape((-1))
    verts = verts[map, :]

    if deorient:
        R = scipy.spatial.transform.Rotation.from_euler('xyz', pose[:3]).inv()
        # R = scipy.spatial.transform.Rotation.from_euler([0,0,-zrot])
        verts = R.apply(verts)

    return verts


# can i cache map?
def read_posed_garment(folder, id, garment, map=None, mask=None, shape=None, shapedirs=None, poses=None, unshape=False, deorient=False):
    # body_posed = readPC2(os.path.join(folder, id, 'Body_posed.pc16'), True)['V']
    # offsets = readPC2(os.path.join(folder, id, garment + '_offsets_dynamic.pc16'), True)['V']
    # verts = (body_posed + offsets) * mask[:, None]
    pc16_path = os.path.join(folder, id, garment + '.pc16')
    # info = loadInfo(os.path.join(folder, id, 'info.mat'))
    verts = readPC2(pc16_path, True)['V']
    if map is None:
        map = sio.loadmat(os.path.join(folder, id, garment + '_map.mat'))['map'].reshape((-1))
    verts = verts[:, map, :]

    if deorient:
        for i in range(verts.shape[0]):
            R = scipy.spatial.transform.Rotation.from_euler('xyz', poses[:3, i]).inv()
            # R = scipy.spatial.transform.Rotation.from_euler([0,0,-zrot])
            verts[i] = R.apply(verts[i])

    return verts


def read_garment_mask(folder, id, garment, return_raw=False):
    raw_mask = sio.loadmat(os.path.join(folder, id, garment + '_mask.mat'))['mask'].reshape((-1))
    if not return_raw:
        mask = raw_mask > 0
    else:
        mask = raw_mask
    return mask


def read_garment_template(folder, id, garment, mask=None):
    offsets_static = sio.loadmat(os.path.join(folder, id, garment + '_offsets_static.mat'))['label'][:,:3]
    body_static = sio.loadmat(os.path.join(folder, id, 'info_body.mat'))['B']
    if mask is None:
        mask = read_garment_mask(folder, id, garment)
    verts_static = (body_static + offsets_static) * mask[:, np.newaxis]
    return verts_static


def read_posed_body_frame(folder, id, frame, pose=None, deorient=False):
    pc16_path = os.path.join(folder, id, 'Body_posed.pc16')
    verts = readPC2Frame(pc16_path, frame, True)

    if deorient:
        R = scipy.spatial.transform.Rotation.from_euler(pose[:3]).inv()
        # R = scipy.spatial.transform.Rotation.from_euler([0,0,-zrot])
        verts = R.apply(verts)
        
    return verts


def read_posed_body(folder, id, poses=None, deorient=False):
    pc16_path = os.path.join(folder, id, 'Body_posed.pc16')
    verts = readPC2(pc16_path, True)['V']

    if deorient:
        for i in range(verts.shape[0]):
            R = scipy.spatial.transform.Rotation.from_euler(poses[:3, i]).inv()
            # R = scipy.spatial.transform.Rotation.from_euler([0,0,-zrot])
            verts[i] = R.apply(verts[i])
        
    return verts



# import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

# import plotly.graph_objects as go

# def plot_3d(mesh, faces):
#     fig = go.Figure()


#     fig.add_trace(go.Mesh3d(
#                 # 8 vertices of a cube
#                 x=mesh[:,0],
#                 y=mesh[:,1],
#                 z=mesh[:,2],
#                 # i, j and k give the vertices of triangles
#                 i = faces[:,0],
#                 j = faces[:,1],
#                 k = faces[:,2]
#             ))
    


#     fig.update_layout(scene_aspectmode='manual', scene_aspectratio=dict(x=1, y=1, z=1),)
#     helper = dict(xaxis = dict(nticks=4, range=[-1,1],),
#                 yaxis = dict(nticks=4, range=[-1,1],),
#                 zaxis = dict(nticks=4, range=[-1,1],),)
#     fig.update_layout(scene = helper)

#     return fig


def quads2tris(F):
    out = []
    for f in F:
        if len(f) == 3: out += [f]
        elif len(f) == 4: out += [[f[0],f[1],f[2]],
                                [f[0],f[2],f[3]]]
        else: print("This should not happen...")
    return np.array(out, np.int32)

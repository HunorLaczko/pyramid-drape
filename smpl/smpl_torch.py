# based on https://github.com/CalciferZh/SMPL/blob/master/smpl_torch_batch.py

import os
import sys
import pickle
import torch
import numpy as np
import scipy.io as sio
from time import time
from torch.nn import Module

class SMPL(Module):
  def __init__(self, device=None, path='', orig_smpl=False, is_skirt=False, batch_size=4):
    
    super(SMPL, self).__init__()
    self.J_regressor = []
    self.v_template = []
    self.shapedirs = []
    self.posedirs = []
    self.weights = []
    self.is_skirt = is_skirt
    with open(os.path.join(path, 'super_model_f.pkl'), 'rb') as f:
        female = pickle.load(f, encoding="latin1")
        self.J_regressor.append(torch.tensor(np.array(female['J_regressor'].todense(), dtype=np.float32)))
        self.v_template.append(torch.tensor(female['v_template'], dtype=torch.float32))
        self.posedirs.append(torch.tensor(female['posedirs'], dtype=torch.float32))
        self.shapedirs.append(torch.tensor(female['shapedirs'], dtype=torch.float32))
        self.weights.append(torch.tensor(female['weights'], dtype=torch.float32))
    with open(os.path.join(path, 'super_model_m.pkl'), 'rb') as f:
        male = pickle.load(f, encoding="latin1")
        self.J_regressor.append(torch.tensor(np.array(male['J_regressor'].todense(), dtype=np.float32)))
        self.v_template.append(torch.tensor(male['v_template'], dtype=torch.float32))
        self.posedirs.append(torch.tensor(male['posedirs'], dtype=torch.float32))
        self.shapedirs.append(torch.tensor(male['shapedirs'], dtype=torch.float32))
        self.weights.append(torch.tensor(male['weights'], dtype=torch.float32))
        kintree_table = male['kintree_table']
        id_to_col = {kintree_table[1, i]: i for i in range(kintree_table.shape[1])}
        self.parent = {i: id_to_col[kintree_table[0, i]] for i in range(1, kintree_table.shape[1])}
    self.J_regressor = torch.stack(self.J_regressor, axis=0).to(device)
    self.v_template = torch.stack(self.v_template, axis=0).to(device)
    self.shapedirs = torch.stack(self.shapedirs, axis=0).to(device)
    self.posedirs = torch.stack(self.posedirs, axis=0).to(device)
    self.weights = torch.stack(self.weights, axis=0).to(device)

    # Topology no head/hands/feet
    F = sio.loadmat(os.path.join(path, 'ssmpl_faces.mat'))
    self.f = np.array(F['faces_default'])
    self.f_skirt = np.array(F['faces_skirt'])
    self.nn = np.array(sio.loadmat(os.path.join(path, 'NN.mat')))
    self.to_keep = sio.loadmat(os.path.join(path, 'keep_remove.mat'))['keep'].reshape(-1)

    self.weights_all = self.weights
    self.weights = self.weights[:,self.to_keep,:]
    
    self.shape = (1,27578,3)
    self.device = device
    self.batch_size = batch_size

    self.with_zeros_ones = torch.tensor(
        [[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32
      ).expand(self.batch_size,-1,-1).to(self.device)
    self.results_zeros = torch.zeros((self.batch_size, 24, 1), dtype=torch.float32).to(self.device)
    self.J_ones = torch.ones((self.batch_size, 24, 1), dtype=torch.float32, device=self.device)
    self.rest_shape_h_ones_full = torch.ones((self.batch_size, self.weights_all.shape[1], 1), dtype=torch.float32).to(self.device)

    if orig_smpl:
        self.to_keep = self.to_keep[self.to_keep < 6890]
        self.J_regressor = self.J_regressor[:,:,:6890]
        self.v_template = self.v_template[:,:6890,:]
        self.shapedirs = self.shapedirs[:,:6890,:,:]
        self.posedirs = self.posedirs[:,:6890,:,:]
        self.weights_all = self.weights_all[:,:6890,:]
        self.weights = self.weights_all[:,self.to_keep,:]
        self.shape = (1, 6890, 3)

    self.rest_shape_h_ones = torch.ones((self.batch_size, self.to_keep.shape[0], 1), dtype=torch.float32).to(self.device)

  @staticmethod
  def rodrigues(r):
    """
    Rodrigues' rotation formula that turns axis-angle tensor into rotation
    matrix in a batch-ed manner.
    Parameter:
    ----------
    r: Axis-angle rotation tensor of shape [batch_size * angle_num, 1, 3].
    Return:
    -------
    Rotation matrix of shape [batch_size * angle_num, 3, 3].
    """
    # eps = r.clone().normal_(std=1e-8)
    # theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)  # dim cannot be tuple
    theta = torch.norm(r, dim=(1, 2), keepdim=True)
    theta = torch.maximum(theta, torch.ones_like(theta) * np.finfo(np.float32).eps)

    theta_dim = theta.shape[0]
    r_hat = r / theta
    # cos = torch.cos(theta)
    z_stick = torch.zeros(theta_dim, dtype=torch.float32).to(r.device)
    m = torch.stack(
      (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
       -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1)
    m = m.view((-1, 3, 3))
    i_cube = (torch.eye(3, dtype=torch.float32).unsqueeze(dim=0) \
             + torch.zeros((theta_dim, 3, 3), dtype=torch.float32)).to(r.device)
    # A = r_hat.permute(0, 2, 1)
    # dot = torch.matmul(A, r_hat)
    # R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
    R = i_cube + m * torch.sin(theta) + torch.matmul(m, m) * (1 - torch.cos(theta))
    return R

  @staticmethod
  def with_zeros(x):
    """
    Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.
    Parameter:
    ---------
    x: Tensor to be appended.
    Return:
    ------
    Tensor after appending of shape [4,4]
    """
    ones = torch.tensor(
        [[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32
      ).expand(x.shape[0],-1,-1).to(x.device)
    ret = torch.cat((x, ones), dim=1)
    return ret

  @staticmethod
  def pack(x):
    """
    Append zero tensors of shape [4, 3] to a batch of [4, 1] shape tensor.
    Parameter:
    ----------
    x: A tensor of shape [batch_size, 4, 1]
    Return:
    ------
    A tensor of shape [batch_size, 4, 4] after appending.
    """
    zeros43 = torch.zeros(
      (x.shape[0], x.shape[1], 4, 3), dtype=torch.float32).to(x.device)
    ret = torch.cat((zeros43, x), dim=3)
    return ret


  def forward(self, pose, betas, verts: torch.Tensor = None, remove: bool = True):
    
    """
          Construct a compute graph that takes in parameters and outputs a tensor as
          model vertices. Face indices are also returned as a numpy ndarray.
          
          20190128: Add batch support.
          Parameters:
          ---------
          pose: Also known as 'theta', an [N, 24, 3] tensor indicating child joint rotation
          relative to parent joint. For root joint it's global orientation.
          Represented in a axis-angle format.
          betas: Parameter for model shape. A tensor of shape [N, 10] as coefficients of
          PCA components. Only 10 components were released by SMPL author.
          trans: Global translation tensor of shape [N, 3].
          Return:
          ------
          A 3-D tensor of [N * 6890 * 3] for vertices,
          and the corresponding [N * 19 * 3] joint positions.
    """
    batch_num = betas.shape[0]

    # v_shaped = torch.tensordot(betas, self.shapedirs, dims=([1], [2])) + self.v_template
    v_shaped = []
    v_shaped.append(torch.tensordot(betas[:,:10], self.shapedirs[0], dims=[[1], [2]]) + self.v_template[0].view(self.shape)) # female
    v_shaped.append(torch.tensordot(betas[:,:10], self.shapedirs[1], dims=[[1], [2]]) + self.v_template[1].view(self.shape)) # male
    v_shaped = torch.stack(v_shaped, axis=-1)
    
    # J = torch.matmul(self.J_regressor, v_shaped)
    if verts is None:
      J = torch.stack([torch.matmul(self.J_regressor[torch.greater(g, 0.5).to(torch.long)], v_shaped[i, :, :, torch.greater(g, 0.5).to(torch.long)]) for i, g in enumerate(betas[:,-1])])
    else:
      J = torch.stack([torch.matmul(self.J_regressor[torch.greater(g, 0.5).to(torch.long)], v_shaped[i, :, :, torch.greater(g, 0.5).to(torch.long)]) for i, g in enumerate(betas[:,-1])])
      J0 = J[:,0,:].view((-1, 1, 3))
      J = J - J0
    R_cube_big = self.rodrigues(pose.reshape(-1, 1, 3)).reshape(batch_num, -1, 3, 3)

    if verts is not None:
      v_posed = verts
    else:
      R_cube = R_cube_big[:, 1:, :, :]
      I_cube = (torch.eye(3, dtype=torch.float32).unsqueeze(dim=0) + \
        torch.zeros((batch_num, R_cube.shape[1], 3, 3), dtype=torch.float32)).to(self.device)
      lrotmin = (R_cube - I_cube).view(batch_num, -1, 1).squeeze(dim=2)
      v_posed = torch.stack([v_shaped[i, :, :, torch.greater(g, 0.5).to(torch.long)] + torch.tensordot(lrotmin[i], self.posedirs[torch.greater(g, 0.5).to(torch.long)], dims=([0], [2])) for i, g in enumerate(betas[:,-1])])

    results = []
    results.append(
      self.with_zeros(torch.cat((R_cube_big[:, 0], J[:, 0, :].view((-1, 3, 1))), dim=2))
    )
    for i in range(1, 24):
      results.append(
        torch.matmul(
          results[self.parent[i]],
          self.with_zeros(
            torch.cat(
              (R_cube_big[:, i], (J[:, i, :] - J[:, self.parent[i], :]).view((-1, 3, 1))),
              dim=2
            )
          )
        )
      )
    
    stacked = torch.stack(results, dim=1)
    results = stacked - \
      self.pack(
        torch.matmul(
          stacked,
            torch.cat((J, self.results_zeros), dim=2).view(
            (batch_num, 24, 4, 1)
          )
        )
      )
    
    joints = torch.matmul(results, torch.transpose(torch.concat([J, self.J_ones], axis=-1)[:,:,None,:],2,3))[:,:,:3,0]
            
    weights = self.weights if verts is not None else self.weights_all 
    rest_shape_h_ones_local = self.rest_shape_h_ones if verts is not None else self.rest_shape_h_ones_full
    if weights.shape[1] == 6890:
      rest_shape_h_ones_local = torch.ones((batch_num, 6890, 1), dtype=torch.float32).to(self.device)
    T = torch.stack([torch.tensordot(results[i], weights[torch.greater(g, 0.5).to(torch.long)], dims=([0], [1])).permute(2, 0, 1) for i, g in enumerate(betas[:,-1])]) 
    rest_shape_h = torch.cat(
      (v_posed, rest_shape_h_ones_local), dim=2
    )
    v = torch.matmul(T, rest_shape_h.view((batch_num, -1, 4, 1)))
    v = v.view((batch_num, -1, 4))[:, :, :3]
    result = v

    root_joint = joints[:,0,:].view((-1, 1, 3))
    result = result - root_joint
    joints = joints - root_joint

    if remove and verts is None:
      result = result[:,self.to_keep,:]

    return result, joints, v_shaped, v_posed

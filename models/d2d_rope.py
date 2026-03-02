"""
This code was originally obtained from:
https://github.com/facebookresearch/deit
and
https://github.com/meta-llama/codellama/blob/main/llama/model.py
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from functools import partial
import math
import torch.nn.functional as F
from torch.amp import autocast
def apply_vanilla_emb(self,x: torch.Tensor, coords: torch.Tensor, theta: float = 1000.0,use_random_project=True):
    _,d = x.shape
    assert d//4
    coords_x = coords[..., 0].unsqueeze(-1)
    coords_y = coords[..., 1].unsqueeze(-1)
    two_i = torch.arange(0,d,2,dtype=torch.float)
    div_term_x = torch.exp(-(two_i[:d/2:2]/d)*math.log(10000.0)).unsqueeze(0)
    div_term_y = torch.exp(-(two_i[d/2::2]/d)*math.log(10000.0)).unsqueeze(0)

    pe = torch.zeros_like(x,dtype=torch.float)
    pe[...,:d/2:2] = torch.sin(coords_x*div_term_x)
    pe[...,1:d/2:2] = torch.cos(coords_x*div_term_x)
    pe[...,d/2::2] = torch.sin(coords_y*div_term_y)
    pe[...,d/2+1::2] = torch.cos(coords_y*div_term_y)
    return x + pe

def apply_vanilla_emb(self,x: torch.Tensor, coords: torch.Tensor, theta: float = 1000.0,use_random_project=True):
    _,d = x.shape
    assert d//4
    coords_x = coords[..., 0].unsqueeze(-1)
    coords_y = coords[..., 1].unsqueeze(-1)
    two_i = torch.arange(0,d,2,dtype=torch.float)
    div_term_x = torch.exp(-(two_i[:d/2:2]/d)*math.log(10000.0)).unsqueeze(0)
    div_term_y = torch.exp(-(two_i[d/2::2]/d)*math.log(10000.0)).unsqueeze(0)

    pe = torch.zeros_like(x,dtype=torch.float)
    pe[...,:d/2:2] = torch.sin(coords_x*div_term_x)
    pe[...,1:d/2:2] = torch.cos(coords_x*div_term_x)
    pe[...,d/2::2] = torch.sin(coords_y*div_term_y)
    pe[...,d/2+1::2] = torch.cos(coords_y*div_term_y)
    return x + pe
    
def apply_learnable_emb(self,x: torch.Tensor, coords: torch.Tensor, theta: float = 1000.0,use_random_project=True):
    _,d = x.shape
    if self.pos_embedding is None:
        self.pos_embedding = nn.Parameter(torch.randn(1024,d).to(x.device))
    coords_copy = coords.clone().long()
    pe_x = self.pos_embedding[coords_copy[...,0]][:,:d/2]
    pe_y = self.pos_embedding[coords_copy[...,1]][:,d/2:]
    pe = torch.cat([pe_x, pe_y], dim=-1)
    return x + pe

def compute_axial_cis(dim: int, t_x: int, t_y: int, theta: float = 100.0):
    ##传递到同一个deivce
    device = t_x.device
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim)).reshape(1,-1).to(device)
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim)).reshape(1,-1).to(device)

    freqs_x = torch.einsum('ijlk,km->ijlm',t_x, freqs_x)
    freqs_y = torch.einsum('ijlk,km->ijlm',t_y, freqs_y)
    with autocast('cuda',enabled=False):
        freqs_cis_x = torch.polar(torch.ones_like(freqs_x).float(), freqs_x.float())
        freqs_cis_y = torch.polar(torch.ones_like(freqs_y).float(), freqs_y.float())
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
        
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, coords: torch.Tensor, theta: float = 1000.0,use_random_project=True):
    
    if use_random_project:
        coords = random_project(coords)
    t_x = coords[..., 0].reshape(*coords[..., 0].shape[:], 1)
    t_y = coords[..., 1].reshape(*coords[..., 0].shape[:], 1)
    dim = xq.shape[-1]
    freqs_cis = compute_axial_cis(dim, t_x, t_y, theta=theta)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)

def rotary_(coords:torch.Tensor, rotation_range=(-torch.pi,torch.pi),mode='2'):
    if mode == '1':
        theta  = torch.FloatTensor(1).uniform_(rotation_range[0],rotation_range[1])
        center= torch.mean(coords, dim=2,keepdim=True)
        rotation_matrix = torch.tensor([
            [torch.cos(theta), -torch.sin(theta)],
            [torch.sin(theta), torch.cos(theta)]
        ])
        center= torch.mean(coords, dim=2,keepdim=True)
        rotation_matrix = torch.tensor([
            [torch.cos(theta), -torch.sin(theta)],
            [torch.sin(theta), torch.cos(theta)]
        ],dtype=coords.dtype).to(coords.device)
        
        rotated_coords = torch.einsum('ijlk,km->ijlm',coords - center, rotation_matrix.t()) + center
        return rotated_coords
    if mode == '2':
        values = [0, torch.pi / 2, torch.pi, 3 * torch.pi / 2]
        theta = torch.tensor(values[torch.randint(0, len(values), (1,)).item()])
        center= torch.mean(coords, dim=2,keepdim=True)
        rotation_matrix = torch.tensor([
            [torch.cos(theta), -torch.sin(theta)],
            [torch.sin(theta), torch.cos(theta)]
        ])
        center= torch.mean(coords/100, dim=2,keepdim=True)*100
        rotation_matrix = torch.tensor([
            [torch.cos(theta), -torch.sin(theta)],
            [torch.sin(theta), torch.cos(theta)]
        ],dtype=coords.dtype).to(coords.device)
        
        rotated_coords = torch.einsum('ijlk,km->ijlm',coords - center, rotation_matrix.t()) + center
        return rotated_coords

def project_(coords:torch.Tensor,mode='2'):
    if mode == '1':
        coords_min = torch.min(coords,dim=-2)[0]
        coords_max = torch.max(coords,dim=-2)[0]
        coords_min=coords_min.reshape(*coords_min.shape[:-1],1,-1)
        coords_max=coords_max.reshape(*coords_max.shape[:-1],1,-1)
        perturbation = torch.min((coords_max - coords_min) * 0.01).item()
        random_perturbation = torch.empty_like(coords).uniform_(-perturbation, perturbation)
        coords = coords + random_perturbation
        projected_coords = (coords - coords_min) / (coords_max - coords_min) * 1024
        return projected_coords

    if mode == '2':
        coords = coords - torch.min(coords,dim=-2,keepdim=True)[0]
        coords = coords.type(torch.int64)
        num  = torch.max(coords).item()+1
        projected = torch.randint(0,10240,(*coords.shape[:-2],num,coords.shape[-1])).type(torch.float16).to(coords.device)
        projected_sorted = torch.sort(projected, dim=-2)[0].to(coords.device)
        if torch.any(coords >= num):
            raise ValueError("coords exceed the range of projected_coords")
        projected_coords = torch.gather(projected_sorted, dim=-2, index=coords)

        return projected_coords

def random_project(coords:torch.Tensor, rotation_range=(-torch.pi,torch.pi), p=0.8):
    
    apply_random_project = torch.rand(1) < p

    if apply_random_project:
        rotated_coords = rotary_(coords, rotation_range=rotation_range)
        projected_coords = project_(rotated_coords)
  
    else:
        projected_coords = coords
    
    return projected_coords



            


    
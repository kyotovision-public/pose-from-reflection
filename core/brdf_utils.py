import numpy as np
import torch
import torch.nn.functional as F
import struct

from .rmap_utils import create_normal_grid

def LoadMERL(path):
    BRDFSamplingResThetaH = 90
    BRDFSamplingResThetaD = 90
    BRDFSamplingResPhiD = 360

    RedScale = 1.0 / 1500.0
    GreenScale = 1.15 / 1500.0
    BlueScale = 1.66 / 1500.0 

    with open(path, 'rb') as f:
        bdata = f.read()
    dims = []
    dims.append(int.from_bytes(bdata[:4], 'little'))
    dims.append(int.from_bytes(bdata[4:8], 'little'))
    dims.append(int.from_bytes(bdata[8:12], 'little'))
    n = np.prod(dims)
    if n != (BRDFSamplingResThetaH * BRDFSamplingResThetaD * BRDFSamplingResPhiD / 2):
        raise Exception('Dimensions don\'t match')
    brdf = struct.unpack(str(3*n)+'d', bdata[12:])

    brdf = np.reshape(np.array(brdf), (3,BRDFSamplingResThetaH,BRDFSamplingResThetaD,BRDFSamplingResPhiD//2))
    brdf = torch.from_numpy(brdf.astype(np.float32))
    brdf = torch.clamp(brdf, 0, None)
    return brdf * torch.tensor([RedScale, GreenScale, BlueScale], dtype=brdf.dtype)[:,None,None,None]

def dot_vec(v1, v2):
    return torch.sum(v1 * v2, dim=-1, keepdim=True)

def normalize_vec(v):
    return v / torch.sqrt(torch.sum(v**2, dim=-1, keepdim=True) + 1e-20)    

def vec2halfdiff(v_in, v_out, n):
    h = normalize_vec(v_in + v_out)

    hDn = dot_vec(h, n)
    hDl = dot_vec(h, v_in)

    theta_h = torch.acos(torch.clamp(hDn, -1, 1))
    theta_d = torch.acos(torch.clamp(hDl, -1, 1))
    a = normalize_vec(-n + hDn * h)
    b = normalize_vec(torch.cross(h,a))

    x = dot_vec(a, v_in)
    y = dot_vec(b, v_in)
    phi_d = torch.atan2(y,x)

    phi_d[dot_vec(a,a) <= 0.001**2] = 0.0

    return theta_h, theta_d, phi_d

def halfdiff2grid(theta_h, theta_d, phi_d):
    BRDFSamplingResThetaH = 90
    BRDFSamplingResThetaD = 90
    BRDFSamplingResPhiD = 360
    
    theta_h = torch.clamp(theta_h, 0, None)
    thetaHalfDeg = ((theta_h / (np.pi / 2.0)) * BRDFSamplingResThetaH)
    temp = thetaHalfDeg * BRDFSamplingResThetaH
    idx1 = torch.clamp(torch.sqrt(temp), 0, BRDFSamplingResThetaH)
    
    tmp = theta_d / (np.pi * 0.5) * BRDFSamplingResThetaD
    idx2 = torch.clamp(tmp, 0, BRDFSamplingResThetaD)

    phi_d[phi_d < 0.0] += np.pi
    tmp = phi_d / np.pi * BRDFSamplingResPhiD / 2
    idx3 = torch.clamp(tmp, 0, BRDFSamplingResPhiD/2)

    idx1 = 2 * idx1 / BRDFSamplingResThetaH - 1
    idx2 = 2 * idx2 / BRDFSamplingResThetaD - 1
    idx3 = 2 * idx3 / (BRDFSamplingResPhiD / 2) - 1

    return torch.cat([idx3, idx2, idx1], dim=-1)

def compute_brdf_reflectance(brdf):
    device = brdf.device
    # compute reflectance at theta_in=0
    v = create_normal_grid(1024, 'sphere')
    l = torch.zeros_like(v)
    l[...,2] = 1.0

    grid = halfdiff2grid(*vec2halfdiff(l,v,l))

    #brdf = LoadMERL('/home/kyamashita/Projects/test-nvidia-optix/SDK/data/lambertian.binary')
    brdf_map = F.grid_sample(
        brdf[None,:,:,:,:],
        grid[None,None,:,:,:].to(device),
        mode='bilinear',
        padding_mode='border',
        align_corners=False
    )[:,:,0,:,:] * (dot_vec(l, v)[...,0] > 0).float().to(device)

    # integration
    reflectance = torch.sum(brdf_map, dim=(2,3)) / (0.25 * brdf_map.size(-2) * brdf_map.size(-1))
    return reflectance
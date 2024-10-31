import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import tinycudann as tcnn

import numba
from numba import cuda
import math

def create_normal_grid(out_size=256, projection_mode = 'sphere'):
    if np.isscalar(out_size):
        out_size = (out_size, out_size)

    v,u = torch.meshgrid(torch.arange(out_size[1]), torch.arange(out_size[0]))
    if projection_mode == 'sphere':
        x = 2 * ((u + 0.5) / out_size[0] - 0.5)
        y = -2 * ((v + 0.5) / out_size[1] - 0.5)
        z = torch.sqrt(torch.clamp(1 - x**2 - y**2, 0, 1))

        m = (z > 0.0).float()
        x = x * m
        y = y * m
        z = z * m

    elif projection_mode == 'stereographic':
        p = 4 * ((u + 0.5) / out_size[0] - 0.5)
        q = -4 * ((v + 0.5) / out_size[1] - 0.5)

        z = -(-1 + p**2 + q**2) / (1 + p**2 + q**2)
        m = (z > 0.0).float()

        x = (2*p) / (1 + p**2 + q**2) * m
        y = (2*q) / (1 + p**2 + q**2) * m
        z = z * m
    elif projection_mode == 'panorama':
        phi = (u + 0.5) / out_size[0] * 2 * np.pi
        theta = (v + 0.5) / out_size[1] * np.pi

        x = torch.cos(phi) * torch.sin(theta)
        y = torch.cos(theta)
        z = torch.sin(phi) * torch.sin(theta)
    elif projection_mode == 'probe':
        s = 2 * ((u + 0.5) / out_size[0] - 0.5)
        t = -2 * ((v + 0.5) / out_size[1] - 0.5)
        r = torch.sqrt(s**2 + t**2)
        phi = np.pi * r
        m = (r <= 1.0).float()

        x = s / r * torch.sin(phi) * m
        y = t / r * torch.sin(phi) * m
        z = torch.cos(phi) * m

    return torch.stack([x,y,z], dim=-1)

def create_rmap_visibility_weight(out_size=256, projection_mode = 'sphere', reflection_map=False):
    if np.isscalar(out_size):
        out_size = (out_size, out_size)

    v,u = torch.meshgrid(torch.arange(out_size[1]), torch.arange(out_size[0]))
    if projection_mode == 'sphere':
        x = 2 * ((u + 0.5) / out_size[0] - 0.5)
        y = -2 * ((v + 0.5) / out_size[1] - 0.5)
        z = torch.sqrt(torch.clamp(1 - x**2 - y**2, 0, 1))
        return 1 / (z + 1e-3)
    elif projection_mode == 'stereographic':
        p = 4 * ((u + 0.5) / out_size[0] - 0.5)
        q = -4 * ((v + 0.5) / out_size[1] - 0.5)

        z = -(-1 + p**2 + q**2) / (1 + p**2 + q**2)
        if reflection_map:
            x2y2 = 1 - z**2
            z = z + 1
            z = z / torch.sqrt(z**2 + x2y2)

        return 4 / (1 + p**2 + q**2)**2 * torch.clamp(z, 0, 1)

def img2rmap(image, normal_map, mask, out_size = 512, projection_mode = 'stereographic'):
    if image.ndim==4:
        result = []
        for i in range(len(image)):
            result.append(img2rmap(image[i], normal_map[i], mask[i], out_size, projection_mode))
        return torch.stack(result, dim=0)

    num_ch = image.size(0)
    is_valid = mask.view(-1) > 0
    pixels = image.reshape(num_ch,-1).transpose(0,1)[is_valid] # N*C
    normals = normal_map.reshape(3,-1).transpose(0,1)[is_valid] # N*3
    
    if projection_mode == 'sphere':
        u = 0.5 * (normals[:,0] + 1) * out_size 
        v = 0.5 * (-normals[:,1] + 1) * out_size
    elif projection_mode == 'stereographic':
        p = normals[:,0] / (1 + normals[:,2])
        q = normals[:,1] / (1 + normals[:,2])
        u = (0.5 + 0.25 * p) * out_size 
        v = (0.5 - 0.25 * q) * out_size
    elif projection_mode in ['probe', 'hemi_probe']:
        phi = torch.acos(torch.clamp(normals[:,2],-0.999999,0.999999))
        rn = torch.sqrt(normals[:,0]**2 + normals[:,1]**2 + 1e-6)
        r = phi / np.pi / rn
        u = normals[:,0] * r
        v = -normals[:,1] * r
        if projection_mode == 'hemi_probe':
            u = 2 * u
            v = 2 * v
        u = 0.5 * (u + 1) * out_size
        v = 0.5 * (v + 1) * out_size
    indices = (out_size * v.long() + u.long())[:,None]
    m = (u >= 0.0) * (u < out_size) * (v >= 0.0) * (v < out_size)
    pixels = pixels[m]
    normals = normals[m]
    indices = indices[m]

    ref_map_sum = torch.zeros((out_size*out_size, num_ch), device=image.device)
    ref_map_num = torch.zeros((out_size*out_size, num_ch), device=image.device)
    ref_map_sum = torch.scatter_add(ref_map_sum,0,indices.repeat(1,3),pixels)
    ref_map_num = torch.scatter_add(ref_map_num,0,indices.repeat(1,3),torch.ones_like(pixels, device=pixels.device))
    ref_map = (ref_map_sum / torch.clamp(ref_map_num, 1e-6, None)).transpose(0,1).view(num_ch, out_size, out_size)

    return ref_map

def compute_mean_rmap_color(rmap, mask=None, projection_mode = 'stereographic'):
    assert projection_mode == 'stereographic'
    rmap_height, rmap_width = rmap.size()[2:4]
    v,u = torch.meshgrid(torch.arange(rmap_height), torch.arange(rmap_width))
    p = 4 * ((u.to(rmap.device) + 0.5) / rmap_width - 0.5)
    q = -4 * ((v.to(rmap.device) + 0.5) / rmap_width - 0.5)
    z = -(-1 + p**2 + q**2) / (1 + p**2 + q**2)
    dA = 4 / (1 + p**2 + q**2)**2
    w = (dA * (z > 0.0).float())[None,None,:,:]
    if not (mask is None):
        w = w * mask
    mean_color = torch.sum(rmap * w, dim=(2,3)) / torch.sum(w, dim=(2,3))
    return mean_color

# prob_volume: [BS,Hn,Wn,H,W]
# ref_rot: [BS,3,3]
# src_rot: [BS,3,3]
def rotate_rmap(rmap, ref_rot, src_rot, projection_mode='probe', hemispherical=False):
    BS,C,Hn,Wn = rmap.size()
    device = rmap.device
    grid_nx, grid_ny, grid_nz = create_normal_grid((Wn, Hn), projection_mode).to(device).unbind(-1)

    normal = torch.stack([grid_nx, -grid_ny, -grid_nz], dim=2) # Hn,Wn,3    
    mask = (torch.sum(normal**2, dim=-1) > 0.25).float()[None,None,:,:]

    # compute transformation
    if (BS == 2) and (ref_rot.is_cuda == True):
        # Avoid a bug of torch.inverse() in pytorch1.7.0+cuda11.0
        # https://github.com/pytorch/pytorch/issues/47272#issuecomment-722278640
        # This has alredy been fixed so that you can remove this when using Pytorch1.7.1>=
        inv_ref_rot = torch.stack([torch.inverse(m) for m in ref_rot], dim=0)
        rot = torch.matmul(src_rot, inv_ref_rot)
    else:
        rot = torch.matmul(src_rot, torch.inverse(ref_rot))
    normal = torch.matmul(rot[:,None,None,:,:], normal[None,:,:,:,None])[:,:,:,:,0]
    grid_nx1 = normal[:,:,:,0]
    grid_ny1 = -normal[:,:,:,1]
    grid_nz1 = -normal[:,:,:,2]
    normal_ = torch.stack([grid_nx1, grid_ny1, grid_nz1], dim=1)

    # resampling
    rotated_rmap = sample_rmap(rmap, normal_, projection_mode = projection_mode, interep_mode='bilinear')

    rotated_rmap = rotated_rmap * mask

    if hemispherical:
        rotated_rmap = rotated_rmap * (grid_nz1 > 0.0).float()[:,None,:,:]

    return rotated_rmap

def sample_rmap(rmap, normal_map, projection_mode = 'stereographic', interep_mode='bicubic'):
    if projection_mode == 'stereographic':
        p = normal_map[:,0] / (1 + normal_map[:,2])
        q = normal_map[:,1] / (1 + normal_map[:,2])
        u = 2 * (0.5 + 0.25 * p) - 1 
        v = 2 * (0.5 - 0.25 * q) - 1
    elif projection_mode == 'sphere':
        r = torch.sqrt(torch.sum(normal_map**2, dim=1) + 1e-4)
        u = normal_map[:,0] / r
        v = -normal_map[:,1] / r
    elif projection_mode in ['probe', 'hemi_probe']:
        phi = torch.acos(torch.clamp(normal_map[:,2],-0.999999,0.999999))
        rn = torch.sqrt(normal_map[:,0]**2 + normal_map[:,1]**2 + 1e-6)
        r = phi / np.pi / rn
        u = normal_map[:,0] * r
        v = -normal_map[:,1] * r
        if projection_mode == 'hemi_probe':
            u = 2 * u
            v = 2 * v

    grid = torch.stack([u,v], dim=-1)

    img = F.grid_sample(rmap, grid, mode=interep_mode, padding_mode='border', align_corners=False)
    mask = (torch.sum(normal_map**2, dim=1, keepdim=True) > (0.25**2)).float()
    return img * mask

def sample_illum_map(illum_map, normal_maps, rot_matrices, interp_mode='bicubic'):
    normals_ = normal_maps * torch.tensor([1,-1,-1],device=normal_maps.device)[None,None,:,None,None]
    normals_ = torch.sum(normals_[:,:,:,None,:,:] * rot_matrices[:,:,:,:,None,None], dim=2)

    normals_ = normals_ / torch.sqrt(torch.sum(normals_**2, dim=2, keepdim=True) + 1e-9)

    theta = torch.acos(torch.clamp(normals_[:,:,1], -1, 1)) # BS*N*H*W
    phi = torch.atan2(normals_[:,:,2], normals_[:,:,0])

    phi = phi + (phi < 0.0).float() * 2 * np.pi

    Hi,Wi = illum_map.size()[-2:]
    N = normal_maps.size(1)
    u = phi / (2 * np.pi) * Wi
    v = theta / (np.pi) * Hi

    u = 2 * (u / Wi) - 1
    v = 2 * (v / Hi) - 1

    grids = torch.stack([u,v], dim=-1)
    imgs = []
    for idx_view in range(N):
        img = F.grid_sample(
            illum_map, 
            grids[:,idx_view], 
            mode=interp_mode, 
            padding_mode='border', 
            align_corners=False
        )
        imgs.append(img)

    masks = (torch.sum(normal_maps**2, dim=2, keepdim=True) > (0.25**2)).float()
    return torch.clamp(torch.stack(imgs, dim=1) * masks, 0, None)


@cuda.jit
def spherical_soft_scatter_sum_forward_kernel(
    # inputs
    in_value, in_normal, out_normal, kappa, 
    # options
    hemispherical,
    # result buffer
    out_value,
):
    num_in, num_ch = in_value.shape
    #num_out = out_value.shape

    idx_out = cuda.blockIdx.x
    idx_thread = cuda.threadIdx.x
    blockdim = cuda.blockDim.x

    out_value_tmp = cuda.shared.array(shape=(32,), dtype=numba.float32)
    if idx_thread == 0.0:
        for idx_ch in range(num_ch):
            out_value_tmp[idx_ch] = 0.0
    cuda.syncthreads()

    n_out = out_normal[idx_out]
    if (not hemispherical) or (n_out[2] > 0.0):
        coef = kappa / (2 * np.pi * (1 - math.exp(-2 * kappa)))
        idx_in = idx_thread
        while idx_in < num_in:
            n_in = in_normal[idx_in]
            v_in = in_value[idx_in]
            dp = n_in[0] * n_out[0] + n_in[1] * n_out[1] + n_in[2] * n_out[2]

            f = coef * math.exp(- kappa * (1 - dp))
            for idx_ch in range(num_ch):
                cuda.atomic.add(out_value_tmp, (idx_ch), v_in[idx_ch] * f)
                #out_value_tmp[idx_ch] += v_in[idx_ch] * f

            idx_in += blockdim

    cuda.syncthreads()
    if idx_thread == 0:
        for idx_ch in range(num_ch):
            out_value[idx_out, idx_ch] = out_value_tmp[idx_ch]

@cuda.jit
def spherical_soft_scatter_sum_backward_kernel(
    # inputs
    in_value, in_normal, out_normal, kappa, grad_output,
    # options
    hemispherical,
    # result buffer
    grad_in_value, grad_in_normal
):
    num_in, num_ch = in_value.shape
    num_out = grad_output.shape[0]

    idx_in = cuda.blockIdx.x
    idx_thread = cuda.threadIdx.x
    blockdim = cuda.blockDim.x

    grad_in_value_tmp = cuda.shared.array(shape=(32,), dtype=numba.float32)
    grad_in_normal_tmp = cuda.shared.array(shape=(3,), dtype=numba.float32)
    if idx_thread == 0.0:
        for idx_ch in range(num_ch):
            grad_in_value_tmp[idx_ch] = 0.0
        grad_in_normal_tmp[0] = 0.0
        grad_in_normal_tmp[1] = 0.0
        grad_in_normal_tmp[2] = 0.0

    cuda.syncthreads()

    v_in = in_value[idx_in]
    n_in = in_normal[idx_in]
    coef = kappa / (2 * np.pi * (1 - math.exp(-2 * kappa)))
    idx_out = idx_thread
    while idx_out < num_out:
        n_out = out_normal[idx_out]
        if (not hemispherical) or (n_out[2] > 0.0):
            grad_out = grad_output[idx_out]
            dp = n_in[0] * n_out[0] + n_in[1] * n_out[1] + n_in[2] * n_out[2]
            f = coef * math.exp(-kappa * (1 - dp))

            grad_nx = 0.0
            grad_ny = 0.0
            grad_nz = 0.0
            for idx_ch in range(num_ch):
                # gradient w.r.t. in_value
                cuda.atomic.add(grad_in_value_tmp, (idx_ch), f * grad_out[idx_ch])
                # gradient w.r.t in_normal
                grad_nx += v_in[idx_ch] * kappa * f * grad_out[idx_ch] * n_out[0]
                grad_ny += v_in[idx_ch] * kappa * f * grad_out[idx_ch] * n_out[1]
                grad_nz += v_in[idx_ch] * kappa * f * grad_out[idx_ch] * n_out[2]
            cuda.atomic.add(grad_in_normal_tmp, (0), grad_nx)
            cuda.atomic.add(grad_in_normal_tmp, (1), grad_ny)
            cuda.atomic.add(grad_in_normal_tmp, (2), grad_nz)

        idx_out += blockdim

    cuda.syncthreads()
    if idx_thread == 0:
        for idx_ch in range(num_ch):
            grad_in_value[idx_in, idx_ch] = grad_in_value_tmp[idx_ch]
        grad_in_normal[idx_in, 0] = grad_in_normal_tmp[0]
        grad_in_normal[idx_in, 1] = grad_in_normal_tmp[1]
        grad_in_normal[idx_in, 2] = grad_in_normal_tmp[2]

class spherical_soft_scatter_sum(torch.autograd.Function):

    # in_normal : Ni*3
    # in_value  : Ni*D
    # out_normal: No*3
    # kappa     : scalar
    # output    : No*D
    @staticmethod
    def forward(ctx, in_value, in_normal, out_normal, kappa, hemispherical):
        #dp = torch.sum(in_normal.detach()[None,:,:] * out_normal.detach()[:,None,:], dim=-1) # (No)x(Ni)
        #f = kappa / (2 * np.pi * (1 - np.exp(-2 * kappa))) * torch.exp(-kappa * (1 - dp)) # (No)x(Ni)
        #out_value = torch.sum(f[:,:,None] * in_value.detach()[None,:,:], dim=1)

        out_value = torch.zeros(
            (out_normal.size(0), in_value.size(1)), 
            dtype=in_value.dtype,
            device=in_value.device 
        )      
        spherical_soft_scatter_sum_forward_kernel[(out_value.size(0)), (1024,)](
            # inputs
            in_value.detach(), in_normal.detach(), out_normal.detach(), kappa.item(), 
            # options
            hemispherical.item(),
            # result buffer
            out_value,
        )

        ctx.save_for_backward(
            in_value, in_normal, out_normal, kappa, hemispherical
        )
        return out_value

    @staticmethod
    def backward(ctx, grad_output):
        in_value, in_normal, out_normal, kappa, hemispherical, = ctx.saved_tensors
        grad_in_value = None
        grad_in_normal = None
        grad_out_normal = None

        #if in_value.requires_grad:
        #    grad_in_value = torch.sum(f[:,:,None] * grad_output[:,None,:], dim=0)
        #if in_normal.requires_grad:
        #    grad_in_normal = torch.sum((kappa * in_value[None,:,:] * f * torch.sum(grad_output, dim=1)[:,None])[:,:,None] * out_normal[:,None,:], dim=0)
        if in_value.requires_grad or in_normal.requires_grad:
            dtype = in_value.dtype
            device = in_value.device
            grad_in_value = torch.zeros_like(in_value, dtype=dtype, device=device)
            grad_in_normal = torch.zeros_like(in_normal, dtype=dtype, device=device)

            spherical_soft_scatter_sum_backward_kernel[(in_value.size(0)), (1024,)](
                # inputs
                in_value.detach(), in_normal.detach(), out_normal.detach(), kappa.item(), 
                grad_output.detach(),
                # options
                hemispherical.item(),
                # result buffer
                grad_in_value, grad_in_normal,
            )

        return grad_in_value, grad_in_normal, grad_out_normal, None, None

class SoftImageToReflectanceMap(torch.nn.Module):
    def __init__(self):
        super(SoftImageToReflectanceMap, self).__init__()
        self.scatter_sum = spherical_soft_scatter_sum.apply

    def forward(self, 
        image, normal_map, mask, 
        kappa = 100.0, out_size = 256, projection_mode = 'stereographic',
        hemispherical=False,
        distortion_coef = None,
        query_normal_map=None
    ):
        if image.dim() == 4:
            rmap = []
            for idx_b in range(image.size(0)):
                rmap.append(self.forward(
                    image[idx_b], normal_map[idx_b], mask[idx_b],
                    kappa, out_size, projection_mode,
                    hemispherical,
                    distortion_coef[idx_b] if not (distortion_coef is None) else None,
                    query_normal_map[idx_b] if not (query_normal_map is None) else None
                ))
            return torch.stack(rmap, dim=0)

        num_ch = image.size(0)
        is_valid = mask.view(-1) > 0
        device = image.device
        in_pixels = image.reshape(num_ch,-1).transpose(0,1)[is_valid] # N*C
        in_normals = normal_map.reshape(3,-1).transpose(0,1)[is_valid] # N*3

        v, u = torch.meshgrid(torch.arange(out_size), torch.arange(out_size))
        if not (query_normal_map is None):
            out_normals = torch.stack(torch.unbind(query_normal_map, dim=0), dim=-1).view(-1,3)
            out_size = query_normal_map.size(-1)
        elif projection_mode == 'stereographic':
            p = 4 * ((u.to(device) + 0.5) / out_size - 0.5)
            q = -4 * ((v.to(device) + 0.5) / out_size - 0.5)
            x = 2 * p / (1 + p**2 + q**2)
            y = 2 * q / (1 + p**2 + q**2)
            z = -(-1 + p**2 + q**2) / (1 + p**2 + q**2)
            out_normals = torch.stack([x,y,z], dim=-1).view(-1,3)
        elif projection_mode == 'sphere':
            x = 2 * ((u.to(device) + 0.5) / out_size - 0.5)
            y = -2 * ((v.to(device) + 0.5) / out_size - 0.5)
            z = torch.sqrt(torch.clamp(1.0 - x**2 - y**2, 0, 1))
            out_normals = torch.stack([x,y,z], dim=-1).view(-1,3)
            hemispherical=True
        else:
            assert False

        if distortion_coef is None:
            distortion_coef = 1.0
        else:
            distortion_coef = distortion_coef.view(-1,1)[is_valid]

        kappa = torch.tensor(kappa, device=device)
        hemispherical = torch.tensor(hemispherical, device=device)
        num = self.scatter_sum(in_pixels**2 * distortion_coef, in_normals, out_normals, kappa, hemispherical)
        sum = self.scatter_sum(in_pixels * distortion_coef, in_normals, out_normals, kappa, hemispherical)
        out_pixels = num / (sum + 1e-3)
        rmap = out_pixels.transpose(0,1).view(num_ch, -1, out_size)
        return rmap

class ReflectanceMapShader(torch.nn.Module):
    def __init__(self, projection_mode='sphere'):
        super(ReflectanceMapShader, self).__init__()
        self.rmap = None
        self.projection_mode = projection_mode

    def set_rmap(self, rmap):
        v,u = torch.meshgrid(torch.arange(rmap.size(-1)), torch.arange(rmap.size(-2)))
        x = 2 * (u + 0.5) / rmap.size(-1) - 1
        y = 2 * (v + 0.5) / rmap.size(-2) - 1
        r = torch.sqrt(x**2 + y**2)
        if self.projection_mode == 'sphere':
            x = x * 0.92 / torch.clamp(r, 0.92, None)
            y = y * 0.92 / torch.clamp(r, 0.92, None)
        elif self.projection_mode == 'stereographic':
            x = x * 0.46 / torch.clamp(r, 0.46, None)
            y = y * 0.46 / torch.clamp(r, 0.46, None)
        grid = torch.stack([x,y], dim=-1)[None].repeat(rmap.size(0),1,1,1)
        self.rmap = F.grid_sample(
            rmap,
            grid.to(rmap.device),
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )
        self.rmap = rmap
        #from .sfs_utils import plot_hdr
        #import matplotlib.pyplot as plt
        #plot_hdr(self.rmap / torch.max(self.rmap))
        #plt.show()

    def generate_rmaps(self, extrinsics=None, x=None, out_size = 256, projection_mode='stereographic'):
        assert not (extrinsics is None)
        num_views = len(extrinsics)
        v = -extrinsics[:,2:3,:3].repeat(1,out_size**2,1)

        n = create_normal_grid(out_size, projection_mode=projection_mode).to(v.device)
        n = n * torch.tensor([1,-1,-1], dtype=n.dtype, device=n.device)
        n = (extrinsics[:,None,:3,:3].transpose(-1,-2) @ n.view(1,-1,3,1))[...,0]

        if x is None:
            x = torch.zeros_like(v, dtype=v.dtype, device=v.device)

        return self.forward(x, n, v, extrinsics).transpose(-1,-2).view(-1,3, out_size, out_size)

    # n: num_views*num_rays*3
    def forward(self, x, n, v, extrinsics):
        rmap_mask = torch.any(self.rmap > 0.0, dim=1, keepdim=True).float()

        # world to camera coord
        n = torch.sum(n[...,None,:] * extrinsics[:,None,:3,:3], dim=-1)
        n = n * torch.tensor([1,-1,-1], dtype=n.dtype, device=n.device)

        mask = (torch.sum(n**2, dim=-1, keepdim=True) >= 0.64).float()
        mask = mask * (n[...,2:3] > 0.0).float()#torch.sigmoid(1000*n[...,2:3]).float()
        #n = n * (n[...,2:3] > 0.01).float() + n.detach() * (n[...,2:3] <= 0.01).float()
        pixel_vals = sample_rmap(
            self.rmap, 
            torch.stack(n.unbind(-1), dim=1)[...,None], 
            projection_mode=self.projection_mode, 
            interep_mode='bilinear'
        )[:,:,:,0].transpose(-1,-2)
        return pixel_vals * mask# * edge_coef

class BlendedReflectanceMapShader(torch.nn.Module):
    def __init__(self, projection_mode='sphere'):
        super(BlendedReflectanceMapShader, self).__init__()
        self.rmap = None
        self.projection_mode = projection_mode

    def erosion(self, rmap_bases, nitr=1):
        if nitr > 1:
            rmap_bases = self.erosion(rmap_bases, nitr-1)

        mask = torch.any(rmap_bases > 0.0, dim=2, keepdim=True).float()
        sum = 0
        num = 0
        for shift, dim in zip([-1,1,-1,1], [-1,-1,-2,-2]):
            sum = sum + torch.roll(rmap_bases, shift, dim)
            num = num + torch.roll(mask, shift, dim)
        return rmap_bases + sum / (num + 1e-10) * (1 - mask)

    def set_rmap_bases(self, rmap_bases, blending_func):
        self.rmap_bases = self.erosion(rmap_bases, 5)
        #from .sfs_utils import plot_hdr
        #import matplotlib.pyplot as plt
        #plot_hdr(self.rmap_bases[:,0])
        #plt.show()
        self.blending_func = blending_func

    # n: num_views*num_rays*3
    def forward(self, x, n, v, extrinsics, idx_rmap=0):
        #rmap_mask = torch.any(self.rmap > 0.0, dim=1, keepdim=True).float()

        blending_weights = self.blending_func(x) # num_views*num_rmays*num_rmaps

        # world to camera coord
        n = torch.sum(n[...,None,:] * extrinsics[:,None,:3,:3], dim=-1)
        n = n * torch.tensor([1,-1,-1], dtype=n.dtype, device=n.device)

        mask = (torch.sum(n**2, dim=-1, keepdim=True) >= 0.64).float()
        mask = mask * (n[...,2:3] > 0.0).float()#torch.sigmoid(1000*n[...,2:3]).float()
        #n = n * (n[...,2:3] > 0.01).float() + n.detach() * (n[...,2:3] <= 0.01).float()
        if self.projection_mode == 'sphere':
            grid = torch.stack([n[...,0], -n[...,1]], dim=-1)
        elif self.projection_mode == 'stereographic':
            p = n[...,0] / (1 + n[...,2])
            q = n[...,1] / (1 + n[...,2])
            #r = torch.sqrt(p**2 + q**2 + 1e-20)
            #s = 0.95 / torch.clamp(r, 0.95, None)
            #p = p * s
            #q = q * s
            u = 2 * (0.5 + 0.25 * p) - 1 
            v = 2 * (0.5 - 0.25 * q) - 1
            grid = torch.stack([u,v], dim=-1)

        pixel_vals = []
        for idx_rmap in range(self.rmap_bases.size(1)):
            pixel_vals.append(torch.nn.functional.grid_sample(
                self.rmap_bases[:,idx_rmap],
                grid[:,:,None,:],
                mode='bilinear',
                padding_mode='border',
                align_corners=False
            )[:,:,:,0].transpose(-1,-2) * mask)
        pixel_vals = torch.stack(pixel_vals, dim=2)

        return torch.sum(pixel_vals * blending_weights[:,:,:,None], dim=2)

class NeuralRadianceFieldShader(torch.nn.Module):
    def __init__(self, homogeneous=False, ref_nerf=False):
        super(NeuralRadianceFieldShader, self).__init__()
        self.homogeneous = homogeneous
        self.ref_nerf = ref_nerf
        self.encoding = tcnn.Encoding(
            3, 
            {
                "otype": "Frequency",
                "n_frequencies": 4,
            }
        )
        self.radiance_field = tcnn.Network( # WithInputEncoding(
        	30, 3,
	        #{
            #    "otype": "Frequency", #"otype": "OneBlob", 
            #    "n_frequencies": 4,#"n_bins": 64
            #}, 
            {
                "otype": "CutlassMLP", 
                "activation": "ReLU", 
                "output_activation": "None", 
                "n_neurons": 512, 
                "n_hidden_layers": 4
            }
        )

    # n: num_views*num_rays*3
    def forward(self, x, n, v, extrinsics):
        num_views, num_rays = x.size(0), x.size(1)

        mask = (torch.sum(n**2, dim=-1) > 0.5)

        # camera coord to world coord
        #n = n * torch.tensor([1,-1,-1], dtype=n.dtype, device=n.device)
        #n = torch.sum(n[...,:,None] * extrinsics[:,None,:3,:3], dim=-2)

        if self.ref_nerf:
            r = -v + 2 * torch.sum(v * n, dim=-1, keepdim=True) * n
            v_freq = self.encoding(r.view(num_views*num_rays,3)).view(num_views,num_rays,-1)
        else:
            v_freq = self.encoding(v.view(num_views*num_rays,3)).view(num_views,num_rays,-1)

        if self.homogeneous:
            x = x * 0.0

        x_in = torch.cat([x,n,v_freq], dim=-1)
        pixel_vals = torch.zeros_like(n, device=n.device).view(num_views*num_rays,-1)
        x_in_ = x_in.view(num_views*num_rays,-1)[mask.view(-1)]
        pixel_vals_ = torch.exp(-2 + self.radiance_field(x_in_))
        pixel_vals[mask.view(-1)] = pixel_vals_.float()
        pixel_vals = pixel_vals.view(num_views, num_rays, 3)

        pixel_masks = torch.exp(-20 * torch.clamp(-torch.sum(v * n, dim=-1, keepdim=True), 0, None))

        return pixel_vals * pixel_masks

def compute_reflection_map(normal_map, intrinsic):
    # viewing direction
    v,u = torch.meshgrid(torch.arange(normal_map.size(-2)),torch.arange(normal_map.size(-1)))
    m = torch.stack([u,v,torch.ones_like(u)], dim=0).to(intrinsic.device) # [3*H*W]
    look = torch.sum(torch.inverse(intrinsic)[:,:,:,None,None] * m[None,None,:,:,:], dim=2)
    view_map = -look * torch.tensor([1,-1,-1], device=look.device)[None,:,None,None]
    view_map = view_map / torch.sqrt(torch.clamp(torch.sum(view_map**2, dim=1, keepdim=True), 1e-6, None))

    reflection_map = -view_map + 2 * torch.sum(normal_map * view_map, dim=1, keepdim=True) * normal_map
    return reflection_map

def rmap2refrmap(rmap, projection_mode="stereographic"):
    out_size = (rmap.size(-2), rmap.size(-1))
    xr,yr,zr = create_normal_grid(out_size, projection_mode).to(rmap.device).unbind(-1)

    mask = ((xr**2 + yr**2 + zr**2) > 0.0).float()[None,None,:,:]

    r = torch.clamp(torch.sqrt(xr**2 + yr**2 + (zr+1)**2), 1e-4, None)
    x = xr / r
    y = yr / r
    z = (zr + 1) / r

    normal_map = torch.stack([x,y,z], dim=0)[None].repeat(rmap.size(0),1,1,1)

    return sample_rmap(rmap, normal_map, projection_mode, interep_mode='bilinear') * mask

def refrmap2rmap(rmap, projection_mode="stereographic"):
    out_size = (rmap.size(-2), rmap.size(-1))
    x,y,z = create_normal_grid(out_size, projection_mode).to(rmap.device).unbind(-1)

    mask = ((x**2 + y**2 + z**2) > 0.0).float()[None,None,:,:]

    xr = 2 * x * z
    yr = 2 * y * z
    zr = 2 * z**2 - 1

    mask = mask * (z[None,None] > 0.0).float()

    normal_map = torch.stack([xr,yr,zr], dim=0)[None].repeat(rmap.size(0),1,1,1)

    return sample_rmap(rmap, normal_map, projection_mode, interep_mode='bilinear') * mask


# rmap: N*C*Hn*Wn
# extrinsics: N*4*4
def rmap2panorama(rmap, extrinsics, projection_mode="stereographic", out_size=(512,256)):
    rot = extrinsics[:,:3,:3] # N*3*3
    v,u = torch.meshgrid(torch.arange(out_size[1]), torch.arange(out_size[0]))
    phi = (u + 0.5) / out_size[0] * 2 * np.pi
    theta = (v + 0.5) / out_size[1] * np.pi

    x = torch.cos(phi) * torch.sin(theta)
    y = torch.cos(theta)
    z = torch.sin(phi) * torch.sin(theta)

    illum_map_weight = torch.sin(theta).to(rmap.device)

    vec = torch.stack([x,y,z], dim=-1).to(rmap.device)
    vec = (rot[:,None,None,:,:] @ vec[None,:,:,:,None])[...,0] # N*H*W*3

    vec = vec * torch.tensor([1,-1,-1], device=vec.device)

    p = vec[...,0] / (1 + vec[...,2])
    q = vec[...,1] / (1 + vec[...,2])
    r = torch.sqrt(p**2 + q**2)

    u = 2 * (0.5 + 0.25 * p) - 1 
    v = 2 * (0.5 - 0.25 * q) - 1
    grid = torch.stack([u,v], dim=-1)

    mask = (r < 0.95).float()[:,None,:,:]
    mask = mask * torch.clamp(vec[...,2], 0, 1)[:,None,:,:]

    return torch.nn.functional.grid_sample(
        rmap,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=False
    ), mask, illum_map_weight

def pad_illum_map(x, padding=1):
    return F.pad(F.pad(x,(padding,padding,0,0), 'circular'), (0,0,padding,padding), 'replicate')

# img: BS,C,H,W
# extrinsic: BS,4,4
def panorama2rmap(illum_map, extrinsic, projection_mode="stereographic", out_size=(128,128)):
    device = illum_map.device
    dtype = illum_map.dtype
    Wn,Hn = out_size

    grid_n = create_normal_grid(out_size=out_size[0], projection_mode = projection_mode) * torch.tensor([1,-1,-1])
    grid_n = grid_n.to(device)
    mask = (grid_n[...,2] < 0.0).float()
    
    grid_n = torch.matmul(extrinsic[:,None,None,:3,:3].transpose(3,4), grid_n[None,:,:,:,None])[:,:,:,:,0]
    grid_theta = torch.acos(torch.clamp(grid_n[:,:,:,1], -1, 1))
    grid_phi = torch.atan2(grid_n[:,:,:,2], grid_n[:,:,:,0])
    grid_phi[grid_phi < 0.0] += 2.0 * np.pi
    u = grid_phi/np.pi - 1.0
    v = 2 * grid_theta / np.pi - 1.0
    grid = torch.stack([u, v], dim=-1)
    out = F.grid_sample(illum_map, grid, mode='bilinear', padding_mode='border')
    return out * mask

class Im2RMMappingModule(nn.Module):
    def __init__(self, sharpness = 150.0, dropout_p=0.1):
        super(Im2RMMappingModule, self).__init__()
        self.log_query_norm = nn.Parameter(torch.randn((1,), requires_grad=True))
        with torch.no_grad():
            self.log_query_norm[:] = np.log(np.clip(sharpness,1e-3,None))
        self.dropout = nn.Dropout(p=dropout_p)

    # in_fea_map:    bs*c*h*w
    # in_normal_map: bs*3*h*w
    # in_mask_map:   bs*1*h*w
    def forward(
            self, 
            in_fea_map, in_normal_map, in_mask_map=None, 
            out_size=(64,64), projection_mode='stereographic',
            use_cosine_map=True,
            use_var_map=True,
            hemispherical=False,
        ):
        wo,ho = out_size
        bs,c,h,w = in_fea_map.size()
        device = in_fea_map.device
        query_normals = create_normal_grid(
            out_size, projection_mode
        ).contiguous().view(-1,3).to(device)[None].repeat(bs,1,1) # bs*n*3
        if hemispherical:
            rm_mask = (query_normals[...,2] > 0).float().view(bs,1,ho,wo)
        else:
            rm_mask = torch.any(query_normals != 0, dim=-1).float().view(bs,1,ho,wo)

        query_normals_ = query_normals[0][rm_mask[0].view(ho*wo) != 0].view(1,-1,3).repeat(bs,1,1)

        vals = in_fea_map.contiguous().view(bs,c,h*w).transpose(-1,-2)
        key_normals = in_normal_map.view(bs,3,h*w).transpose(1,2).contiguous().view(bs,h*w, 3)

        normal_cosines = query_normals_ @ key_normals.transpose(-1,-2)
        dps = normal_cosines / np.sqrt(3)
        dps_ = dps - torch.max(dps, dim=-1)[0][:,:,None]
        weights_ = torch.exp(self.log_query_norm.exp() * dps_)
        weights_ = self.dropout(weights_)
        if not (in_mask_map is None):
            weights_ = weights_ * in_mask_map.view(bs,1,h*w)

        weights = weights_ / (torch.sum(weights_, dim=-1, keepdim=True) + 1e-9)

        mapped_vals_ = (weights @ vals) # bs*M*35

        if True:
            mapped_vals_ = torch.cat([query_normals_, mapped_vals_], dim=-1)
        if use_var_map:
            mean = mapped_vals_[...,-3:]
            sq_mean = (weights @ vals[...,-3:]**2)
            var = torch.sum(sq_mean - mean**2, dim=-1, keepdim=True)
            mapped_vals_ = torch.cat([var, mapped_vals_], dim=-1)
        if use_cosine_map:
            rm_cosines = torch.sum(weights * normal_cosines, dim=-1, keepdim=True)
            mapped_vals_ = torch.cat([rm_cosines, mapped_vals_], dim=-1)

        mapped_vals = torch.zeros((bs,ho*wo,mapped_vals_.size(-1)), device=mapped_vals_.device)
        mapped_vals[rm_mask.view(bs,ho*wo) != 0] = mapped_vals_.view(bs*mapped_vals_.size(1),-1)

        fea_rmap = mapped_vals.transpose(-1,-2).view(bs,-1,ho,wo)

        fea_rmap = fea_rmap# * rm_mask

        return fea_rmap

# rvec: N*3
# ret:  N*3*3
def rvec2rotmat(rvec):
    theta = torch.sqrt(torch.sum(rvec**2, dim=-1, keepdim=True) + 1e-9) # N*1
    r = rvec / theta # N*3
    cosTheta = torch.cos(theta)[...,None]
    sinTheta = torch.sin(theta)[...,None]

    R = cosTheta * torch.eye(3, dtype=rvec.dtype, device=rvec.device)
    R = R + (1. - cosTheta) * r[...,:,None] @ r[...,None,:]

    rx, ry, rz = r.unbind(-1)
    zeros_ = torch.zeros_like(rx)
    R = R + sinTheta * torch.stack([
        torch.stack([zeros_, -rz, ry], dim=-1),
        torch.stack([rz, zeros_, -rx], dim=-1),
        torch.stack([-ry, rx, zeros_], dim=-1)
    ], dim=-2)
    return R

@cuda.jit
def gaussian_blur_rmap_forward_kernel(
    # inputs
    in_rmap, kappa, 
    # result buffer
    out_rmap, out_norm_coef
):
    BS,C,H,W = in_rmap.shape

    idx_batch = cuda.blockIdx.x
    v0 = cuda.blockIdx.y
    u0 = cuda.blockIdx.z
    idx_thread = cuda.threadIdx.x
    blockdim = cuda.blockDim.x

    s0 = 2 * ((u0 + 0.5) / W - 0.5)
    t0 = -2 * ((v0 + 0.5) / H - 0.5)
    r0 = math.sqrt(s0**2 + t0**2)


    if idx_thread == 0:
        for idx_ch in range(C):
            out_rmap[idx_batch,idx_ch,v0,u0] = 0.0
        out_norm_coef[idx_batch,v0,u0] = 0.0
    cuda.syncthreads()

    if r0 <= 1.0:
        phi0 = np.pi * r0
        x0 = s0 / r0 * math.sin(phi0)
        y0 = t0 / r0 * math.sin(phi0)
        z0 = math.cos(phi0)

        idx_src = idx_thread
        while idx_src < H * W:
            v = idx_src // W
            u = idx_src % W

            s = 2 * ((u + 0.5) / W - 0.5)
            t = -2 * ((v + 0.5) / H - 0.5)
            r = math.sqrt(s**2 + t**2)

            if r <= 1.0:
                phi = np.pi * r
                x = s / r * math.sin(phi)
                y = t / r * math.sin(phi)
                z = math.cos(phi)

                dp = x0 * x + y0 * y + z0 * z
                acos_z = math.acos(max(-0.99999,min(z,0.99999)))
                sinc_z = math.sin(abs(acos_z) + 1e-10) / (abs(acos_z) + 1e-10)
                solid_angle = (2 * np.pi)**2 / (H * W) * sinc_z
                w = math.exp(kappa * (dp - 1.))

                if w > 1e-3:
                    for idx_ch in range(C):
                        cuda.atomic.add(
                            out_rmap, 
                            (idx_batch, idx_ch, v0, u0), 
                            w * in_rmap[idx_batch,idx_ch,v,u] * solid_angle
                        )
                    cuda.atomic.add(
                        out_norm_coef, 
                        (idx_batch, v0, u0), 
                        w * solid_angle
                    )

            idx_src += blockdim

@cuda.jit
def gaussian_blur_rmap_backward_kernel(
    # inputs
    in_rmap, kappa, out_norm_coef, 
    grad_output,
    # result buffer
    grad_rmap 
):
    BS,C,H,W = in_rmap.shape

    idx_batch = cuda.blockIdx.x
    v0 = cuda.blockIdx.y
    u0 = cuda.blockIdx.z
    idx_thread = cuda.threadIdx.x
    blockdim = cuda.blockDim.x

    s0 = 2 * ((u0 + 0.5) / W - 0.5)
    t0 = -2 * ((v0 + 0.5) / H - 0.5)
    r0 = math.sqrt(s0**2 + t0**2)

    if idx_thread == 0:
        for idx_ch in range(C):
            grad_rmap[idx_batch,idx_ch,v0,u0] = 0.0
    cuda.syncthreads()

    if r0 <= 1.0:
        phi0 = np.pi * r0
        x0 = s0 / r0 * math.sin(phi0)
        y0 = t0 / r0 * math.sin(phi0)
        z0 = math.cos(phi0)

        idx_src = idx_thread
        while idx_src < H * W:
            v = idx_src // W
            u = idx_src % W

            s = 2 * ((u + 0.5) / W - 0.5)
            t = -2 * ((v + 0.5) / H - 0.5)
            r = math.sqrt(s**2 + t**2)

            if r <= 1.0:
                phi = np.pi * r
                x = s / r * math.sin(phi)
                y = t / r * math.sin(phi)
                z = math.cos(phi)

                dp = x0 * x + y0 * y + z0 * z
                acos_z = math.acos(max(-0.99999,min(z,0.99999)))
                sinc_z = math.sin(abs(acos_z) + 1e-10) / (abs(acos_z) + 1e-10)
                solid_angle = (2 * np.pi)**2 / (H * W) * sinc_z
                w = math.exp(kappa * (dp - 1.))

                if w > 1e-3:
                    for idx_ch in range(C):
                        cuda.atomic.add(
                            grad_rmap, 
                            (idx_batch, idx_ch, v, u), 
                            w * grad_output[idx_batch,idx_ch,v0,u0] / out_norm_coef[idx_batch,v0,u0] * solid_angle
                        )

            idx_src += blockdim

class gaussian_blur_rmap_func(torch.autograd.Function):

    # rmap   : BS*C*H*W
    # kappa  : scalar
    # output : BS*C*H*W
    @staticmethod
    def forward(ctx, rmap, kappa):
        #dp = torch.sum(in_normal.detach()[None,:,:] * out_normal.detach()[:,None,:], dim=-1) # (No)x(Ni)
        #f = kappa / (2 * np.pi * (1 - np.exp(-2 * kappa))) * torch.exp(-kappa * (1 - dp)) # (No)x(Ni)
        #out_value = torch.sum(f[:,:,None] * in_value.detach()[None,:,:], dim=1)

        BS,C,H,W = rmap.size()
        out_rmap_ = torch.zeros_like(rmap)
        out_norm_coef = torch.zeros_like(rmap[:,0])

        gaussian_blur_rmap_forward_kernel[(BS,H,W), (256,)](
            # inputs
            rmap.detach(), kappa.item(),
            # result buffer
            out_rmap_, out_norm_coef
        )
        out_rmap = out_rmap_ / (out_norm_coef + 1e-10)

        #import matplotlib.pyplot as plt
        #plt.imshow(out_norm_coef[0].detach().cpu())
        #plt.show()

        #print(out_norm_coef[:,H//2,W//2] / (2*np.pi/kappa*(1-torch.exp(-2*kappa))))

        ctx.save_for_backward(
            rmap, kappa, out_norm_coef
        )
        return out_rmap

    @staticmethod
    def backward(ctx, grad_output):
        rmap, kappa, out_norm_coef, = ctx.saved_tensors
        grad_rmap = None
        grad_kappa = None

        #if in_value.requires_grad:
        #    grad_in_value = torch.sum(f[:,:,None] * grad_output[:,None,:], dim=0)
        #if in_normal.requires_grad:
        #    grad_in_normal = torch.sum((kappa * in_value[None,:,:] * f * torch.sum(grad_output, dim=1)[:,None])[:,:,None] * out_normal[:,None,:], dim=0)
        if rmap.requires_grad:
            BS,C,H,W = rmap.size()
            grad_rmap = torch.zeros_like(rmap)

            gaussian_blur_rmap_backward_kernel[(BS,H,W), (256,)](
                # inputs
                rmap.detach(), kappa.item(), out_norm_coef.detach(), 
                grad_output.detach(),
                # result buffer
                grad_rmap
            )

        return grad_rmap, grad_kappa


def gaussian_blur_rmap(rmap, kappa=10):
    if False:
        BS,C,H,W = rmap.size()

        normal_grid = create_normal_grid(rmap.size(-1),'probe').to(rmap.device)
        rmap_mask = (torch.sum(normal_grid**2,dim=-1) > 0.5).float()
        normal_cosines = torch.sum(normal_grid[:,:,None,None,:] * normal_grid[None,None,:,:,:], dim=-1)
        w = torch.exp(kappa * (normal_cosines - 1)) * rmap_mask[:,:,None,None] * rmap_mask[None,None,:,:]
        solid_angle = (2 * np.pi)**2 / (H * W) * torch.sinc(torch.acos(torch.clamp(normal_grid[:,:,2], -0.99999, 0.99999)) / np.pi)

        return torch.sum(rmap[:,:,None,None,:,:] * w[None,None,:,:,:,:] * solid_angle, dim=(-1,-2)) / (torch.sum(w[None,None,:,:,:,:] * solid_angle, dim=(-1,-2)) + 1e-10)

    return gaussian_blur_rmap_func.apply(rmap, torch.tensor([kappa], device=rmap.device))
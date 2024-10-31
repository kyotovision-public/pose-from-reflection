import torch
import torch.nn as nn
import torch.nn.functional as F

import fastsweep
import drjit

import numba
from numba import cuda
import math

import numpy as np
from scipy.spatial.transform import Rotation as scipy_rotation
import skimage
import trimesh
#import open3d as o3d
from tqdm import tqdm
import pymesh

from.device_funcs import *
from .rmap_utils import SoftImageToReflectanceMap, create_rmap_visibility_weight, img2rmap
from .nie_utils import remove_duplicates

basis_scale = 1
basis_radius = 4 * basis_scale + 1

# device functions
@cuda.jit(device=True)
def clip_device(x, min_, max_):
    return min(max(min_, x), max_)

# device functions
@cuda.jit(device=True)
def sgn_device(x):
    return 1.0 if x >= 0 else -1.0

@cuda.jit(device=True)
def compute_weight_device(t):
    #t = 2.5 - clip_device(abs(t), 0, 2.5)
    #if t < 1.0:
    #    return 1 / 24.0 * t**4
    #elif t < 2.0:
    #    u = t - 1.0
    #    return 1 / 24.0 * (-4 * u**4 + 4 * u**3 + 6 * u**2 + 4 * u + 1)
    #else:
    #    u = t - 2.0
    #    return 1 / 24.0 * (6 * u**4 - 12 * u**3 - 6 * u**2 + 12 * u + 11)

    t = 2 - clip_device(abs(t), 0, 2)
    if t < 1.0:
        return 1 / 6.0 * t**3
    else:
        u = t - 1.0
        return 1 / 6.0 * (-3 * u**3 + 3 * u**2 + 3 * u + 1)

@cuda.jit(device=True)
def compute_grad_weight_device(t):
    #sgn = -sgn_device(t)
    #t = 2.5 - clip_device(abs(t), 0, 2.5)
    #if t < 1.0:
    #    return sgn * 4 / 24.0 * t**3
    #elif t < 2.0:
    #    u = t - 1.0
    #    return sgn * 1 / 24.0 * (-16 * u**3 + 12 * u**2 + 12 * u + 4)
    #else:
    #    u = t - 2.0
    #    return sgn * 1 / 24.0 * (24 * u**3 - 36 * u**2 - 12 * u + 12)

    sgn = -sgn_device(t)
    t = 2 - clip_device(abs(t), 0, 2)
    if t < 1.0:
        return sgn * (1 / 2.0 * t**2)
    else:
        u = t - 1.0
        return sgn * (1 / 2.0 * (-3.0 * u**2 + 2.0 * u + 1.0))

@cuda.jit(device=True)
def compute_grad_grad_weight_device(t):
    #t = 2.5 - clip_device(abs(t), 0, 2.5)
    #if t < 1.0:
    #    return 12 / 24.0 * t**2
    #elif t < 2.0:
    #    u = t - 1.0
    #    return 1 / 24.0 * (-48 * u**2 + 24 * u + 12)
    #else:
    #    u = t - 2.0
    #    return 1 / 24.0 * (72 * u**2 - 72 * u - 12)

    if abs(t) > 2:
        return 0.0

    t = 2 - clip_device(abs(t), 0, 2)
    if t < 1.0:
        return (t)
    else:
        u = t - 1.0
        return (-3.0 * u + 1.0)

@cuda.jit(device=True)
def compute_grad_grad_grad_weight_device(t):
    #t = 2.5 - clip_device(abs(t), 0, 2.5)
    #if t < 1.0:
    #    return 12 / 24.0 * t**2
    #elif t < 2.0:
    #    u = t - 1.0
    #    return 1 / 24.0 * (-48 * u**2 + 24 * u + 12)
    #else:
    #    u = t - 2.0
    #    return 1 / 24.0 * (72 * u**2 - 72 * u - 12)

    if abs(t) > 2:
        return 0.0

    sgn = -sgn_device(t)
    t = 2 - clip_device(abs(t), 0, 2)
    if t < 1.0:
        return 1.0 * sgn
    else:
        return -3.0 * sgn

@cuda.jit
def compute_sdf_value_forward_kernel(
    # inputs
    grid, x,
    # parameters
    grid_range,
    # sampling parameters
    out_value
):
    grid_resolution_u = grid.shape[-1]
    grid_resolution_v = grid.shape[-2]
    grid_resolution_w = grid.shape[-3]

    idx_sample = cuda.blockIdx.x

    idx_thread = cuda.threadIdx.x
    blockdim = cuda.blockDim.x

    x = x[idx_sample]

    u = 0.5 * (x[0] / grid_range + 1) * grid_resolution_u
    v = 0.5 * (x[1] / grid_range + 1) * grid_resolution_v
    w = 0.5 * (x[2] / grid_range + 1) * grid_resolution_w

    u = clip_device(u, 1, grid_resolution_u - 1 - 1e-4)
    v = clip_device(v, 1, grid_resolution_v - 1 - 1e-4)
    w = clip_device(w, 1, grid_resolution_w - 1 - 1e-4)

    sdf_value_tmp = cuda.shared.array(shape=(128,), dtype=numba.float32)

    sdf_value_tmp[idx_thread] = 0.0

    idx_ofs = idx_thread
    idx_ofs = idx_thread
    while idx_ofs < (basis_radius)**3:
                ofs_u = (idx_ofs % basis_radius) - 2 * basis_scale
                ofs_v = ((idx_ofs // basis_radius) % basis_radius) - 2 * basis_scale
                ofs_w = (idx_ofs // basis_radius**2) - 2 * basis_scale

                u_err = u - (int(u) + ofs_u + 0.5)
                v_err = v - (int(v) + ofs_v + 0.5)
                w_err = w - (int(w) + ofs_w + 0.5)

                wu = compute_weight_device(u_err / basis_scale) / basis_scale
                wv = compute_weight_device(v_err / basis_scale) / basis_scale
                ww = compute_weight_device(w_err / basis_scale) / basis_scale

                if (wu == 0) or (wv == 0) or (ww == 0):
                    pass
                else:

                    val = grid[
                        clip_device(int(w) + ofs_w, 0, grid_resolution_w-1), 
                        clip_device(int(v) + ofs_v, 0, grid_resolution_v-1), 
                        clip_device(int(u) + ofs_u, 0, grid_resolution_u-1)
                    ]

                    sdf_value_tmp[idx_thread] += wu * wv * ww * val

                idx_ofs += blockdim

    cuda.syncthreads()
    if idx_thread == 0:
        for j in range(1,blockdim):
            sdf_value_tmp[0] += sdf_value_tmp[j]
        out_value[idx_sample,0] = sdf_value_tmp[0]

@cuda.jit
def compute_sdf_value_backward_kernel(
    # inputs
    grid, x, grad_out_value,
    # parameters
    grid_range,
    # sampling parameters
    grad_grid, grad_x
):
    grid_resolution = grid.shape[0]

    idx_sample = cuda.blockIdx.x

    idx_thread = cuda.threadIdx.x
    blockdim = cuda.blockDim.x

    x = x[idx_sample]
    grad_out_value = grad_out_value[idx_sample,0]

    grad_x_tmp = cuda.shared.array(shape=(125,3), dtype=numba.float32)
    grad_x_tmp[idx_thread][0] = grad_x_tmp[idx_thread][1] = grad_x_tmp[idx_thread][2] = 0.0

    if (grad_out_value == 0):
        pass
    else:
        u = 0.5 * (x[0] / grid_range + 1) * grid_resolution
        v = 0.5 * (x[1] / grid_range + 1) * grid_resolution
        w = 0.5 * (x[2] / grid_range + 1) * grid_resolution

        u = clip_device(u, 1, grid_resolution - 1 - 1e-4)
        v = clip_device(v, 1, grid_resolution - 1 - 1e-4)
        w = clip_device(w, 1, grid_resolution - 1 - 1e-4)


        idx_ofs = idx_thread
        while idx_ofs < (basis_radius)**3:
                    ofs_u = (idx_ofs % basis_radius) - 2 * basis_scale
                    ofs_v = ((idx_ofs // basis_radius) % basis_radius) - 2 * basis_scale
                    ofs_w = (idx_ofs // basis_radius**2) - 2 * basis_scale

                    u_err = u - (int(u) + ofs_u + 0.5)
                    v_err = v - (int(v) + ofs_v + 0.5)
                    w_err = w - (int(w) + ofs_w + 0.5)

                    wu = compute_weight_device(u_err / basis_scale) / basis_scale
                    wv = compute_weight_device(v_err / basis_scale) / basis_scale
                    ww = compute_weight_device(w_err / basis_scale) / basis_scale

                    if (wu == 0) or (wv == 0) or (ww == 0):
                        pass
                    else:

                        val = grid[
                            clip_device(int(w) + ofs_w, 0, grid_resolution-1), 
                            clip_device(int(v) + ofs_v, 0, grid_resolution-1), 
                            clip_device(int(u) + ofs_u, 0, grid_resolution-1)
                        ]

                        grad_u = compute_grad_weight_device(u_err / basis_scale) / (basis_scale**2)
                        grad_v = compute_grad_weight_device(v_err / basis_scale) / (basis_scale**2)
                        grad_w = compute_grad_weight_device(w_err / basis_scale) / (basis_scale**2)

                        c = 0.5 * grid_resolution / grid_range

                        grad_val = 0.0
                        if grad_out_value != 0.0:
                            grad_val += wu * wv * ww * grad_out_value

                            grad_x_tmp[idx_thread][0] += wv * ww * val * grad_u * c * grad_out_value
                            grad_x_tmp[idx_thread][1] += wu * ww * val * grad_v * c * grad_out_value
                            grad_x_tmp[idx_thread][2] += wu * wv * val * grad_w * c * grad_out_value

                        if grad_val != 0:
                            cuda.atomic.add(grad_grid, (
                                clip_device(int(w) + ofs_w, 0, grid_resolution-1), 
                                clip_device(int(v) + ofs_v, 0, grid_resolution-1), 
                                clip_device(int(u) + ofs_u, 0, grid_resolution-1)
                            ), grad_val)

                    idx_ofs += blockdim

    cuda.syncthreads()
    if idx_thread == 0:
        for j in range(1,blockdim):
            grad_x_tmp[0,0] += grad_x_tmp[j,0]
            grad_x_tmp[0,1] += grad_x_tmp[j,1]
            grad_x_tmp[0,2] += grad_x_tmp[j,2]
        grad_x[idx_sample,0] = grad_x_tmp[0][0]
        grad_x[idx_sample,1] = grad_x_tmp[0][1]
        grad_x[idx_sample,2] = grad_x_tmp[0][2]

@cuda.jit
def compute_sdf_value_and_grad_forward_kernel(
    # inputs
    grid, x,
    # parameters
    grid_range,
    # sampling parameters
    out_value, out_grad
):
    grid_resolution = grid.shape[-1]

    idx_sample = cuda.blockIdx.x

    idx_thread = cuda.threadIdx.x
    blockdim = cuda.blockDim.x

    x = x[idx_sample]

    u = 0.5 * (x[0] / grid_range + 1) * grid_resolution
    v = 0.5 * (x[1] / grid_range + 1) * grid_resolution
    w = 0.5 * (x[2] / grid_range + 1) * grid_resolution

    u = clip_device(u, 1, grid_resolution - 1 - 1e-4)
    v = clip_device(v, 1, grid_resolution - 1 - 1e-4)
    w = clip_device(w, 1, grid_resolution - 1 - 1e-4)

    sdf_value_tmp = cuda.shared.array(shape=(128,), dtype=numba.float32)
    sdf_grad_tmp = cuda.shared.array(shape=(128,3), dtype=numba.float32)

    sdf_value_tmp[idx_thread] = 0.0
    sdf_grad_tmp[idx_thread][0] = sdf_grad_tmp[idx_thread][1] = sdf_grad_tmp[idx_thread][2] = 0.0

    idx_ofs = idx_thread
    while idx_ofs < (basis_radius)**3:
                ofs_u = (idx_ofs % basis_radius) - 2 * basis_scale
                ofs_v = ((idx_ofs // basis_radius) % basis_radius) - 2 * basis_scale
                ofs_w = (idx_ofs // basis_radius**2) - 2 * basis_scale

                u_err = u - (int(u) + ofs_u + 0.5)
                v_err = v - (int(v) + ofs_v + 0.5)
                w_err = w - (int(w) + ofs_w + 0.5)

                wu = compute_weight_device(u_err / basis_scale) / basis_scale
                wv = compute_weight_device(v_err / basis_scale) / basis_scale
                ww = compute_weight_device(w_err / basis_scale) / basis_scale

                if (wu == 0) or (wv == 0) or (ww == 0):
                    pass
                else:

                    val = grid[
                        clip_device(int(w) + ofs_w, 0, grid_resolution-1), 
                        clip_device(int(v) + ofs_v, 0, grid_resolution-1), 
                        clip_device(int(u) + ofs_u, 0, grid_resolution-1)
                    ]

                    sdf_value_tmp[idx_thread] += wu * wv * ww * val

                    grad_u = compute_grad_weight_device(u_err / basis_scale) / (basis_scale**2)
                    grad_v = compute_grad_weight_device(v_err / basis_scale) / (basis_scale**2)
                    grad_w = compute_grad_weight_device(w_err / basis_scale) / (basis_scale**2)

                    sdf_grad_tmp[idx_thread][0] += wv * ww * val * grad_u
                    sdf_grad_tmp[idx_thread][1] += wu * ww * val * grad_v
                    sdf_grad_tmp[idx_thread][2] += wu * wv * val * grad_w

                idx_ofs += blockdim

    cuda.syncthreads()
    num_threads_valid = blockdim
    while num_threads_valid > 1:
        curr_ofs = ((num_threads_valid + 1) // 2)
        if idx_thread < curr_ofs:
            if (idx_thread + curr_ofs) < num_threads_valid:
                j = idx_thread + curr_ofs
                sdf_value_tmp[idx_thread] += sdf_value_tmp[j]
                sdf_grad_tmp[idx_thread,0] += sdf_grad_tmp[j,0]
                sdf_grad_tmp[idx_thread,1] += sdf_grad_tmp[j,1]
                sdf_grad_tmp[idx_thread,2] += sdf_grad_tmp[j,2]
        num_threads_valid = curr_ofs
        cuda.syncthreads()

    c = 0.5 * grid_resolution / grid_range # gradient of uvw w.r.t. x
    if idx_thread == 0:
        out_value[idx_sample,0] = sdf_value_tmp[0]
        out_grad[idx_sample,0] = sdf_grad_tmp[0][0] * c
        out_grad[idx_sample,1] = sdf_grad_tmp[0][1] * c
        out_grad[idx_sample,2] = sdf_grad_tmp[0][2] * c

@cuda.jit
def compute_sdf_value_and_grad_backward_kernel(
    # inputs
    grid, x, grad_out_value, grad_out_grad,
    # parameters
    grid_range,
    # sampling parameters
    grad_grid, grad_x
):
    grid_resolution = grid.shape[0]

    idx_sample = cuda.blockIdx.x

    idx_thread = cuda.threadIdx.x
    blockdim = cuda.blockDim.x

    x = x[idx_sample]
    grad_out_value = grad_out_value[idx_sample,0]
    grad_out_grad = grad_out_grad[idx_sample]

    grad_x_tmp = cuda.shared.array(shape=(125,3), dtype=numba.float32)
    grad_x_tmp[idx_thread][0] = grad_x_tmp[idx_thread][1] = grad_x_tmp[idx_thread][2] = 0.0

    if (grad_out_value == 0) and (grad_out_grad[0] == 0) and (grad_out_grad[1] == 0) and (grad_out_grad[2] == 0):
        pass
    else:
        u = 0.5 * (x[0] / grid_range + 1) * grid_resolution
        v = 0.5 * (x[1] / grid_range + 1) * grid_resolution
        w = 0.5 * (x[2] / grid_range + 1) * grid_resolution

        u = clip_device(u, 1, grid_resolution - 1 - 1e-4)
        v = clip_device(v, 1, grid_resolution - 1 - 1e-4)
        w = clip_device(w, 1, grid_resolution - 1 - 1e-4)

        idx_ofs = idx_thread
        while idx_ofs < (basis_radius)**3:
                    ofs_u = (idx_ofs % basis_radius) - 2 * basis_scale
                    ofs_v = ((idx_ofs // basis_radius) % basis_radius) - 2 * basis_scale
                    ofs_w = (idx_ofs // basis_radius**2) - 2 * basis_scale

                    u_err = u - (int(u) + ofs_u + 0.5)
                    v_err = v - (int(v) + ofs_v + 0.5)
                    w_err = w - (int(w) + ofs_w + 0.5)

                    wu = compute_weight_device(u_err / basis_scale) / basis_scale
                    wv = compute_weight_device(v_err / basis_scale) / basis_scale
                    ww = compute_weight_device(w_err / basis_scale) / basis_scale

                    if (wu == 0) or (wv == 0) or (ww == 0):
                        pass
                    else:

                        val = grid[
                            clip_device(int(w) + ofs_w, 0, grid_resolution-1), 
                            clip_device(int(v) + ofs_v, 0, grid_resolution-1), 
                            clip_device(int(u) + ofs_u, 0, grid_resolution-1)
                        ]

                        grad_u = compute_grad_weight_device(u_err / basis_scale) / (basis_scale**2)
                        grad_v = compute_grad_weight_device(v_err / basis_scale) / (basis_scale**2)
                        grad_w = compute_grad_weight_device(w_err / basis_scale) / (basis_scale**2)

                        c = 0.5 * grid_resolution / grid_range

                        grad_val = 0.0
                        if grad_out_value != 0.0:
                            grad_val += wu * wv * ww * grad_out_value

                            grad_x_tmp[idx_thread][0] += wv * ww * val * grad_u * c * grad_out_value
                            grad_x_tmp[idx_thread][1] += wu * ww * val * grad_v * c * grad_out_value
                            grad_x_tmp[idx_thread][2] += wu * wv * val * grad_w * c * grad_out_value

                        if (grad_out_grad[0] != 0) or (grad_out_grad[1] != 0) or (grad_out_grad[2] != 0):
                            grad_val += wv * ww * grad_u * c * grad_out_grad[0]
                            grad_val += wu * ww * grad_v * c * grad_out_grad[1]
                            grad_val += wu * wv * grad_w * c * grad_out_grad[2]

                            grad2_u = compute_grad_grad_weight_device(u_err / basis_scale) / (basis_scale**3)
                            grad2_v = compute_grad_grad_weight_device(v_err / basis_scale) / (basis_scale**3)
                            grad2_w = compute_grad_grad_weight_device(w_err / basis_scale) / (basis_scale**3)

                            grad_x_tmp[idx_thread][0] += grad2_u * wv * ww * val * c**2 * grad_out_grad[0]
                            grad_x_tmp[idx_thread][0] += grad_u * grad_v * ww * val * c**2 * grad_out_grad[1]
                            grad_x_tmp[idx_thread][0] += grad_u * wv * grad_w * val * c**2 * grad_out_grad[2]

                            grad_x_tmp[idx_thread][1] += grad_u * grad_v * ww * val * c**2 * grad_out_grad[0]
                            grad_x_tmp[idx_thread][1] += wu * grad2_v * ww * val * c**2 * grad_out_grad[1]
                            grad_x_tmp[idx_thread][1] += wu * grad_v * grad_w * val * c**2 * grad_out_grad[2]

                            grad_x_tmp[idx_thread][2] += grad_u * wv * grad_w * val * c**2 * grad_out_grad[0]                
                            grad_x_tmp[idx_thread][2] += wu * grad_v * grad_w * val * c**2 * grad_out_grad[1]
                            grad_x_tmp[idx_thread][2] += wu * wv * grad2_w * val * c**2 * grad_out_grad[2]

                        if grad_val != 0:
                            cuda.atomic.add(grad_grid, (
                                clip_device(int(w) + ofs_w, 0, grid_resolution-1), 
                                clip_device(int(v) + ofs_v, 0, grid_resolution-1), 
                                clip_device(int(u) + ofs_u, 0, grid_resolution-1)
                            ), grad_val)

                    idx_ofs += blockdim

    cuda.syncthreads()
    num_threads_valid = blockdim
    while num_threads_valid > 1:
        curr_ofs = ((num_threads_valid + 1) // 2)
        if idx_thread < curr_ofs:
            if (idx_thread + curr_ofs) < num_threads_valid:
                j = idx_thread + curr_ofs
                grad_x_tmp[idx_thread,0] += grad_x_tmp[j,0]
                grad_x_tmp[idx_thread,1] += grad_x_tmp[j,1]
                grad_x_tmp[idx_thread,2] += grad_x_tmp[j,2]
        num_threads_valid = curr_ofs
        cuda.syncthreads()

    if idx_thread == 0:
        #for j in range(1,blockdim):
        #    grad_x_tmp[0,0] += grad_x_tmp[j,0]
        #    grad_x_tmp[0,1] += grad_x_tmp[j,1]
        #    grad_x_tmp[0,2] += grad_x_tmp[j,2]
        grad_x[idx_sample,0] = grad_x_tmp[0][0]
        grad_x[idx_sample,1] = grad_x_tmp[0][1]
        grad_x[idx_sample,2] = grad_x_tmp[0][2]

@cuda.jit
def compute_hessian_forward_kernel(
    # inputs
    grid, x,
    # parameters
    grid_range,
    # sampling parameters
    out_hessian_matrix
):
    grid_resolution = grid.shape[0]

    idx_sample = cuda.blockIdx.x

    idx_thread = cuda.threadIdx.x
    blockdim = cuda.blockDim.x

    x = x[idx_sample]

    u = 0.5 * (x[0] / grid_range + 1) * grid_resolution
    v = 0.5 * (x[1] / grid_range + 1) * grid_resolution
    w = 0.5 * (x[2] / grid_range + 1) * grid_resolution

    u = clip_device(u, 1, grid_resolution - 1 - 1e-4)
    v = clip_device(v, 1, grid_resolution - 1 - 1e-4)
    w = clip_device(w, 1, grid_resolution - 1 - 1e-4)

    sdf_hessian_tmp = cuda.shared.array(shape=(128,6), dtype=numba.float32)
    for i in range(6):
        sdf_hessian_tmp[idx_thread,i] = 0.0

    idx_ofs = idx_thread
    while idx_ofs < (basis_radius)**3:
                ofs_u = (idx_ofs % basis_radius) - 2 * basis_scale
                ofs_v = ((idx_ofs // basis_radius) % basis_radius) - 2 * basis_scale
                ofs_w = (idx_ofs // basis_radius**2) - 2 * basis_scale

                u_err = u - (int(u) + ofs_u + 0.5)
                v_err = v - (int(v) + ofs_v + 0.5)
                w_err = w - (int(w) + ofs_w + 0.5)

                wu = compute_weight_device(u_err / basis_scale) / basis_scale
                wv = compute_weight_device(v_err / basis_scale) / basis_scale
                ww = compute_weight_device(w_err / basis_scale) / basis_scale

                if (wu == 0) or (wv == 0) or (ww == 0):
                    pass
                else:

                    val = grid[
                        clip_device(int(w) + ofs_w, 0, grid_resolution-1), 
                        clip_device(int(v) + ofs_v, 0, grid_resolution-1), 
                        clip_device(int(u) + ofs_u, 0, grid_resolution-1)
                    ]

                    grad_u = compute_grad_weight_device(u_err / basis_scale) / (basis_scale**2)
                    grad_v = compute_grad_weight_device(v_err / basis_scale) / (basis_scale**2)
                    grad_w = compute_grad_weight_device(w_err / basis_scale) / (basis_scale**2)

                    grad2_u = compute_grad_grad_weight_device(u_err / basis_scale) / (basis_scale**3)
                    grad2_v = compute_grad_grad_weight_device(v_err / basis_scale) / (basis_scale**3)
                    grad2_w = compute_grad_grad_weight_device(w_err / basis_scale) / (basis_scale**3)

                    c = 0.5 * grid_resolution / grid_range
                    sdf_hessian_tmp[idx_thread][0] += grad2_u * wv * ww * val * c**2
                    sdf_hessian_tmp[idx_thread][1] += grad2_v * wu * ww * val * c**2
                    sdf_hessian_tmp[idx_thread][2] += grad2_w * wu * wv * val * c**2

                    sdf_hessian_tmp[idx_thread][3] += grad_u * grad_v * ww * val * c**2
                    sdf_hessian_tmp[idx_thread][4] += grad_v * grad_w * wu * val * c**2
                    sdf_hessian_tmp[idx_thread][5] += grad_u * grad_w * wv * val * c**2

                idx_ofs += blockdim

    cuda.syncthreads()
    num_threads_valid = blockdim
    while num_threads_valid > 1:
        curr_ofs = ((num_threads_valid + 1) // 2)
        if idx_thread < curr_ofs:
            if (idx_thread + curr_ofs) < num_threads_valid:
                j = idx_thread + curr_ofs
                for k in range(6):
                    sdf_hessian_tmp[idx_thread,k] += sdf_hessian_tmp[j,k]
        num_threads_valid = curr_ofs
        cuda.syncthreads()

    if idx_thread == 0:
        #for j in range(1,blockdim):
        #    for k in range(6):
        #        sdf_hessian_tmp[0,k] += sdf_hessian_tmp[j,k]
        out_hessian_matrix[idx_sample,0,0] = sdf_hessian_tmp[0][0]
        out_hessian_matrix[idx_sample,1,1] = sdf_hessian_tmp[0][1]
        out_hessian_matrix[idx_sample,2,2] = sdf_hessian_tmp[0][2]
        out_hessian_matrix[idx_sample,0,1] = out_hessian_matrix[idx_sample,1,0] = sdf_hessian_tmp[0][3]
        out_hessian_matrix[idx_sample,1,2] = out_hessian_matrix[idx_sample,2,1] = sdf_hessian_tmp[0][4]
        out_hessian_matrix[idx_sample,0,2] = out_hessian_matrix[idx_sample,2,0] = sdf_hessian_tmp[0][5]

@cuda.jit
def compute_hessian_backward_kernel(
    # inputs
    grid, x, grad_out,
    # parameters
    grid_range,
    # sampling parameters
    grad_grid, grad_x
):
    grid_resolution = grid.shape[0]

    idx_sample = cuda.blockIdx.x

    idx_thread = cuda.threadIdx.x
    blockdim = cuda.blockDim.x

    x = x[idx_sample]
    grad_out = grad_out[idx_sample]

    u = 0.5 * (x[0] / grid_range + 1) * grid_resolution
    v = 0.5 * (x[1] / grid_range + 1) * grid_resolution
    w = 0.5 * (x[2] / grid_range + 1) * grid_resolution

    u = clip_device(u, 1, grid_resolution - 1 - 1e-4)
    v = clip_device(v, 1, grid_resolution - 1 - 1e-4)
    w = clip_device(w, 1, grid_resolution - 1 - 1e-4)

    grad_x_tmp = cuda.shared.array(shape=(128,3), dtype=numba.float32)
    for i in range(3):
        grad_x_tmp[idx_thread,i] = 0.0

    idx_ofs = idx_thread
    while idx_ofs < (basis_radius)**3:
                ofs_u = (idx_ofs % basis_radius) - 2 * basis_scale
                ofs_v = ((idx_ofs // basis_radius) % basis_radius) - 2 * basis_scale
                ofs_w = (idx_ofs // basis_radius**2) - 2 * basis_scale

                u_err = u - (int(u) + ofs_u + 0.5)
                v_err = v - (int(v) + ofs_v + 0.5)
                w_err = w - (int(w) + ofs_w + 0.5)

                wu = compute_weight_device(u_err / basis_scale) / basis_scale
                wv = compute_weight_device(v_err / basis_scale) / basis_scale
                ww = compute_weight_device(w_err / basis_scale) / basis_scale

                if (wu == 0) or (wv == 0) or (ww == 0):
                    pass
                else:

                    grad_val = 0.0

                    grad_u = compute_grad_weight_device(u_err / basis_scale) / (basis_scale**2)
                    grad_v = compute_grad_weight_device(v_err / basis_scale) / (basis_scale**2)
                    grad_w = compute_grad_weight_device(w_err / basis_scale) / (basis_scale**2)

                    grad2_u = compute_grad_grad_weight_device(u_err / basis_scale) / (basis_scale**3)
                    grad2_v = compute_grad_grad_weight_device(v_err / basis_scale) / (basis_scale**3)
                    grad2_w = compute_grad_grad_weight_device(w_err / basis_scale) / (basis_scale**3)

                    grad3_u = compute_grad_grad_grad_weight_device(u_err / basis_scale) / (basis_scale**4)
                    grad3_v = compute_grad_grad_grad_weight_device(v_err / basis_scale) / (basis_scale**4)
                    grad3_w = compute_grad_grad_grad_weight_device(w_err / basis_scale) / (basis_scale**4)

                    c = 0.5 * grid_resolution / grid_range
                    grad_val += grad2_u * wv * ww * c**2 * grad_out[0,0]
                    grad_val += grad2_v * wu * ww * c**2 * grad_out[1,1]
                    grad_val += grad2_w * wu * wv * c**2 * grad_out[2,2]

                    grad_val += grad_u * grad_v * ww * c**2 * (grad_out[0,1] + grad_out[1,0])
                    grad_val += grad_v * grad_w * wu * c**2 * (grad_out[1,2] + grad_out[2,1])
                    grad_val += grad_u * grad_w * wv * c**2 * (grad_out[0,2] + grad_out[2,0])

                    if grad_val != 0:
                        cuda.atomic.add(grad_grid, (
                            clip_device(int(w) + ofs_w, 0, grid_resolution-1), 
                            clip_device(int(v) + ofs_v, 0, grid_resolution-1), 
                            clip_device(int(u) + ofs_u, 0, grid_resolution-1)
                        ), grad_val)

                    # TODO: backward to x
                    val = grid[
                        clip_device(int(w) + ofs_w, 0, grid_resolution-1), 
                        clip_device(int(v) + ofs_v, 0, grid_resolution-1), 
                        clip_device(int(u) + ofs_u, 0, grid_resolution-1)
                    ]

                    #sdf_hessian_tmp[idx_thread][0] += grad2_u * wv * ww * val * c**2
                    #sdf_hessian_tmp[idx_thread][1] += grad2_v * wu * ww * val * c**2
                    #sdf_hessian_tmp[idx_thread][2] += grad2_w * wu * wv * val * c**2

                    #sdf_hessian_tmp[idx_thread][3] += grad_u * grad_v * ww * val * c**2
                    #sdf_hessian_tmp[idx_thread][4] += grad_v * grad_w * wu * val * c**2
                    #sdf_hessian_tmp[idx_thread][5] += grad_u * grad_w * wv * val * c**2

                    grad_x_tmp[idx_thread][0] += grad3_u * wv * ww * val * c**2 * grad_out[0,0]
                    grad_x_tmp[idx_thread][0] += grad2_v * grad_u * ww * val * c**2 * grad_out[1,1]
                    grad_x_tmp[idx_thread][0] += grad2_w * grad_u * wv * val * c**2 * grad_out[2,2]
                    grad_x_tmp[idx_thread][0] += grad2_u * grad_v * ww * val * c**2 * (grad_out[0,1] + grad_out[1,0])
                    grad_x_tmp[idx_thread][0] += grad_v * grad_w * grad_u * val * c**2 * (grad_out[1,2] + grad_out[2,1])
                    grad_x_tmp[idx_thread][0] += grad2_u * grad_w * wv * val * c**2 * (grad_out[0,2] + grad_out[2,0])

                    grad_x_tmp[idx_thread][1] += grad2_u * grad_v * ww * val * c**2 * grad_out[0,0]
                    grad_x_tmp[idx_thread][1] += grad3_v * wu * ww * val * c**2 * grad_out[1,1]
                    grad_x_tmp[idx_thread][1] += grad2_w * wu * grad_v * val * c**2 * grad_out[2,2]
                    grad_x_tmp[idx_thread][1] += grad_u * grad2_v * ww * val * c**2 * (grad_out[0,1] + grad_out[1,0])
                    grad_x_tmp[idx_thread][1] += grad2_v * grad_w * wu * val * c**2 * (grad_out[1,2] + grad_out[2,1])
                    grad_x_tmp[idx_thread][1] += grad_u * grad_w * grad_v * val * c**2 * (grad_out[0,2] + grad_out[2,0])

                    grad_x_tmp[idx_thread][2] += grad2_u * wv * grad_w * val * c**2 * grad_out[0,0]
                    grad_x_tmp[idx_thread][2] += grad2_v * wu * grad_w * val * c**2 * grad_out[1,1]
                    grad_x_tmp[idx_thread][2] += grad3_w * wu * wv * val * c**2 * grad_out[2,2]
                    grad_x_tmp[idx_thread][2] += grad_u * grad_v * grad_w * val * c**2 * (grad_out[0,1] + grad_out[1,0])
                    grad_x_tmp[idx_thread][2] += grad_v * grad2_w * wu * val * c**2 * (grad_out[1,2] + grad_out[2,1])
                    grad_x_tmp[idx_thread][2] += grad_u * grad2_w * wv * val * c**2 * (grad_out[0,2] + grad_out[2,0])

                idx_ofs += blockdim

    # TODO: backward to x
    cuda.syncthreads()
    num_threads_valid = blockdim
    while num_threads_valid > 1:
        curr_ofs = ((num_threads_valid + 1) // 2)
        if idx_thread < curr_ofs:
            if (idx_thread + curr_ofs) < num_threads_valid:
                j = idx_thread + curr_ofs
                grad_x_tmp[idx_thread,0] += grad_x_tmp[j,0]
                grad_x_tmp[idx_thread,1] += grad_x_tmp[j,1]
                grad_x_tmp[idx_thread,2] += grad_x_tmp[j,2]
        num_threads_valid = curr_ofs
        cuda.syncthreads()

    if idx_thread == 0:
        grad_x[idx_sample,0] = grad_x_tmp[0][0]
        grad_x[idx_sample,1] = grad_x_tmp[0][1]
        grad_x[idx_sample,2] = grad_x_tmp[0][2]

# src: BS*C*D*H*W
# new_grid: BS*D'*H'*W'*3
def grid_sample_3D_B_Spline(
        src,
        new_grid
):
    if len(src) > 1:
        results = []
        for i in range(len(src)):
            results.append(grid_sample_3D_B_Spline(src[i:i+1], new_grid[i:i+1]))

    assert src.size(1) == 1
    assert new_grid.size(0) == 1
    result = torch.zeros_like(new_grid[0].view(-1,3)[:,0:1], device=src.device, dtype=src.dtype)
    compute_sdf_value_forward_kernel[(result.size(0),),(125,)](
        # inputs
        src[0,0], new_grid[0].view(-1,3),
        # parameters
        1,
        # sampling parameters
        result
    )
    return result.view(1,1,new_grid.size(1),new_grid.size(2),new_grid.size(3))

class compute_sdf_value(torch.autograd.Function):
    # grid : S * S * S
    # x    : N * 3
    @staticmethod
    def forward(ctx, grid, x, grid_range):
        dtype = x.dtype
        device = x.device

        sdf_value = torch.zeros_like(x[:,0:1], dtype=dtype, device=device)

        compute_sdf_value_forward_kernel[(x.size(0),),(125,)](
            # inputs
            grid.detach(), x.detach(),
            # parameters
            grid_range.item(),
            # sampling parameters
            sdf_value
        )

        ctx.save_for_backward(
            grid, x, grid_range
        )
        return sdf_value

    @staticmethod
    def backward(ctx, grad_out_value):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grid, x, grid_range, = ctx.saved_tensors

        dtype = x.dtype
        device = x.device

        grad_grid = torch.zeros_like(grid, dtype=dtype, device=device)
        grad_x = torch.zeros_like(x, dtype=dtype, device=device)

        compute_sdf_value_backward_kernel[(x.size(0),),(125,)](
            # inputs
            grid.detach(), x.detach(), grad_out_value.detach(),
            # parameters
            grid_range.item(),
            # sampling parameters
            grad_grid, grad_x
        )

        return grad_grid, grad_x, None

class b_spline_interpolation(torch.autograd.Function):
    # grid : S * S * S
    # x    : N * 3
    @staticmethod
    def forward(ctx, grid, x, grid_range):
        dtype = x.dtype
        device = x.device

        sdf_value = torch.zeros_like(x[:,0:1], dtype=dtype, device=device)
        sdf_grad = torch.zeros_like(x, dtype=dtype, device=device)

        compute_sdf_value_and_grad_forward_kernel[(x.size(0),),(125,)](
            # inputs
            grid.detach(), x.detach(),
            # parameters
            grid_range.item(),
            # sampling parameters
            sdf_value, sdf_grad
        )

        ctx.save_for_backward(
            grid, x, grid_range
        )
        return sdf_value, sdf_grad

    @staticmethod
    def backward(ctx, grad_out_value, grad_out_grad):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grid, x, grid_range, = ctx.saved_tensors

        dtype = x.dtype
        device = x.device

        grad_grid = torch.zeros_like(grid, dtype=dtype, device=device)
        grad_x = torch.zeros_like(x, dtype=dtype, device=device)

        compute_sdf_value_and_grad_backward_kernel[(x.size(0),),(125,)](
            # inputs
            grid.detach(), x.detach(), grad_out_value.detach(), grad_out_grad.detach(),
            # parameters
            grid_range.item(),
            # sampling parameters
            grad_grid, grad_x
        )

        return grad_grid, grad_x, None


class compute_sdf_hessian(torch.autograd.Function):
    # grid : S * S * S
    # x    : N * 3
    @staticmethod
    def forward(ctx, grid, x, grid_range):
        out_hessian_matrix = torch.zeros((x.size(0),3,3), dtype=x.dtype, device=x.device)

        compute_hessian_forward_kernel[(x.size(0),),(125,)](
            # inputs
            grid.detach(), x.detach(),
            # parameters
            grid_range.item(),
            # sampling parameters
            out_hessian_matrix
        )

        ctx.save_for_backward(
            grid, x, grid_range
        )
        return out_hessian_matrix

    @staticmethod
    def backward(ctx, grad_out):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grid, x, grid_range, = ctx.saved_tensors

        dtype = x.dtype
        device = x.device

        grad_grid = torch.zeros_like(grid, dtype=dtype, device=device)
        grad_x = torch.zeros_like(x, dtype=dtype, device=device)

        compute_hessian_backward_kernel[(x.size(0),),(125,)](
            # inputs
            grid.detach(), x.detach(), grad_out.detach(),
            # parameters
            grid_range.item(),
            # sampling parameters
            grad_grid, grad_x
        )

        return grad_grid, grad_x, None

@cuda.jit
def sphere_tracing_kernel(
    # inputs
    grid, ray_origin, ray_direction, t0, max_t, mask,
    is_converged,
    # parameters
    grid_range, threshold,
    # output
    t_surf, t_eval, grad_t_eval
):
    grid_resolution = grid.shape[0]

    idx_sample = cuda.blockIdx.x

    idx_thread = cuda.threadIdx.x
    blockdim = cuda.blockDim.x

    r = ray_direction[idx_sample]
    t0 = t0[idx_sample,0]
    dtdr0 = grad_t_eval[idx_sample]
    max_t = max_t[idx_sample,0]

    x = cuda.shared.array(shape=(3), dtype=numba.float32)
    if idx_thread == 0:
        x[0] = ray_origin[idx_sample,0] + t0 * r[0]
        x[1] = ray_origin[idx_sample,1] + t0 * r[1]
        x[2] = ray_origin[idx_sample,2] + t0 * r[2]
    sdf_value = 1000.0
    cuda.syncthreads()

    sum_w = 0.0
    sum_w_t = 0.0
    sum_grad_w = (0.0,0.0,0.0)
    sum_grad_w_t = (0.0,0.0,0.0)

    dxdr = (
        (t0, 0, 0),
        (0, t0, 0),
        (0, 0, t0)
    )
    t = t0
    dtdr_t = (dtdr0[0],dtdr0[1],dtdr0[2])
    dxdr = add_mat33_device(
        dxdr,
        (
            (r[0] * dtdr_t[0], r[0] * dtdr_t[1], r[0] * dtdr_t[2]),
            (r[1] * dtdr_t[0], r[1] * dtdr_t[1], r[1] * dtdr_t[2]),
            (r[2] * dtdr_t[0], r[2] * dtdr_t[1], r[2] * dtdr_t[2]),
        )
    )
    #prev_c = 0.0
    sum_weight_dist = 0.0
    prev_sdf_value = 0.0
    prev_sdf_grad = (0.0,0.0,0.0)
    prev_t = t
    prev_dfdr_t = (0.0, 0.0, 0.0)

    if mask[idx_sample] != 0.0:
        for idx_step in range(1000):
            # query SDF value
            u = 0.5 * (x[0] / grid_range + 1) * grid_resolution
            v = 0.5 * (x[1] / grid_range + 1) * grid_resolution
            w = 0.5 * (x[2] / grid_range + 1) * grid_resolution

            u = clip_device(u, 1, grid_resolution - 1 - 1e-4)
            v = clip_device(v, 1, grid_resolution - 1 - 1e-4)
            w = clip_device(w, 1, grid_resolution - 1 - 1e-4)

            sdf_value_tmp = cuda.shared.array(shape=(128,), dtype=numba.float32)
            sdf_grad_tmp = cuda.shared.array(shape=(128,3), dtype=numba.float32)
            sdf_hessian_tmp = cuda.shared.array(shape=(128,6), dtype=numba.float32)
            sdf_value_tmp[idx_thread] = 0.0
            sdf_grad_tmp[idx_thread,0] = 0.0
            sdf_grad_tmp[idx_thread,1] = 0.0
            sdf_grad_tmp[idx_thread,2] = 0.0
            for i in range(6):
                sdf_hessian_tmp[idx_thread,i] = 0.0
            #sdf_value = 0.0
            idx_ofs = idx_thread
            while idx_ofs < (basis_radius)**3:
                        ofs_u = (idx_ofs % basis_radius) - 2 * basis_scale
                        ofs_v = ((idx_ofs // basis_radius) % basis_radius) - 2 * basis_scale
                        ofs_w = (idx_ofs // basis_radius**2) - 2 * basis_scale

                        u_err = u - (int(u) + ofs_u + 0.5)
                        v_err = v - (int(v) + ofs_v + 0.5)
                        w_err = w - (int(w) + ofs_w + 0.5)

                        wu = compute_weight_device(u_err / basis_scale) / basis_scale
                        wv = compute_weight_device(v_err / basis_scale) / basis_scale
                        ww = compute_weight_device(w_err / basis_scale) / basis_scale

                        if (wu == 0) or (wv == 0) or (ww == 0):
                            pass
                        else:
                            val = grid[
                                clip_device(int(w) + ofs_w, 0, grid_resolution-1), 
                                clip_device(int(v) + ofs_v, 0, grid_resolution-1), 
                                clip_device(int(u) + ofs_u, 0, grid_resolution-1)
                            ]
                            sdf_value_tmp[idx_thread] += wu * wv * ww * val

                            grad_u = compute_grad_weight_device(u_err / basis_scale) / (basis_scale**2)
                            grad_v = compute_grad_weight_device(v_err / basis_scale) / (basis_scale**2)
                            grad_w = compute_grad_weight_device(w_err / basis_scale) / (basis_scale**2)

                            c = 0.5 * grid_resolution / grid_range

                            sdf_grad_tmp[idx_thread][0] += wv * ww * val * grad_u * c
                            sdf_grad_tmp[idx_thread][1] += wu * ww * val * grad_v * c
                            sdf_grad_tmp[idx_thread][2] += wu * wv * val * grad_w * c

                            grad2_u = compute_grad_grad_weight_device(u_err / basis_scale) / (basis_scale**3)
                            grad2_v = compute_grad_grad_weight_device(v_err / basis_scale) / (basis_scale**3)
                            grad2_w = compute_grad_grad_weight_device(w_err / basis_scale) / (basis_scale**3)

                            sdf_hessian_tmp[idx_thread][0] += grad2_u * wv * ww * val * c**2
                            sdf_hessian_tmp[idx_thread][1] += grad2_v * wu * ww * val * c**2
                            sdf_hessian_tmp[idx_thread][2] += grad2_w * wu * wv * val * c**2

                            sdf_hessian_tmp[idx_thread][3] += grad_u * grad_v * ww * val * c**2
                            sdf_hessian_tmp[idx_thread][4] += grad_v * grad_w * wu * val * c**2
                            sdf_hessian_tmp[idx_thread][5] += grad_u * grad_w * wv * val * c**2

                        idx_ofs += blockdim

            cuda.syncthreads()
            num_threads_valid = blockdim
            while num_threads_valid > 1:
                curr_ofs = ((num_threads_valid + 1) // 2)
                if idx_thread < curr_ofs:
                    if (idx_thread + curr_ofs) < num_threads_valid:
                        j = idx_thread + curr_ofs
                        sdf_value_tmp[idx_thread] += sdf_value_tmp[j]
                        sdf_grad_tmp[idx_thread,0] += sdf_grad_tmp[j,0]
                        sdf_grad_tmp[idx_thread,1] += sdf_grad_tmp[j,1]
                        sdf_grad_tmp[idx_thread,2] += sdf_grad_tmp[j,2]
                        sdf_hessian_tmp[idx_thread,0] += sdf_hessian_tmp[j,0]
                        sdf_hessian_tmp[idx_thread,1] += sdf_hessian_tmp[j,1]
                        sdf_hessian_tmp[idx_thread,2] += sdf_hessian_tmp[j,2]
                        sdf_hessian_tmp[idx_thread,3] += sdf_hessian_tmp[j,3]
                        sdf_hessian_tmp[idx_thread,4] += sdf_hessian_tmp[j,4]
                        sdf_hessian_tmp[idx_thread,5] += sdf_hessian_tmp[j,5]
                num_threads_valid = curr_ofs
                cuda.syncthreads()

            cuda.syncthreads()
            sdf_value = sdf_value_tmp[0]
            sdf_grad = (sdf_grad_tmp[0,0], sdf_grad_tmp[0,1], sdf_grad_tmp[0,2])
            sdf_hessian = (
                (sdf_hessian_tmp[0,0], sdf_hessian_tmp[0,3], sdf_hessian_tmp[0,5]),
                (sdf_hessian_tmp[0,3], sdf_hessian_tmp[0,1], sdf_hessian_tmp[0,4]),
                (sdf_hessian_tmp[0,5], sdf_hessian_tmp[0,4], sdf_hessian_tmp[0,2]),
            )

            if idx_thread == 0:
                dfdr_t = (
                    sdf_grad[0] * dxdr[0][0] + sdf_grad[1] * dxdr[1][0] + sdf_grad[2] * dxdr[2][0],
                    sdf_grad[0] * dxdr[0][1] + sdf_grad[1] * dxdr[1][1] + sdf_grad[2] * dxdr[2][1],
                    sdf_grad[0] * dxdr[0][2] + sdf_grad[1] * dxdr[1][2] + sdf_grad[2] * dxdr[2][2],
                )
                d2fdrdx = matmul333_device(sdf_hessian, dxdr)

                alpha = 0.05
                sdf_grad_norm = math.sqrt(sdf_grad[0]**2 + sdf_grad[1]**2 + sdf_grad[2]**2)
                sdf_normal = normalize_device(sdf_grad)
                nDr = sdf_normal[0] * r[0] + sdf_normal[1] * r[1] + sdf_normal[2] * r[2]
                weight_edge_denom = 1e-6 + abs(sdf_value) + alpha * nDr**2
                weight_edge = 1 / weight_edge_denom**3

                ImNNt = (
                    (1-sdf_normal[0]*sdf_normal[0], 0-sdf_normal[0]*sdf_normal[1], 0-sdf_normal[0]*sdf_normal[2]),
                    (0-sdf_normal[1]*sdf_normal[0], 1-sdf_normal[1]*sdf_normal[1], 0-sdf_normal[1]*sdf_normal[2]),
                    (0-sdf_normal[2]*sdf_normal[0], 0-sdf_normal[2]*sdf_normal[1], 1-sdf_normal[2]*sdf_normal[2])
                )
                dndr = scale_mat33_device(
                    1 / (sdf_grad_norm + 1e-9),
                    matmul333_device(
                        matmul333_device(ImNNt, sdf_hessian),
                        dxdr
                    )
                )
                grad_weight_edge_denom = scale_vector_device(
                    sgn_device(sdf_value), 
                    dfdr_t
                )
                grad_weight_edge_denom = add_vector_device(
                    grad_weight_edge_denom,
                    scale_vector_device(
                        alpha * 2 * nDr,
                        mul_vec_mat_device(r, dndr)
                    )
                )
                grad_weight_edge_denom = add_vector_device(
                    grad_weight_edge_denom,
                    scale_vector_device(
                        alpha * 2 * nDr,
                        r
                    )
                )
                grad_weight_edge = scale_vector_device(-3 / weight_edge_denom**4, grad_weight_edge_denom)

                sdf_diff = abs(prev_sdf_value) - abs(sdf_value)
                weight_dist_denom = min(0.05, max(abs(sdf_value), 1e-20))
                sum_weight_dist += max(sdf_diff / weight_dist_denom, 1)
                weight_dist = 1#min(sum_weight_dist, 1)

                bbox_dist = max(0, grid_range - max(abs(x[0]), abs(x[1]), abs(x[2])))
                weight_bbox = min(bbox_dist / 0.01, 1)
                grad_weight_bbox = (0.0, 0.0, 0.0)
                if (weight_bbox < 1.0) and (bbox_dist > 0.0):
                    if (abs(x[0]) >= abs(x[1])) and (abs(x[0]) >= abs(x[2])):
                        grad_weight_bbox = add_vector_device(
                            grad_weight_bbox, 
                            scale_vector_device(-sgn_device(x[0]) / 0.01, dxdr[0])
                        )
                    if (abs(x[1]) >= abs(x[0])) and (abs(x[1]) >= abs(x[2])):
                        grad_weight_bbox = add_vector_device(
                            grad_weight_bbox, 
                            scale_vector_device(-sgn_device(x[1]) / 0.01, dxdr[1])
                        )
                    if (abs(x[2]) >= abs(x[0])) and (abs(x[2]) >= abs(x[1])):
                        grad_weight_bbox = add_vector_device(
                            grad_weight_bbox, 
                            scale_vector_device(-sgn_device(x[2]) / 0.01, dxdr[2])
                        )

                weight = weight_edge * weight_bbox * weight_dist
                grad_weight = scale_vector_device(weight_bbox * weight_dist, grad_weight_edge)
                grad_weight = add_vector_device(
                    grad_weight, 
                    scale_vector_device(weight_edge * weight_dist, grad_weight_bbox)
                )

                T = 0.5 * (max(sdf_value,0) + max(prev_sdf_value,0))
                dTdr_t = (0, 0, 0)
                if sdf_value > 0:
                    dTdr_t = add_vector_device(
                        dTdr_t,
                        scale_vector_device(0.5, dfdr_t),
                    )
                if prev_sdf_value > 0:
                    dTdr_t = add_vector_device(
                        dTdr_t,
                        scale_vector_device(0.5, prev_dfdr_t),
                    )

                #dTdr_t = add_vector_device(
                #    scale_vector_device(0.5 * sgn_device(sdf_value), dfdr_t),
                #    scale_vector_device(0.5 * sgn_device(prev_sdf_value), prev_dfdr_t)
                #)

                #T = 1
                #dTdr_t = (0, 0, 0)

                sum_w += weight * T
                sum_w_t += weight * t * T
                sum_grad_w = add_vector_device(
                    sum_grad_w,
                    scale_vector_device(T, grad_weight)
                )
                sum_grad_w = add_vector_device(
                    sum_grad_w,
                    scale_vector_device(weight, dTdr_t)
                )
                sum_grad_w_t = add_vector_device(
                    sum_grad_w_t,
                    scale_vector_device(t * T, grad_weight)
                )
                sum_grad_w_t = add_vector_device(
                    sum_grad_w_t,
                    scale_vector_device(t * weight, dTdr_t)
                )
                sum_grad_w_t = add_vector_device(
                    sum_grad_w_t,
                    scale_vector_device(weight * T, dtdr_t)
                )
                prev_sdf_value = sdf_value
                prev_sdf_grad = sdf_grad
                prev_t = t
                prev_dfdr_t = dfdr_t

            if sdf_value < threshold:
                break

            # update x0
            if idx_thread == 0:
                c = sdf_value#sgn_device(sdf_value) * clip_device(abs(0.99*sdf_value), 0.25 * threshold, 1e3)
                x[0] += c * r[0]
                x[1] += c * r[1]
                x[2] += c * r[2]

                dxdr = add_mat33_device(
                    dxdr,
                    ((c,0,0),(0,c,0),(0,0,c))
                )
                if True:#abs(0.99*sdf_value) > (0.25 * threshold):
                    dxdr = add_mat33_device(
                        dxdr,
                        (
                            (r[0]*dfdr_t[0], r[0]*dfdr_t[1], r[0]*dfdr_t[2]),
                            (r[1]*dfdr_t[0], r[1]*dfdr_t[1], r[1]*dfdr_t[2]),
                            (r[2]*dfdr_t[0], r[2]*dfdr_t[1], r[2]*dfdr_t[2])
                        )
                    )

                t += c
                dtdr_t = add_vector_device(dtdr_t, dfdr_t)

            cuda.syncthreads()

            if (abs(x[0]) > grid_range) or (abs(x[1]) > grid_range) or (abs(x[2]) > grid_range):
                break

            if (x[0]**2 + x[1]**2 + x[2]**2) > 1.0:
                break

            t_ = (x[0] - ray_origin[idx_sample,0]) * r[0] + (x[1] - ray_origin[idx_sample,1]) * r[1] + (x[2] - ray_origin[idx_sample,2]) * r[2]
            if (t_ > max_t):
                break

    if idx_thread == 0:
        t_surf[idx_sample,0] = (x[0] - ray_origin[idx_sample,0]) * r[0] + (x[1] - ray_origin[idx_sample,1]) * r[1] + (x[2] - ray_origin[idx_sample,2]) * r[2]
        t_eval[idx_sample,0] = sum_w_t / max(1e-12, sum_w)

        grad_t_eval[idx_sample,0] = sum_grad_w_t[0] / max(1e-6, sum_w) - sum_w_t / max(1e-6, sum_w**2) * sum_grad_w[0]
        grad_t_eval[idx_sample,1] = sum_grad_w_t[1] / max(1e-6, sum_w) - sum_w_t / max(1e-6, sum_w**2) * sum_grad_w[1]
        grad_t_eval[idx_sample,2] = sum_grad_w_t[2] / max(1e-6, sum_w) - sum_w_t / max(1e-6, sum_w**2) * sum_grad_w[2]

        if sum_w < 1e-7:
            t_eval[idx_sample,0] = max_t
            grad_t_eval[idx_sample,0] = 0
            grad_t_eval[idx_sample,1] = 0
            grad_t_eval[idx_sample,2] = 0

        if sdf_value < threshold:
            is_converged[idx_sample] = 1.0

@cuda.jit
def minimum_sdf_kernel(
    # inputs
    grid, x, ray_direction, t_min, t_max,
    # parameters
    grid_range
):
    grid_resolution = grid.shape[0]

    idx_sample = cuda.blockIdx.x

    idx_thread = cuda.threadIdx.x
    blockdim = cuda.blockDim.x

    r = ray_direction[idx_sample]
    t_min_ = t_min[idx_sample]
    t_max_ = t_max[idx_sample]

    sdf_tmp = cuda.shared.array(shape=(128), dtype=numba.float32)
    num_points = 128

    if t_max_ > t_min_:
        idx_pt = idx_thread
        while idx_pt < num_points:
            t = t_min_ + (t_max_ - t_min_) * idx_pt / num_points
            x0 = (
                x[idx_sample,0] + t * r[0],
                x[idx_sample,1] + t * r[1],
                x[idx_sample,2] + t * r[2]
            )

            # query SDF value
            u = 0.5 * (x0[0] / grid_range + 1) * grid_resolution
            v = 0.5 * (x0[1] / grid_range + 1) * grid_resolution
            w = 0.5 * (x0[2] / grid_range + 1) * grid_resolution

            u = clip_device(u, 1, grid_resolution - 1 - 1e-4)
            v = clip_device(v, 1, grid_resolution - 1 - 1e-4)
            w = clip_device(w, 1, grid_resolution - 1 - 1e-4)

            sdf_value = 0.0
            for ofs_w in range(-2,3):#[-2,-1,0,1,2]:
                for ofs_v in range(-2,3):#[-2,-1,0,1,2]:
                    for ofs_u in range(-2,3):#[-2,-1,0,1,2]:
                        val = grid[
                            clip_device(int(w) + ofs_w, 0, grid_resolution-1), 
                            clip_device(int(v) + ofs_v, 0, grid_resolution-1), 
                            clip_device(int(u) + ofs_u, 0, grid_resolution-1)
                        ]
                        wu = compute_weight_device(u - (int(u) + ofs_u + 0.5))
                        wv = compute_weight_device(v - (int(v) + ofs_v + 0.5))
                        ww = compute_weight_device(w - (int(w) + ofs_w + 0.5))

                        sdf_value += wu * wv * ww * val

            sdf_tmp[idx_pt] = sdf_value
            
            idx_pt += blockdim

    cuda.syncthreads()

    if (idx_thread == 0) and (t_max_ > t_min_):
        t_min_new = t_min_
        t_max_new = t_max_
        i = 0
        min_sdf = sdf_tmp[0]
        while i < (num_points - 1):
            # minimal
            #if (sdf_tmp[i] < 0) and (sdf_tmp[i] < sdf_tmp[i+1]):
            #    t_min_new = t_min_ + (t_max_ - t_min_) * i / num_points
            #    t_max_new = t_min_ + (t_max_ - t_min_) * (i+1) / num_points
            #    break
            if sdf_tmp[i] < min_sdf:
                min_sdf = sdf_tmp[i]
                t_min_new = t_min_ + (t_max_ - t_min_) * i / num_points
                t_max_new = t_min_ + (t_max_ - t_min_) * (i + 1) / num_points

            i += 1

        t_min[idx_sample] = t_min_new
        t_max[idx_sample] = t_max_new

@cuda.jit
def nearest_normal_pixels_kernel(
    # inputs
    depth_map, normal_map, mask, query_normals,
    # params
    block_size,
    # result
    result_pixels, result_scores
):
    image_height, image_width = normal_map.shape[-2:]
    idx_batch = cuda.blockIdx.x
    idx_query = cuda.blockIdx.y

    idx_thread = cuda.threadIdx.x
    blockdim = cuda.blockDim.x

    num_blocks_h = (image_height - 1) // block_size + 1
    num_blocks_w = (image_width - 1) // block_size + 1
    num_blocks = num_blocks_h * num_blocks_w

    query_n = query_normals[idx_batch,idx_query,:]
    r2 = query_n[0]**2 + query_n[1]**2 + query_n[2]**2

    if r2 > (0.5**2):
        idx_block = idx_thread

        while idx_block < num_blocks:
            idx_block_v = idx_block // num_blocks_w
            idx_block_u = idx_block % num_blocks_w
            u_ofs = idx_block_u * block_size
            v_ofs = idx_block_v * block_size

            argmax_u = u_ofs
            argmax_v = v_ofs
            max_score = -1e6
            for idx_pixel in range(block_size**2):
                u = u_ofs + idx_pixel % block_size 
                v = v_ofs + idx_pixel // block_size 
                if (u >= image_width) or (v >= image_height):
                    continue
                pixel_m = mask[idx_batch,0,v,u]            
                if pixel_m > 0.0:
                    pixel_n = normal_map[idx_batch,:,v,u]
                    pixel_score = pixel_n[0] * query_n[0] + pixel_n[1] * query_n[1] + pixel_n[2] * query_n[2]

                    if pixel_score > max_score:
                        argmax_u = u
                        argmax_v = v
                        max_score = pixel_score

            depth = depth_map[idx_batch,0,argmax_v,argmax_u]
            result_scores[idx_batch,idx_query,idx_block] = max_score
            result_pixels[idx_batch,idx_query,idx_block,0] = (argmax_u + 0.5) * depth
            result_pixels[idx_batch,idx_query,idx_block,1] = (argmax_v + 0.5) * depth
            result_pixels[idx_batch,idx_query,idx_block,2] = depth


            idx_block += blockdim

def get_gaussian_kernel_3d(sigma=2.0):
    size = int(4*sigma)
    if (size % 2) == 0:
        size += 1
    center = 0.5 * size
    w,v,u = torch.meshgrid(
        torch.arange(size),
        torch.arange(size),
        torch.arange(size),
    )
    grid = torch.stack([u,v,w], dim=-1) + 0.5
    d2 = torch.sum((grid - center)**2, dim=-1)
    kernel = torch.exp(-0.5 * d2)
    return (kernel / torch.sum(kernel))[None,None]

class SDFGrid(nn.Module):
    def __init__(self, resolution=128, range=1.0, initial_radius=1):
        super(SDFGrid, self).__init__()

        self.range = range
        self.resolution = resolution
        self.threshold = 1e-4

        self.grid = nn.Parameter(torch.zeros(
            (resolution, resolution, resolution), 
            dtype=torch.float32,
            requires_grad=True
        ))

        self.grid_sample = b_spline_interpolation.apply
        self.grid_sample_sdf_value = compute_sdf_value.apply
        self.grid_sample_sdf_hessian = compute_sdf_hessian.apply

        w,v,u = torch.meshgrid(torch.arange(resolution),torch.arange(resolution),torch.arange(resolution))
        x = range * (2 * (u.float() + 0.5) / resolution - 1)
        y = range * (2 * (v.float() + 0.5) / resolution - 1)
        z = range * (2 * (w.float() + 0.5) / resolution - 1)
        r2 = x**2 + y**2 + z**2
        with torch.no_grad():
            self.grid[:] = torch.sqrt(r2) - initial_radius
            self.redistance()

        self.img2rmap = SoftImageToReflectanceMap()

    def resize_grid(self, resolution):
        self.resolution = resolution
        with torch.no_grad():
            new_grid = F.interpolate(self.grid[None,None], resolution, mode='trilinear')[0,0]
            new_grid.requires_grad=True
        self.grid = nn.Parameter(new_grid)

        self.validate()


    def redistance(self):
        with torch.no_grad():
            if not self.grid.is_cuda:
                grid = drjit.cuda.TensorXf(self.grid.numpy())
            else:
                grid = drjit.cuda.TensorXf(self.grid.detach())
            
            self.grid[:] = torch.tensor(fastsweep.redistance(grid / (2 * self.range))) * (2 * self.range)

    def validate(self):
        with torch.no_grad():
            grid_coords = self.get_grid_coords()
            grid_radius2 = torch.sum(grid_coords**2, dim=-1)
            self.grid[grid_radius2 > 1] = 100.0
            self.redistance()

    def compute_weight(self, t):
        #return torch.clamp(1-torch.abs(t),0,1)
        t = 2 - torch.clamp(torch.abs(t), 0, 2)
        b1 = 1 / 6.0 * t**3
        m1 = (t < 1.0).float()
        u = t - 1.0
        b2 = 1 / 6.0 * (-3 * u**3 + 3 * u**2 + 3 * u + 1)
        m2 = (t >= 1.0).float()
        return b1 * m1 + b2 * m2

    def compute_grad_weight(self, t):
        #return torch.sgn(t) * (torch.abs(t) < 1.0).float()
        sgn = -torch.sgn(t)
        t = 2 - torch.clamp(torch.abs(t), 0, 2)
        gb1 = 1 / 2.0 * t**2
        m1 = (t < 1.0).float()
        u = t - 1.0
        gb2 = 1 / 2.0 * (-3.0 * u**2 + 2.0 * u + 1.0)
        m2 = (t >= 1.0).float()

        return sgn * (gb1 * m1 + gb2 * m2)

    def get_grid_coords(self, upsampling_ratio=1, device=None):
        upsampling_ratio = int(upsampling_ratio)
        w,v,u = torch.meshgrid(
            torch.arange(upsampling_ratio * self.grid.size(0)),
            torch.arange(upsampling_ratio * self.grid.size(1)),
            torch.arange(upsampling_ratio * self.grid.size(2)),
        )

        x = (2 * (u + 0.5) / (upsampling_ratio * self.resolution) - 1) * self.range
        y = (2 * (v + 0.5) / (upsampling_ratio * self.resolution) - 1) * self.range
        z = (2 * (w + 0.5) / (upsampling_ratio * self.resolution) - 1) * self.range

        if device is None:
            device = self.grid.device

        return torch.stack([x,y,z], dim=-1).to(device)


    # x: N * 3
    # is_valid: N
    # out: N * 1, N * 3
    def forward(self, x, backward_to_grid=True, mask=None):
        if not (mask is None):
            out_val = torch.zeros_like(x[...,:1], dtype=x.dtype, device=x.device)
            out_grad = torch.zeros_like(x, dtype=x.dtype, device=x.device)
            val, grad = self.forward(x[mask], backward_to_grid=backward_to_grid)
            out_val[mask] = val
            out_grad[mask] = grad
            return out_val, out_grad

        grid = self.grid
        if not backward_to_grid:
            grid = grid.detach()

        if x.is_cuda:
            return self.grid_sample(grid, x, torch.tensor(self.range))

        u = 0.5 * (x[:,0] / self.range + 1) * self.resolution
        v = 0.5 * (x[:,1] / self.range + 1) * self.resolution
        w = 0.5 * (x[:,2] / self.range + 1) * self.resolution

        u = torch.clamp(u, 1, self.resolution - 1 - 1e-4)
        v = torch.clamp(v, 1, self.resolution - 1 - 1e-4)
        w = torch.clamp(w, 1, self.resolution - 1 - 1e-4)


        sdf_value = torch.zeros_like(u, dtype=x.dtype, device=x.device)
        sdf_grad = torch.zeros_like(x, dtype=x.dtype, device=x.device)
        for ofs_w in [-2,-1,0,1,2]:
            for ofs_v in [-2,-1,0,1,2]:
                for ofs_u in [-2,-1,0,1,2]:
                    val = grid[
                        torch.clamp(w.long() + ofs_w, 0, self.resolution-1), 
                        torch.clamp(v.long() + ofs_v, 0, self.resolution-1), 
                        torch.clamp(u.long() + ofs_u, 0, self.resolution-1)
                    ]
                    wu = self.compute_weight(u - (u.int() + ofs_u + 0.5))
                    wv = self.compute_weight(v - (v.int() + ofs_v + 0.5))
                    ww = self.compute_weight(w - (w.int() + ofs_w + 0.5))
                    sdf_value = sdf_value + wu * wv * ww * val

                    grad_u = self.compute_grad_weight(u - (u.int() + ofs_u + 0.5))
                    grad_v = self.compute_grad_weight(v - (v.int() + ofs_v + 0.5))
                    grad_w = self.compute_grad_weight(w - (w.int() + ofs_w + 0.5))

                    c = 0.5 * self.resolution / self.range
                    sdf_grad[:,0] = sdf_grad[:,0] + wv * ww * val * grad_u * c
                    sdf_grad[:,1] = sdf_grad[:,1] + wu * ww * val * grad_v * c
                    sdf_grad[:,2] = sdf_grad[:,2] + wu * wv * val * grad_w * c

        return sdf_value[:,None], sdf_grad

    # x: N * 3
    # out: N * 1, N * 3
    def compute_sdf_value(self, x, backward_to_grid=True):
        grid = self.grid
        if not backward_to_grid:
            grid = grid.detach()

        if x.is_cuda:
            return self.grid_sample_sdf_value(grid, x, torch.tensor(self.range))

        u = 0.5 * (x[:,0] / self.range + 1) * self.resolution
        v = 0.5 * (x[:,1] / self.range + 1) * self.resolution
        w = 0.5 * (x[:,2] / self.range + 1) * self.resolution

        u = torch.clamp(u, 1, self.resolution - 1 - 1e-4)
        v = torch.clamp(v, 1, self.resolution - 1 - 1e-4)
        w = torch.clamp(w, 1, self.resolution - 1 - 1e-4)


        sdf_value = torch.zeros_like(u, dtype=x.dtype, device=x.device)
        for ofs_w in [-2,-1,0,1,2]:
            for ofs_v in [-2,-1,0,1,2]:
                for ofs_u in [-2,-1,0,1,2]:
                    val = grid[
                        torch.clamp(w.long() + ofs_w, 0, self.resolution-1), 
                        torch.clamp(v.long() + ofs_v, 0, self.resolution-1), 
                        torch.clamp(u.long() + ofs_u, 0, self.resolution-1)
                    ]
                    wu = self.compute_weight(u - (u.int() + ofs_u + 0.5))
                    wv = self.compute_weight(v - (v.int() + ofs_v + 0.5))
                    ww = self.compute_weight(w - (w.int() + ofs_w + 0.5))
                    sdf_value = sdf_value + wu * wv * ww * val

        return sdf_value[:,None]

    def compute_hessian_matrix(self, x, backward_to_grid=True):
        grid = self.grid

        if not backward_to_grid:
            grid = grid.detach()

        return self.grid_sample_sdf_hessian(grid, x, torch.tensor(self.range))

    # dn = W @ dx
    def compute_shape_operator(self, x):
        Hf = self.compute_hessian_matrix(x)
        grad_sdf = self.forward(x)[1]
        grad_sdf_norm = torch.sqrt(torch.sum(grad_sdf**2, dim=-1, keepdim=True) + 1e-4)
        N = (grad_sdf / grad_sdf_norm)[...,:,None]
        I = torch.eye(3, dtype=x.dtype, device=x.device)[None]

        S = (I - N @ N.transpose(-1,-2)) @ Hf / grad_sdf_norm[...,None]

        return S

    def compute_laplacian(self, x, backward_to_grid=True):
        Hf = self.compute_hessian_matrix(x, backward_to_grid=backward_to_grid)
        L = Hf[:,0,0] + Hf[:,1,1] + Hf[:,2,2]
        return L[:,None]

    def compute_mean_squared_surface_laplacian(self):
        x = self.get_grid_coords().view(-1,3)
        x = x[torch.abs(self.grid.view(-1)) < (1.74 * 0.5 * self.range / self.resolution)]
        x = x.repeat(9,1)
        if True:
            jitter_ofs = (torch.rand_like(x, device=x.device) - 0.5) * 0.5 * self.range / self.resolution
            x = x + jitter_ofs
        with torch.no_grad():
            for _ in range(2):
                sdf, n = self.forward(x)
                x -= sdf * n
        Hf = self.compute_hessian_matrix(x)
        L = Hf[:,0,0] + Hf[:,1,1] + Hf[:,2,2]
        return torch.mean(L**2)

    def compute_curvatures(self, x, backward_to_grid=True):        
        Hf = self.compute_hessian_matrix(x, backward_to_grid)
        grad_sdf = self.forward(x, backward_to_grid)[1]

        grad_sdf_norm = torch.sqrt(torch.sum(grad_sdf**2, dim=-1, keepdim=True) + 1e-9)
        N = (grad_sdf / grad_sdf_norm)[...,:,None]
        I = torch.eye(3, dtype=x.dtype, device=x.device)[None]

        S = (I - N @ N.transpose(-1,-2)) @ Hf / grad_sdf_norm[...,None]

        mc = 0.5 * (S[:,0,0] + S[:,1,1] + S[:,2,2])[...,None] # mean curvature

        M = torch.cat([Hf, grad_sdf[...,:,None]], dim=-1)
        Nt0 = torch.cat([grad_sdf[...,None,:], torch.zeros_like(N[...,0:1,0:1], dtype=N.dtype, device=N.device)], dim=-1)
        M = torch.cat([M, Nt0], dim=-2)
        gc = (-torch.det(M) / grad_sdf_norm[...,0]**4)[...,None]

        dif_c = torch.sqrt(torch.clamp(mc**2 - gc, 0, None) + 1e-3)
        k1 = mc - dif_c
        k2 = mc + dif_c

        return mc, gc, k1, k2

    def compute_approximate_area(self, eps=0.1):
        def dirac_delta(x):
            c = 1 / (eps * np.sqrt(np.pi))
            return c * torch.exp(-(x/eps)**2)
        return torch.mean(dirac_delta(self.grid)) * (2 * self.range)**3


    def sample_random_points(self, num_samples=65535):
        x = torch.rand((num_samples, 3), dtype=self.grid.dtype, device=self.grid.device)
        x = 2 * x - 1
        x = x * self.range
        return x

    def find_nearest_surface_points(self, x, max_itr=20, th=1e-5):
        x = x.clone()
        mask = torch.ones_like(x[:,0], dtype=torch.bool, device=x.device)
        for _ in range(max_itr):
            with torch.no_grad():
                sdf_vals, sdf_normals = self.forward(x[mask])
                sdf_normals = sdf_normals / torch.clamp(torch.sum(sdf_normals**2, dim=-1, keepdim=True), 0.4, None)
                x[mask] -= sdf_vals * sdf_normals

                mask[mask.clone()] *= abs(sdf_vals.view(-1)) > th
                #print(torch.max(torch.abs(sdf_vals)))
                if torch.max(torch.abs(sdf_vals)) < th:
                    break
        return x

    def extract_mesh(self, resolution=128, refine_verts=True, use_random_offset=False, use_random_rotation=True, use_remeshing=True, verbose=False):
        device = self.grid.device
        u,v,w = torch.meshgrid(
            torch.arange(resolution),
            torch.arange(resolution),
            torch.arange(resolution),
        )

        x = (2 * (u + 0.5) / resolution - 1) * self.range
        y = (2 * (v + 0.5) / resolution - 1) * self.range
        z = (2 * (w + 0.5) / resolution - 1) * self.range

        grid_coords = torch.stack([x,y,z], dim=-1).view(-1,3)

        ofs = torch.zeros((1))
        if use_random_offset:
            ofs = 2 / resolution * self.range * (torch.rand((1)) - 0.5)
        grid_coords = grid_coords + ofs

        rot = torch.eye(3)
        if use_random_rotation:
            rot = torch.from_numpy(scipy_rotation.random().as_matrix().astype(np.float32))
        grid_coords = grid_coords @ rot

        grid_coords_n = grid_coords.view(resolution,resolution,resolution,3) / self.range
        grid = F.grid_sample(
            self.grid.detach()[None,None], 
            grid_coords_n[None].to(device), 
            mode="bilinear", 
            padding_mode="border"
        )[0,0].cpu()

        grid_mask = torch.abs(grid) < 1e-2
        grid_coords_selected = grid_coords[grid_mask.view(-1)]

        with torch.no_grad():
            chunk_size = 8192
            sdf_vals = []
            if False:#verbose:
                bar = tqdm(range((len(grid_coords_selected) - 1) // chunk_size + 1))
                bar.set_description('SDF Eval for Mesh Recon')
            else:
                bar = range((len(grid_coords_selected) - 1) // chunk_size + 1)
            for i in bar:
                x_chunk = grid_coords_selected[chunk_size*i:chunk_size*(i+1)].to(device)
                sdf_vals.append(self.forward(x_chunk)[0].cpu())
            grid[grid_mask] = torch.cat(sdf_vals, dim=0).view(-1)

        grid = grid.cpu().numpy()

        verts, faces, _, _ = skimage.measure.marching_cubes(
            grid, level=0.0, spacing=[1]*3, method='lorensen'
        )

        if use_remeshing:
            for _ in range(1):
                mesh = pymesh.form_mesh(verts, faces)
                #mesh = pymesh.split_long_edges(mesh, 0.004)[0]
                mesh = pymesh.subdivide(mesh, order=1, method='loop')
                mesh = pymesh.collapse_short_edges(mesh, abs_threshold=0.8, preserve_feature=True)[0]
                verts = mesh.vertices.astype(np.float32)
                faces = mesh.faces.astype(np.int32)

        verts = torch.from_numpy(verts.copy()).to(device)
        faces = torch.from_numpy(faces.copy()).to(device)

        verts = self.range * (verts * 2 / (resolution - 1) - 1) + ofs.to(device)
        verts = verts @ rot.to(device)

        if refine_verts:
            chunk_size = 8192
            with torch.no_grad():
                for _ in range(20):
                    sdf, n = self.forward(verts)
                    n = n / torch.clamp(torch.sum(n**2, dim=-1, keepdim=True), 0.4, None)
                    verts -= sdf * n
                    #print(torch.max(torch.abs(sdf)))
                    if torch.max(torch.abs(sdf)) < 1e-5:
                        break

        verts, faces = remove_duplicates(verts, faces)[:2]
        verts = verts.float()
        faces = faces.int()
        mask_valid_face = (faces[:,0] != faces[:,1]) * (faces[:,1] != faces[:,2]) * (faces[:,2] != faces[:,0])
        if torch.any(mask_valid_face == False):
            print('Warning: invalid faces are removed')
            return self.extract_mesh(
                resolution, 
                refine_verts, 
                use_random_offset, 
                use_random_rotation, 
                use_remeshing, 
                verbose
            )
            #faces = faces[mask_valid_face]

        return verts, faces

    def sample_surface_points(self, subpixel_resolution=8, requires_grad=False):
        import trimesh
        #import matplotlib.pyplot as plt
        grid = F.interpolate(self.grid[None,None].detach(), scale_factor=1)[0,0].cpu().numpy()
        #plt.imshow(grid[grid.shape[0]//2])
        #plt.show()
        recon_mesh_tri = trimesh.voxel.ops.matrix_to_marching_cubes(grid <= 0, pitch=1.0)
        x = self.range * (torch.tensor(recon_mesh_tri.vertices).float().to(self.grid.device) * 2 / grid.shape[-1] - 1)
        x = torch.stack([x[...,2], x[...,1], x[...,0]], dim=-1)

        sub_reso = subpixel_resolution
        sub_w, sub_v, sub_u = torch.meshgrid(
            torch.arange(sub_reso),
            torch.arange(sub_reso),
            torch.arange(sub_reso)
        )
        sub_grid = torch.stack([sub_u, sub_v, sub_w], dim=-1).to(self.grid.device).float()
        if True:
            jitter_ofs = (torch.rand_like(sub_grid, device=sub_grid.device) - 0.5)
            sub_grid = sub_grid + jitter_ofs
        sub_grid = 2 * (sub_grid + 0.5) / sub_reso - 1.0
        sub_grid = sub_grid * 2 * self.range / self.resolution
        sub_ofs = sub_grid.view(-1,3)

        x = (x[:,None,:] + sub_ofs[None,:,:]).view(-1,3)

        #x = self.get_grid_coords().view(-1,3)
        #x = x[torch.abs(self.grid.view(-1)) < (8 * 1.74 * 0.5 * self.range / self.resolution)]
        #x = x[:,None,:].repeat(1,16,1).view(-1,3)
        #if True:
        #    jitter_ofs = (torch.rand_like(x, device=x.device) - 0.5) * 0.5 * self.range / self.resolution
        #    x = x + jitter_ofs

        with torch.no_grad():
            sdf, n = self.forward(x)
            x = x[torch.abs(sdf[:,0]) < 1e-3]
            for _ in range(20):
                sdf, n = self.forward(x)
                n = n / torch.clamp(torch.sum(n**2, dim=-1, keepdim=True), 0.4, None)
                x -= sdf * n
                #print(torch.max(torch.abs(sdf)))
                if torch.max(torch.abs(sdf)) < 1e-5:
                    break
        #print(x.size())
        if not requires_grad:
            return x
        
        f, grad_f = self.forward(x)
        n = grad_f.detach() / (torch.sum(grad_f.detach()**2, dim=-1, keepdim=True) + 1e-3)
        return x - f * n



        x = torch.rand((num_samples, 3), dtype=self.grid.dtype, device=self.grid.device)
        x = 2 * x - 1
        with torch.no_grad():
            f, grad_f = self.forward(x)
            x = x[f[:,0] < (2 * self.range / self.resolution)]
            for _ in range(4):
                sdf, n = self.forward(x)
                n = n / torch.sum(n**2, dim=-1, keepdim=True)
                x -= sdf * n
                print(torch.max(torch.abs(sdf)))
        return x


    def sample_surface_curvatures(self, backward_to_grid=True):
        #x = self.get_grid_coords().view(-1,3)
        #x = x[torch.abs(self.grid.view(-1)) < (8 * 1.74 * 0.5 * self.range / self.resolution)]
        #x = x[:,None,:].repeat(1,9,1).view(-1,3)
        #if True:
        #    jitter_ofs = (torch.rand_like(x, device=x.device) - 0.5) * 0.5 * self.range / self.resolution
        #    x = x + jitter_ofs

        #with torch.no_grad():
        #    sdf, n = self.forward(x)
        #    x = x[sdf[:,0] < (self.range / self.resolution)]
        #    #for _ in range(4):
        #    #    sdf, n = self.forward(x)
        #    #    n = n / torch.sum(n**2, dim=-1, keepdim=True)
        #    #    x -= sdf * n
        #    #    print(torch.max(torch.abs(sdf)))

        x = self.sample_surface_points().detach()
        #print(len(x), 'points are sampled')

        sdf, n = self.forward(x)
        n = n# / torch.clamp(torch.sum(n**2, dim=-1, keepdim=True), 0.4, None)
        x = x - n.detach() * sdf

        return self.compute_curvatures(x, backward_to_grid)

        Hf = self.compute_hessian_matrix(x)
        grad_sdf = self.forward(x)[1]

        #Hf = torch.mean(Hf.view(-1,9,3,3), dim=1)
        #grad_sdf = torch.mean(grad_sdf.view(-1,9,3), dim=1)

        grad_sdf_norm = torch.sqrt(torch.sum(grad_sdf**2, dim=-1, keepdim=True) + 1e-9)
        N = (grad_sdf / grad_sdf_norm)[...,:,None]
        I = torch.eye(3, dtype=x.dtype, device=x.device)[None]

        S = (I - N @ N.transpose(-1,-2)) @ Hf / grad_sdf_norm[...,None]

        n = self.forward(x)[1]
        mc = 0.5 * (S[:,0,0] + S[:,1,1] + S[:,2,2]) # mean curvature

        #trace_Hf = Hf[:,0,0] + Hf[:,1,1] + Hf[:,2,2]
        #mc = ((grad_sdf[...,None,:] @ Hf @ grad_sdf[...,:,None])[...,0,0] - grad_sdf_norm[...,0]**2 * trace_Hf) / (2 * grad_sdf_norm[...,0]**3)

        M = torch.cat([Hf, grad_sdf[...,:,None]], dim=-1)
        Nt0 = torch.cat([grad_sdf[...,None,:], torch.zeros_like(N[...,0:1,0:1], dtype=N.dtype, device=N.device)], dim=-1)
        M = torch.cat([M, Nt0], dim=-2)
        gc = -torch.det(M) / grad_sdf_norm[...,0]**4

        return mc, gc
        print(mc[0], gc[0])
        exit()

        U,s,Vh = torch.svd(S)
        print(S[0])
        print(U[0] @ torch.diag(s[0]) @ Vh[0].T)
        print(Vh[0].T @ n[0,:,None])
        eigvals, eigvecs = torch.linalg.eig(S[0])
        print()
        exit()
        print(torch.linalg.svd(S[0]))
        print(n[0])
        print(S[0] @ n[0,:,None])
        exit()
        return torch.mean(S)
    
    def save_mesh(self, out_file, resolution=256, scale_factor=1.0, offset=0.0):
        verts, faces = self.extract_mesh(resolution, refine_verts=True, verbose=True)
        mesh = trimesh.Trimesh(vertices=verts.cpu().numpy() * scale_factor + offset, faces=faces.cpu().numpy())
        mesh.export(out_file)

    # x: N * 3
    # is_valid: N
    # out: N * 1, N * 3
    def sample_grid_grad(self, x, mask=None):
        if not (mask is None):
            out_val = torch.zeros_like(x[...,:1], dtype=x.dtype, device=x.device)
            out_grad = torch.zeros_like(x, dtype=x.dtype, device=x.device)
            val, grad = self.sample_grid_grad(x[mask])
            out_val[mask] = val
            out_grad[mask] = grad
            return out_val, out_grad

        grid = self.grid.grad

        return self.grid_sample(grid, x, torch.tensor(self.range))[0]

    # alternative of sphere_tracing_kernel() for debugging
    # ray_origin     : N * 3
    # ray_direction : N * 3
    def sphere_tracing_kernel_pt(self, ray_origin, ray_direction, t, max_t):
        # assert |ray_direction| == 1
        ray_direction = ray_direction / torch.sqrt(torch.sum(ray_direction**2, dim=-1, keepdim=True) + 1e-6)

        ray_direction.requires_grad = True
        x = ray_origin + t * ray_direction
        autograd_should_be_disabled = False
        if x.requires_grad == False:
            torch.set_grad_enabled(True)
            x = ray_origin + t * ray_direction
            autograd_should_be_disabled = True

        prev_f = torch.zeros_like(x[...,0:1], device=x.device)
        sum_weight_dist = 0.0
        sum_w = 0.0
        sum_w_t = 0.0
        is_active = x[...,0] == x[...,0]
        for _ in range(100):
            f, grad_f = self.forward(x, backward_to_grid=False, mask=is_active)
            alpha = 0.05
            grad_f_norm = torch.sqrt(torch.sum(grad_f**2, dim=-1, keepdim=True) + 1e-6)
            sdf_normal = grad_f / grad_f_norm
            nDr = torch.sum(sdf_normal * ray_direction, dim=-1, keepdim=True)
            weight_edge_denom = 1e-6 + torch.abs(f) + alpha * nDr**2
            weight_edge = 1 / weight_edge_denom**3

            sdf_diff = torch.clamp(torch.abs(prev_f) - torch.abs(f), min=0.0)
            weight_dist_denom = torch.clamp(torch.abs(f) + 1e-6, max=0.05)
            sum_weight_dist = sum_weight_dist + torch.clamp(sdf_diff / weight_dist_denom, min=0, max=1)
            weight_dist = torch.clamp(sum_weight_dist, min=0, max=1)

            bbox_dist = torch.clamp(self.range - torch.max(torch.abs(x), dim=-1, keepdim=True)[0], min=0)
            weight_bbox = torch.clamp(bbox_dist / 0.01, max=1)

            weight = weight_edge * weight_bbox * weight_dist * is_active[...,None].float() + 1e-20

            dt = 0.5 * (torch.abs(f) + torch.abs(prev_f))

            sum_w += weight * dt
            sum_w_t += weight * t * dt

            is_active = is_active * ((f > self.threshold) * (t < max_t))[...,0]
            if not torch.any(is_active):
                break

            # sphere tracing
            x = x + torch.abs(f) * ray_direction * is_active[...,None].float()
            t = t + torch.abs(f) * is_active[...,None].float()

            prev_f = f

        t0 = torch.sum((x.detach() - ray_origin) * ray_direction.detach(), dim=-1, keepdim=True)
        t_eval = sum_w_t / sum_w

        d_output = torch.ones_like(
            t_eval, 
            requires_grad=False, 
            device=t_eval.device
        )
        dtdr = torch.autograd.grad(
            outputs=t_eval,
            inputs=ray_direction,
            grad_outputs=d_output,
            create_graph=False,
            retain_graph=False,
            only_inputs=True
        )[0]
        if autograd_should_be_disabled:
            torch.set_grad_enabled(False)

        is_converged = (f.detach() < self.threshold).float()[...,0]

        t_surf = t0.detach()
        t_eval = t_eval.detach()
        ray_direction = ray_direction.detach()

        return t_surf, t_eval, dtdr, is_converged

    # ray_origin     : N * 3
    # ray_direction : N * 3
    def sphere_tracing(
        self, 
        ray_origin, 
        ray_direction, 
        max_depth=100.0
    ):
        #if ray_origin.size(0) == 589824:
        #    self.sphere_tracing_pt(ray_origin, ray_direction, max_depth=100.0)
        # assert |ray_direction| == 1
        ray_direction = ray_direction / torch.sqrt(torch.sum(ray_direction**2, dim=-1, keepdim=True) + 1e-6)

        #cuda.synchronize()
        #torch.cuda.synchronize()
        #print('sphere tracing started')
        with torch.no_grad():
            # ray-sphere intersection
            a = torch.sum(ray_direction**2, dim=-1)
            b = 2 * torch.sum(ray_direction* ray_origin, dim=-1)
            c = torch.sum(ray_origin**2, dim=-1) - 1
            D = b**2 - 4 * a * c
            mask_valid = (c < 0.0) + (D > 0.0)
    
            t = (-b - torch.sqrt(torch.clamp(b**2 - 4 * a * c, 1e-9, None))) / torch.clamp(2 * a, 1e-9)
            t = torch.clamp(t[:,None], 0, max_depth)
            #x0 = ray_origin + t * ray_direction

            dadr = 2 * ray_direction
            dbdr = 2 * ray_origin
            dDdr = 2 * b[...,None] * dbdr - 4 * c[...,None] * dadr
            dt0dr = (dbdr - 1 / (2 * torch.sqrt(torch.clamp(D[...,None], 1e-9, None))) * dDdr) / torch.clamp(2 * a[...,None], 1e-9)
            dt0dr = dt0dr - (t / torch.clamp(a[...,None], 5e-10)) * dadr

            min_t = t.clone()

            max_t = (-b + torch.sqrt(torch.clamp(b**2 - 4 * a * c, 1e-9, None))) / torch.clamp(2 * a, 1e-9)
            max_t = torch.clamp(max_t[:,None], 0, max_depth)

            if True:
                mask_valid = (mask_valid == mask_valid).float()
                is_converged = 0 * mask_valid.clone()
                t_surf = torch.zeros_like(t, dtype=t.dtype, device=t.device)
                t_eval = torch.zeros_like(t, dtype=t.dtype, device=t.device)
                dtdr = dt0dr #dtdr = torch.zeros_like(ray_direction, dtype=t.dtype, device=t.device)
                sphere_tracing_kernel[(ray_direction.size(0),),(125,)](
                    # inputs
                    self.grid.detach(), ray_origin.detach(), ray_direction.detach(), t, max_t, mask_valid,
                    # results
                    is_converged,
                    # parameters
                    self.range, self.threshold,
                    # out
                    t_surf, t_eval, dtdr
                )
            else:
                t_surf, t_eval, dtdr, is_converged = self.sphere_tracing_kernel_pt(
                    ray_origin, ray_direction, t, max_t
                )
            t_eval = t_eval
            t0 = t_surf

            t0[is_converged == 0.0] = max_depth
            x0 = ray_origin + t0 * ray_direction # sphere tracing result

        #cuda.synchronize()
        #torch.cuda.synchronize()
        #print('reparam for sphere tracing started')

        # reparameterization
        dr = torch.zeros_like(ray_direction, device=ray_direction.device, requires_grad=True)
        ray_direction = ray_direction + dr
        if ray_direction.requires_grad == False:
            ray_direction = ray_direction.detach()
            distortion_coef = torch.ones_like(ray_direction[...,0:1], device=ray_direction.device)
        elif False:
            # using autograd to compute dTdr
            t_eval = t_eval + torch.sum(dtdr * dr, dim=-1, keepdim=True)

            xt = ray_origin + t_eval * ray_direction # continuous evaluation point for reperameterization

            # reparameterization
            # TODO detach f_t_p0 and grad_f_t_p0
            f_t, grad_f_t = self.forward(xt)#, mask=(is_converged > 0.0))
            f_t_p0, grad_f_t_p0 = self.forward(xt, backward_to_grid=False)#, mask=(is_converged > 0.0))
            V = -grad_f_t_p0 / (torch.sum(grad_f_t_p0**2, dim=-1, keepdim=True) + 1e-6) * f_t
            eps = torch.clamp(1 - torch.max(torch.abs(xt), dim=-1, keepdim=True)[0], 1e-20, 0.01)
            w_v = torch.clamp(1 - f_t_p0 / (torch.clamp(t_eval - min_t, 1e-20, None) * eps), 0, None)
            V_bar = V * w_v

            T_bar = t_eval * ray_direction + (V_bar - V_bar.detach())
            ray_direction_ = T_bar / torch.sqrt(torch.sum(T_bar**2, dim=-1, keepdim=True) + 1e-9)

            dTdr = []
            for idx_dim in range(3):
                d_output = torch.ones_like(
                    ray_direction_[...,idx_dim], 
                    requires_grad=False, 
                    device=ray_direction_.device
                )
                dTdr.append(torch.autograd.grad(
                    outputs=ray_direction_[...,idx_dim],
                    inputs=dr,
                    grad_outputs=d_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True)[0]
                )
            dTdr = torch.stack(dTdr, dim=-2)
            tr_dTdr = (dTdr[...,0,0] + dTdr[...,1,1] + dTdr[...,2,2])[...,None]

            ray_direction = ray_direction_
            distortion_coef = 1 + tr_dTdr - tr_dTdr.detach()
        else:
            # manually compute dTdr
            ray_direction = ray_direction.detach()
            I = torch.eye(3, dtype=t_eval.dtype, device=t_eval.device)

            xt = ray_origin + t_eval * ray_direction # continuous evaluation point for reperameterization
            dxdr = ray_direction[...,:,None] @ dtdr[...,None,:] + t_eval[...,None] * I
            #print('dxdr', dxdr.size())

            f_t, grad_f_t = self.forward(xt)#, mask=(is_converged > 0.0))
            dfdr = grad_f_t[...,None,:] @ dxdr
            #print('dfdr', dfdr.size())

            with torch.no_grad():
                f_t_p0, grad_f_t_p0 = self.forward(xt)#, mask=(is_converged > 0.0))
                hessian_f_t_p0 = self.compute_hessian_matrix(xt)
            dfp0dr = grad_f_t_p0[...,None,:] @ dxdr
            dgfp0dr = hessian_f_t_p0 @ dxdr
            #print('dfp0dr', dfp0dr.size())
            #print('dgfp0dr', dgfp0dr.size())

            gf2 = (torch.sum(grad_f_t_p0**2, dim=-1, keepdim=True) + 1e-6)
            V = -grad_f_t_p0 / gf2 * f_t
            dVdr = -f_t[...,None] * (I / gf2[...,None] - 2 * grad_f_t_p0[...,:,None] @ grad_f_t_p0[...,None,:] / (gf2[...,None]**2)) @ dgfp0dr
            dVdr = dVdr - (grad_f_t_p0 / gf2)[...,:,None] @ dfdr
            #print('dVdr', dVdr.size())

            eps = torch.clamp(1 - torch.max(torch.abs(xt), dim=-1, keepdim=True)[0], 1e-20, 0.01)
            depsdx_nonzero = (eps > 1e-20) * (eps < 0.01) * (torch.abs(xt) == torch.max(torch.abs(xt), dim=-1, keepdim=True)[0])
            depsdx = depsdx_nonzero.float() * torch.sgn(xt)
            depsdr = depsdx[...,None,:] @ dxdr
            #print('depsdr', depsdr.size())

            w_v_denom = torch.clamp(t_eval - min_t, 1e-20, None) * eps
            w_v = torch.clamp(1 - f_t_p0 / w_v_denom, 0, None)
            ddenomdr = torch.clamp(t_eval - min_t, 1e-20, None)[...,None] * depsdr
            ddenomdr = ddenomdr + ((((t_eval - min_t) > 1e-20).float() * eps) * (dtdr - dt0dr))[...,None,:]
            dwdr_nonzero = w_v > 0
            dwdr = dwdr_nonzero.float()[...,:,None] * (-dfp0dr / w_v_denom[...,:,None] + (f_t_p0 / (w_v_denom**2))[...,:,None] * ddenomdr)
            dwdr[w_v[...,0] <= 0] = 0.0
            #print('dwdr', dwdr.size())

            V_bar = V * w_v
            dVbdr = w_v[...,:,None] * dVdr + V[...,:,None] @ dwdr
            #print('dVbdr', dVbdr.size())

            T_bar = t_eval * ray_direction + (V_bar - V_bar.detach())
            dTbdr = dxdr + dVbdr
            #print('dTbdr', dTbdr.size())

            ray_direction = T_bar / torch.sqrt(torch.sum(T_bar**2, dim=-1, keepdim=True) + 1e-9)
            dTdr = (I - ray_direction[...,:,None] @ ray_direction[...,None,:]) @ dTbdr / torch.sqrt(torch.sum(T_bar**2, dim=-1, keepdim=True) + 1e-3)[...,:,None]
            #print('dTdr', dTdr.size())

            tr_dTdr = (dTdr[...,0,0] + dTdr[...,1,1] + dTdr[...,2,2])[...,None]
            distortion_coef = 1 + tr_dTdr - tr_dTdr.detach()

            #print(tr_dTdr)



        #try:
        #    self.cnt
        #except:
        #    self.cnt = 0
        #self.cnt += 1

        #sub_reso = 8
        if False:#(x0.size(0) == (65536 * sub_reso**2)) and (self.cnt > 2):
            print(tr_dTdr)
            import matplotlib.pyplot as plt
            xt_slice = xt.detach().view(256,256,sub_reso, sub_reso,3)[128,:,sub_reso//2,:,:].reshape(-1,3)
            r_slice = ray_direction.detach().view(256,256,sub_reso, sub_reso,3)[128,:,sub_reso//2,:,:].reshape(-1,3)
            motion_dir = r_slice[1:] - r_slice[:-1]
            #ray_motion_dir /= torch.sqrt(torch.sum(ray_motion_dir**2, dim=-1, keepdim=True))
            t0_slice = t0.view(256,256,sub_reso,sub_reso)[128,:,sub_reso//2,:].reshape(-1)
            t_eval_slice = t_eval.view(256,256,sub_reso,sub_reso)[128,:,sub_reso//2,:].reshape(-1)
            #grad_t_eval_slice = (dtdr).view(256,256,sub_reso,sub_reso,3)[128,:,sub_reso//2,:].reshape(-1,3)
            grad_t_eval_slice = (dtdr * is_converged[...,None]).view(256,256,sub_reso,sub_reso,3)[128,:,sub_reso//2,:].reshape(-1,3)
            w_v_slice = w_v.view(256,256,sub_reso,sub_reso)[128,:,sub_reso//2,:].reshape(-1)
            #grad_w_v_slice = grad_w_v.view(256,256,9,3)[128,:,4]
            plt.subplot(3,1,1)
            plt.plot(t0_slice.cpu())
            plt.plot(t_eval_slice.detach().cpu())
            plt.ylim([torch.min(t_eval_slice).item()-0.1, torch.max(t_eval_slice).item()+0.1])
            plt.subplot(3,1,2)
            plt.plot(w_v_slice.detach().cpu())
            plt.subplot(3,1,3)
            import numpy as np
            plt.plot(np.cumsum(torch.sum(grad_t_eval_slice[:-1] * motion_dir, dim=-1).cpu()))
            plt.grid()
            plt.show()


        # differentiable surface intersection
        f, grad_f = self.forward(x0, mask=(is_converged > 0.0))
        with torch.no_grad():
            gDl = torch.sum(grad_f * ray_direction, dim=1, keepdim=True)
            is_converged *= gDl[...,0] < 0
            gDl = torch.clamp(gDl, None, 0) - 1e-2

        ray_direction = ray_direction * is_converged[:,None] + ray_direction.detach() * (1 - is_converged)[:,None]
        distortion_coef = distortion_coef * is_converged[:,None] + distortion_coef.detach() * (1 - is_converged)[:,None]

        x = ray_origin + t0 * ray_direction - f / gDl * ray_direction * is_converged[:,None]

        grad_f = self.forward(x, mask=(is_converged > 0.0))[1]
        n = grad_f / torch.sqrt(torch.sum(grad_f**2, dim=-1, keepdim=True) + 1e-5) * is_converged[:,None]

        if True:
            with torch.no_grad():
                x_eval = ray_origin + t_eval * ray_direction
                grad_f_eval = self.forward(x_eval, mask=(is_converged == 0.0))[1]
                n_eval = grad_f_eval / torch.sqrt(torch.sum(grad_f_eval**2, dim=-1, keepdim=True) + 1e-5) * (1 - is_converged[:,None])
            n = n + n_eval

        #cuda.synchronize()
        #torch.cuda.synchronize()
        #print('sphere tracing done')

        return x, n, is_converged, distortion_coef

    def minimum_sdf(self, ray_origin, ray_direction):
        # assert |ray_direction| == 1
        ray_direction = ray_direction / torch.sqrt(torch.sum(ray_direction**2, dim=-1, keepdim=True) + 1e-6)

        with torch.no_grad():
            # ray-sphere intersection
            a = torch.sum(ray_direction**2, dim=-1)
            b = 2 * torch.sum(ray_direction* ray_origin, dim=-1)
            c = torch.sum(ray_origin**2, dim=-1) - 1
            D = b**2 - 4 * a * c

            t_min = (-b - torch.sqrt(torch.clamp(D, 0, None))) / torch.clamp(2 * a, 1e-9)
            t_max = (-b + torch.sqrt(torch.clamp(D, 0, None))) / torch.clamp(2 * a, 1e-9)
            
            #ofs = torch.arange(num_points, dtype=ray_origin.dtype, device=ray_origin.device) / num_points
            #samples = ray_origin[:,None,:] + (t_min[:,None] + ofs[None,:] * (t_max - t_min)[:,None])[:,:,None] * ray_direction[:,None,:]
            for _ in range(3):
                minimum_sdf_kernel[(ray_origin.size(0),),(128,)](
                    # inputs
                    self.grid.detach(), ray_origin.detach(), ray_direction.detach(), t_min, t_max,
                    # parameters
                    self.range,
                )
            x = ray_origin + t_min[:,None] * ray_direction
        #min_sdf = torch.min(self.forward(samples.view(-1,3))[0].view(-1,num_samples,1), dim=1)[0]
        min_sdf = self.forward(x)[0]

        return min_sdf

    # img:              BS*3*H*W
    # mask:             BS*1*H*W
    # proj_matrix:      BS*4*4
    # extrinsic_matrix: BS*4*4
    # grid_normal:      BS*Hn*Wn*3
    def recover_sparse_rmap(
        self, 
        img, 
        mask, 
        proj_matrix, 
        extrinsic_matrix, 
        out_size=256,
        scale_factor=1.
    ):
        img = F.interpolate(
            img, 
            scale_factor=scale_factor,
            mode='bilinear', 
            align_corners=False
        )
        mask = F.interpolate(
            mask, 
            scale_factor=scale_factor,
            mode='bilinear', 
            align_corners=False
        )
        mask[mask < 0.999] = 0.0
        M = torch.tensor([
            [scale_factor, 0, 0, 0],
            [0, scale_factor, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0 ,1]
        ], dtype=proj_matrix.dtype, device=proj_matrix.device)
        proj_matrix = M @ proj_matrix

        # render normal map
        with torch.no_grad():
            normal_map = self.render(
                proj_matrix,
                extrinsic_matrix,
                image_width=img.size(-1),
                image_height=img.size(-2),
                subpixel_resolution=1
            )[2]

            mask *= (torch.sum(normal_map**2, dim=1, keepdim=True) > 0.8).float()

            sparse_rmap = img2rmap(
                img, 
                normal_map, 
                mask,
                out_size
            )
        return sparse_rmap

    # img:              BS*3*H*W
    # mask:             BS*1*H*W
    # proj_matrix:      BS*4*4
    # extrinsic_matrix: BS*4*4
    # grid_normal:      BS*Hn*Wn*3
    def recover_sparse_rmap_2(
        self, 
        ref_imgs, 
        ref_masks, 
        proj_matrices, 
        extrinsics, 
        out_size=256,
        jitter=False
    ):
        image_width = ref_imgs.size(-1)
        image_height = ref_imgs.size(-2)
        grid_length = (2 * self.range) / self.resolution
        ref_masks *= torch.any(ref_imgs > 0.0, dim=1, keepdim=True).float()
        # dense sampling on SDF surface
        w,v,u = torch.meshgrid(torch.arange(self.resolution),torch.arange(self.resolution),torch.arange(self.resolution))
        x = self.range * (2 * (u.float() + 0.5) / self.resolution - 1)
        y = self.range * (2 * (v.float() + 0.5) / self.resolution - 1)
        z = self.range * (2 * (w.float() + 0.5) / self.resolution - 1)
        grid_coords = torch.stack([x,y,z], dim=-1).to(self.grid.device)
        if jitter:
            jitter_ofs = (torch.rand_like(grid_coords, device=grid_coords.device) - 0.5)
            grid_coords = grid_coords + jitter_ofs * grid_length
        grids_on_surface = torch.abs(self.grid.detach()) < (2 * grid_length)
        surf_points = grid_coords[grids_on_surface] # N * 3
        with torch.no_grad():
            f, grad_f = self.forward(surf_points)
            surf_points = surf_points - f * grad_f

        sparse_rmaps = []
        for idx_view in range(extrinsics.size(0)):
            # visibility
            ray_origin = torch.inverse(extrinsics)[idx_view,:3,3][None,:].repeat(surf_points.size(0),1)
            ray_direction = surf_points - ray_origin
            ray_direction = ray_direction / torch.sqrt(torch.sum(ray_direction**2, dim=-1, keepdim=True))
            surf_points_st, surf_normal_st, st_is_valid, dist_coef_st = self.sphere_tracing(ray_origin, ray_direction)
            mask_vis = (torch.sum((surf_points_st - surf_points)**2, dim=-1) < (2*grid_length)) * (st_is_valid > 0)

            surf_points_vis = surf_points_st[mask_vis]
            normal_vis = surf_normal_st[mask_vis]
            ray_direction_vis = surf_points_vis - ray_origin[mask_vis]
            dist_coef_vis = dist_coef_st[mask_vis]

            # eval surface normal
            if True:
                surf_points_vis = surf_points_vis.detach()
                ray_direction_vis = ray_direction_vis.detach()
                normal_vis = self.forward(surf_points_vis)[1] # N * 3
                dist_coef_vis = dist_coef_vis.detach()

            normal_vis = normal_vis / torch.sqrt(torch.sum(normal_vis**2, dim=-1, keepdim=True) + 1e-24)


            normal_vis = (extrinsics[idx_view,:3,:3] @ normal_vis.T).T
            normal_vis = normal_vis * torch.tensor([1,-1,-1], dtype=normal_vis.dtype, device=normal_vis.device)

            # sample projected coords
            m_h = (proj_matrices[idx_view][:3,:3] @ surf_points_vis.T + proj_matrices[idx_view][:3,3:4]).T
            grid2d_x = 2 * m_h[:,0] / m_h[:,2] / image_width - 1
            grid2d_y = 2 *  m_h[:,1] / m_h[:,2] / image_height - 1
            grid2d = torch.stack([grid2d_x, grid2d_y], dim=-1) # N * 2

            # eval pixel value
            pixel_val_vis = torch.clamp(F.grid_sample(
                ref_imgs[idx_view:idx_view+1],
                grid2d[None,None,:,:],
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            ), 0, None)[0,:,0,:].T # N * 3

            pixel_mask_vis = torch.nn.functional.grid_sample(
                ref_masks[idx_view:idx_view+1],
                grid2d[None,None,:,:],
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )[0,:,0,:].T # N * 3

            # sampling on the image plane
            pixel_mask_vis = (pixel_mask_vis > 0.99).float()

            normal_map = self.render(
                proj_matrices[idx_view:idx_view+1],
                extrinsics[idx_view:idx_view+1],
                image_width=image_width,
                image_height=image_height,
                subpixel_resolution=1,
                jitter=jitter
            )[2]

            mask = ref_masks[idx_view:idx_view+1] * (torch.sum(normal_map**2, dim=1, keepdim=True) > 0.8).float()

            pixel_val_img = ref_imgs[idx_view:idx_view+1].view(3,image_height*image_width).transpose(-1,-2)
            normal_img = normal_map.view(3,image_height*image_width).transpose(-1,-2)
            pixel_mask_img = mask.view(1,image_height*image_width).transpose(-1,-2)

            pixel_val_vis = torch.cat([pixel_val_vis, pixel_val_img], dim=0)
            normal_vis = torch.cat([normal_vis, normal_img], dim=0)
            pixel_mask_vis = torch.cat([pixel_mask_vis, pixel_mask_img], dim=0)

            sparse_rmap = img2rmap(
                pixel_val_vis.transpose(-1,-2)[None,:,:,None], 
                normal_vis.transpose(-1,-2)[None,:,:,None], 
                pixel_mask_vis.transpose(-1,-2)[None,:,:,None],
                out_size
            )
            sparse_rmaps.append(sparse_rmap)
        return torch.cat(sparse_rmaps, dim=0)

            

    # img:              BS*3*H*W
    # mask:             BS*1*H*W
    # proj_matrix:      BS*4*4
    # extrinsic_matrix: BS*4*4
    # grid_normal:      BS*Hn*Wn*3
    def recover_rmap_old(self, img, mask, proj_matrix, extrinsic_matrix, grid_normal, block_size=16):
        batch_size = img.size(0)
        image_height = img.size(-2)
        image_width = img.size(-1)
        device=img.device
        grid_normal_norm = torch.sqrt(torch.sum(grid_normal**2, dim=-1, keepdim=True))
        grid_normal = grid_normal / (grid_normal_norm + 1e-6)

        # render normal map
        with torch.no_grad():
            depth_map, normal_map = self.render(
                proj_matrix,
                extrinsic_matrix,
                image_width=img.size(-1),
                image_height=img.size(-2),
                subpixel_resolution=1
            )[1:3]

        # find knn
        query_normals = grid_normal.view(batch_size,-1,3)

        num_blocks_w = (img.size(-1) - 1) // block_size + 1
        num_blocks_h = (img.size(-2) - 1) // block_size + 1
        num_blocks = num_blocks_h * num_blocks_w

        max_scores = torch.zeros_like(query_normals[:,:,0:1],device=device).repeat(1,1,num_blocks)
        max_scores[:] = -1e6
        argmax_pixels = torch.zeros_like(max_scores[...,None],device=device).repeat(1,1,1,3)
        argmax_pixels[...,2] = 1.0

        nearest_normal_pixels_kernel[(batch_size,query_normals.size(1)),(num_blocks,)](
            # inputs
            depth_map, normal_map, mask, query_normals,
            # params
            block_size,
            # result
            argmax_pixels, max_scores
        )

        argmax_pixels_h = torch.cat([argmax_pixels, torch.ones_like(argmax_pixels[...,:1], device=device)], dim=-1)
        inv_proj = torch.inverse(proj_matrix)
        argmax_surf_points = torch.sum(inv_proj[:,None,None,:3,:4] * argmax_pixels_h[:,:,:,None,:], dim=-1)
        target_normals = torch.sum(extrinsic_matrix[:,None,:3,:3] * (query_normals * torch.tensor([1,-1,-1],device=device))[...,None], dim=-2)
        target_normals = target_normals[:,:,None,:].repeat(1,1,num_blocks,1)

        def solve_dif_x(sdf_grad, sdf_hessian, dif_sdf, dif_n):
            sdf_grad_norm = torch.sqrt(torch.sum(sdf_grad**2, dim=-1, keepdim=True))
            sdf_normal = (sdf_grad / sdf_grad_norm)[...,:,None]
            I = torch.eye(3, dtype=sdf_grad.dtype, device=sdf_grad.device)[None]
            shape_operator = (I - sdf_normal @ sdf_normal.transpose(-1,-2)) @ sdf_hessian / sdf_grad_norm[...,None]

            U,s,V = torch.svd(shape_operator)
            U_cosine = torch.sum(U * sdf_normal, dim=-2)
            sorted_indices = torch.argsort(torch.abs(U_cosine),dim=-1)
            s = torch.gather(s, -1, sorted_indices[...,:])
            U = torch.gather(U, -1, sorted_indices[...,None,:].repeat(1,3,1))
            V = torch.gather(V, -1, sorted_indices[...,None,:].repeat(1,3,1))
            A = s[...,:2,None] * V.transpose(-1,-2)[...,:2,:]
            b = U.transpose(-1,-2)[...,:2,:] @ dif_n[...,:,None]
            A = torch.cat([A, sdf_grad[...,None,:]], dim=-2)
            b = torch.cat([b, dif_sdf[...,None,:]], dim=-2)

            U,s,V = torch.svd(A)

            inv_s = 1 / s
            inv_s[torch.abs(s) < 1e-2] = 0.0
            inv_S = torch.zeros_like(U, dtype=U.dtype, device=U.device)
            inv_S[...,0,0] = inv_s[...,0]
            inv_S[...,1,1] = inv_s[...,1]
            inv_S[...,2,2] = inv_s[...,2]
            pinv_A = V @ inv_S @ U.transpose(-1,-2)

            dif_x = (pinv_A @ b)[...,0]

            return dif_x

            print(shape_operator[1])
            print(U[1])
            print(s[1])
            print(V[1])
            print(sdf_normal[1])
            print(A[1])
            print(b[1])
            #print(dif_x)
            print(torch.linalg.eig(shape_operator[1]))
            exit()

            s_ = torch.gather(s, -1, sorted_indices[...,:])
            s_[...,2:3] = sdf_grad_norm
            U_ = torch.gather(U, -1, sorted_indices[...,None,:].repeat(1,3,1))
            sgn_u3 = torch.sign(torch.sum(U_[...,:,2] * sdf_grad, dim=-1, keepdim=True))
            sgn_u3[sgn_u3 == 0] = 1.0
            U_[...,:,2] *= sgn_u3
            inv_s_ = 1 / s_
            inv_s_[torch.abs(s_) < 1e-2] = 0.0
            inv_S_ = torch.zeros_like(U_, dtype=U_.dtype, device=U_.device)
            inv_S_[...,0,0] = inv_s_[...,0]
            inv_S_[...,1,1] = inv_s_[...,1]
            inv_S_[...,2,2] = inv_s_[...,2]
            pinv_A = U_ @ inv_S_

            b = U_.transpose(-1,-2)[...,:2,:3] @ dif_n[...,:,None]
            b = torch.cat([b, dif_sdf[...,:,None]], dim=-2)

            dif_x = (pinv_A @ b)[...,0]

            if False:
                print(U.size(),s.size(), V.size(), sorted_indices.size())
                print('s:')
                print(s[:10])
                print('s_:')
                print(s_[:10])
                print('U:')
                print(U[0])
                print('U_:')
                print(U_[0])
                print('V:')
                print(V[0])
                print('V_:')
                print(pinv_A[0])
                print(b[0])
                print(dif_x[0])
                exit()

            return dif_x

        # max_scores: bs*num_targets*num_candidates
        # argmax_surf_points: bs*num_targets*num_candidates*3
        initial_cosine_threshold = 0.9
        is_valid = max_scores > initial_cosine_threshold
        with torch.no_grad():
            valid_surf_points = argmax_surf_points.view(-1,3)[is_valid.view(-1)]
            valid_target_normals = target_normals.view(-1,3)[is_valid.view(-1)]

            # make sure that the points are on the surface
            sdf_val, sdf_grad = self.forward(valid_surf_points)
            sdf_grad_norm = torch.sqrt(torch.sum(sdf_grad**2, dim=-1, keepdim=True))
            valid_surf_points -= sdf_val * sdf_grad / (sdf_grad_norm**2 + 1e-3)

            for _ in range(10):
                # update surface points
                sdf_val, sdf_grad = self.forward(valid_surf_points)
                sdf_hessian = self.compute_hessian_matrix(valid_surf_points)
                sdf_grad_norm = torch.sqrt(torch.sum(sdf_grad**2, dim=-1, keepdim=True))
                sdf_normal = sdf_grad / (sdf_grad_norm + 1e-6)

                dif_n = (valid_target_normals - sdf_normal)
                dif_x = solve_dif_x(sdf_grad, sdf_hessian, 0 * sdf_val, dif_n)
                dif_x_norm = torch.sqrt(torch.sum(dif_x**2, dim=-1, keepdim=True))

                valid_surf_points += dif_x / (dif_x_norm + 1e-3) * torch.clamp(dif_x_norm, None, 1e-3)

                # make sure that the points are on the surface
                sdf_val, sdf_grad = self.forward(valid_surf_points)
                sdf_grad_norm = torch.sqrt(torch.sum(sdf_grad**2, dim=-1, keepdim=True))
                valid_surf_points -= sdf_val * sdf_grad / (sdf_grad_norm**2 + 1e-3)

                # eval
                sdf_val, sdf_grad = self.forward(valid_surf_points)
                sdf_grad_norm = torch.sqrt(torch.sum(sdf_grad**2, dim=-1, keepdim=True))
                sdf_normal = sdf_grad / (sdf_grad_norm + 1e-6)
                valid_scores = torch.sum(sdf_normal * valid_target_normals, dim=-1, keepdim=True)
                
                mean_residual = 1-torch.mean(valid_scores)
                print('1-score:',mean_residual, ', sdf range:',torch.min(sdf_val), torch.max(sdf_val))
                if mean_residual < 1e-5:
                    break

        # differentiable surface point tracking
        sdf_val, sdf_grad = self.forward(valid_surf_points.detach())
        with torch.no_grad():
            sdf_hessian = self.compute_hessian_matrix(valid_surf_points)
        dif_x = -solve_dif_x(sdf_grad.detach(), sdf_hessian, sdf_val, sdf_grad-sdf_grad.detach())
        print(torch.min(dif_x), torch.max(dif_x))
        valid_surf_points = valid_surf_points + dif_x

        is_valid.view(-1)[is_valid.clone().view(-1)] *= ((1 - valid_scores) < 2e-4).view(-1)
        argmax_surf_points.view(-1,3)[is_valid.view(-1)] = valid_surf_points[(1 - valid_scores.view(-1)) < 2e-4]

        # TODO: visibility check

        # projection onto image plane
        argmax_pixel_coords_h = torch.sum(proj_matrix[:,None,None,:3,:3] * argmax_surf_points[:,:,:,None,:], dim=-1) + proj_matrix[:,None,None,:3,3]
        argmax_pixel_coords = argmax_pixel_coords_h[...,:2] / torch.clamp(argmax_pixel_coords_h[...,2:3], 1e-3, None)
        argmax_pixel_coords_n = 2 * argmax_pixel_coords / torch.tensor([image_width, image_height], dtype=argmax_pixel_coords.dtype, device=device) - 1

        # sample pixel values
        argmax_pixel_values = torch.clamp(F.grid_sample(
            img,
            argmax_pixel_coords_n,
            mode='bicubic',
            padding_mode='zeros',
            align_corners=False
        ), 0, None)

        # compute mean of sampled pixel values
        argmax_pixel_weights = is_valid[:,None] * torch.sum(argmax_pixel_values, dim=1, keepdim=True)
        mean_sampled_pixel_value = torch.sum(argmax_pixel_values * argmax_pixel_weights, dim=-1) / (torch.sum(argmax_pixel_weights, dim=-1) + 1e-4)
        recon_rmap = mean_sampled_pixel_value.view(batch_size, img.size(1), grid_normal.size(1), grid_normal.size(2))

        #loss = torch.sum(argmax_surf_points)
        #loss.backward()
        #print(torch.min(self.grid.grad), torch.max(self.grid.grad))
        #print(is_valid.size())
        #print(argmax_surf_points.size())
        #print(argmax_pixel_coords.size())
        #print(argmax_pixel_values.size())
        #print(torch.cat([argmax_pixel_coords_n, is_valid[...,None]], dim=-1)[0,8000])

        if False:
            from .sfs_utils import plot_normal_map, plot_hdr
            import matplotlib.pyplot as plt
        
            for i in range(batch_size):
                plt.subplot(4,batch_size,1+i)
                plot_hdr(img, i)
                plt.subplot(4,batch_size,batch_size+1+i)
                plot_normal_map(normal_map, i)
                plt.subplot(4,batch_size,2*batch_size+1+i)
                plot_normal_map(grid_normal.transpose(-1,-2).transpose(-3,-2), i)
                plt.subplot(4,batch_size,3*batch_size+1+i)
                plot_hdr(recon_rmap, i)
            plt.show()

        return recon_rmap

        max_scores,indices = torch.sort(max_scores, dim=-1, descending=True)
        argmax_pixels = torch.gather(argmax_pixels, -2, indices[...,None].repeat(1,1,1,3))
        argmax_surf_points = torch.gather(argmax_surf_points, -2, indices[...,None].repeat(1,1,1,3))

        argmax_pixel_vis = argmax_pixels[...,0,:].view(batch_size,grid_normal.size(1),grid_normal.size(2),3)
        argmax_surf_points_vis = argmax_surf_points[...,0,:].view(batch_size,grid_normal.size(1),grid_normal.size(2),3)


        from .sfs_utils import plot_normal_map
        import matplotlib.pyplot as plt
        plot_normal_map(normal_map)
        plt.show()

        for i in range(batch_size):
            plt.subplot(3,batch_size,1+i)
            plt.imshow(argmax_surf_points_vis[i,:,:,0].detach().cpu(), vmin=-1, vmax=1)
            plt.subplot(3,batch_size,batch_size+1+i)
            plt.imshow(argmax_surf_points_vis[i,:,:,1].detach().cpu(), vmin=-1, vmax=1)
            plt.subplot(3,batch_size,2*batch_size+1+i)
            plt.imshow(argmax_surf_points_vis[i,:,:,2].detach().cpu(), vmin=-1, vmax=1)
        plt.show()

        print('hahaha')
        print(argmax_pixels.size())
        print(argmax_surf_points.size())
        print(normal_map.size())
        print(mask.size())
        print(query_normals.size())
        print(max_scores.size())
        print(argmax_pixels.size())

        exit()

    # num_view*num_pixels*2
    def render_pixels(
        self, 
        pixel_coords, 
        proj_matrices, 
        extrinsics, 
        subpixel_resolution=3, 
        shading_function=None, 
        max_depth=100.0, 
        jitter=False
    ):
        device=proj_matrices.device
        pixel_coords = torch.cat([pixel_coords, torch.ones_like(pixel_coords[...,:1])], dim=-1)

        q, p = torch.meshgrid(torch.arange(subpixel_resolution), torch.arange(subpixel_resolution))
        p = (p.float() + 0.5) / subpixel_resolution - 0.5
        q = (q.float() + 0.5) / subpixel_resolution - 0.5
        subpixel_offsets = torch.stack([p,q, torch.zeros_like(p)], dim=-1).view(-1,3).to(device)        

        m_sub = pixel_coords[:,:,None,:] + subpixel_offsets[None,None,:,:]
        if jitter:
            jitter_ofs = (torch.rand_like(m_sub, device=device) - 0.5) / subpixel_resolution
            jitter_ofs[...,-1] = 0.0
            m_sub = m_sub + jitter_ofs
    
        inv_proj = torch.inverse(proj_matrices)[:,:3,:3]

        ray_dir = torch.sum(inv_proj[:,None,None,:,:] * m_sub[...,None,:], dim=-1)
        ray_dir = ray_dir / torch.sqrt(torch.sum(ray_dir**2, dim=-1, keepdim=True) + 1e-6)

        ray_origin = torch.inverse(extrinsics)[:,:3,3][:,None,None,:].repeat(1,ray_dir.size(1),ray_dir.size(2),1)

        x,n,is_valid, dist_coef = self.sphere_tracing(
            ray_origin.view(-1,3),
            ray_dir.view(-1,3),
            max_depth=max_depth
        )

        x = x.view(-1,ray_origin.size(1),subpixel_resolution**2,3)
        n = n.view(-1,ray_origin.size(1),subpixel_resolution**2,3)
        is_valid = is_valid.view(-1,ray_origin.size(1),subpixel_resolution**2,1).float()
        dist_coef = dist_coef.view(-1,ray_origin.size(1),subpixel_resolution**2,1)

        d = torch.sum((x - ray_origin) * extrinsics[:,None,None,2,:3], dim=-1, keepdim=True)
        n = torch.sum(n[...,None,:] * extrinsics[:,None,None,:3,:3], dim=-1)
        n = n * torch.tensor([1,-1,-1], dtype=n.dtype, device=device)

        u_h = torch.sum((x) * proj_matrices[:,None,None,0,:3], dim=-1) + proj_matrices[:,None,None,0,3]
        v_h = torch.sum((x) * proj_matrices[:,None,None,1,:3], dim=-1) + proj_matrices[:,None,None,1,3]
        w_h = torch.sum((x) * proj_matrices[:,None,None,2,:3], dim=-1) + proj_matrices[:,None,None,2,3]
        u_ = u_h / torch.clamp(w_h, 1e-5, None)
        v_ = v_h / torch.clamp(w_h, 1e-5, None)

        p_ = u_ - pixel_coords[...,0:1]
        q_ = v_ - pixel_coords[...,1:2]
        r2 = p_**2 + q_**2
        weight = torch.exp(-10*r2)[...,None] # n_view*n_pixel*n_sample*1
        weight = weight * dist_coef

        weight_sum = (torch.sum(weight, dim=-2) + 1e-6)#.detach()
        if shading_function is None:
            pixel_vals = None
        else:
            shading = shading_function(
                x.view(len(proj_matrices),-1,3),
                n.view(len(proj_matrices),-1,3),
                -ray_dir.view(len(proj_matrices),-1,3),
                extrinsics,
            ).view(len(proj_matrices),-1,subpixel_resolution**2,3) * is_valid
            pixel_vals = torch.sum(weight * shading, dim=-2) / weight_sum

        x = torch.sum(weight * x, dim=-2) / weight_sum
        normal = torch.sum(weight * n * is_valid, dim=-2) / weight_sum
        mask = torch.sum(weight * is_valid, dim=-2) / weight_sum
        depth = torch.sum(weight * d, dim=-2) / weight_sum

        return pixel_vals, depth, normal, mask, x

    def render(
        self, 
        proj_matrices, 
        extrinsics, 
        image_width=128, 
        image_height=128, 
        subpixel_resolution=3, 
        shading_function=None, 
        max_depth=100.0, 
        jitter=False
    ):
        device=proj_matrices.device

        v, u = torch.meshgrid(torch.arange(image_height), torch.arange(image_width))
        m = torch.stack([u,v], dim=-1).to(device) + 0.5 # H*W*3
        pixel_coords = m[None].view(1,-1,2).repeat(len(proj_matrices),1,1)

        pixel_val, depth, normal, mask, x = self.render_pixels(
            pixel_coords, 
            proj_matrices, 
            extrinsics, 
            subpixel_resolution=subpixel_resolution, 
            shading_function=shading_function, 
            max_depth=max_depth, 
            jitter=jitter
        )

        if not (shading_function is None):
            img = pixel_val.view(len(proj_matrices), image_height, image_width, -1)
        else:
            img = None
        depth = depth.view(len(proj_matrices), image_height, image_width, -1)
        normal = normal.view(len(proj_matrices), image_height, image_width, -1)
        mask = mask.view(len(proj_matrices), image_height, image_width, -1)
        x = x.view(len(proj_matrices), image_height, image_width, -1)

        if not (img is None):
            img = img.transpose(-1,-2).transpose(-2,-3)
        x = x.transpose(-1,-2).transpose(-2,-3)
        depth = depth.transpose(-1,-2).transpose(-2,-3)
        normal = normal.transpose(-1,-2).transpose(-2,-3)
        mask = mask.transpose(-1,-2).transpose(-2,-3)

        return img, depth, normal, mask, x

        q, p = torch.meshgrid(torch.arange(subpixel_resolution), torch.arange(subpixel_resolution))
        p = (p.float() + 0.5) / subpixel_resolution
        q = (q.float() + 0.5) / subpixel_resolution
        subpixel_offsets = torch.stack([p,q, torch.zeros_like(p)], dim=-1).view(-1,3).to(device)

        v, u = torch.meshgrid(torch.arange(image_height), torch.arange(image_width))
        m = torch.stack([u,v,torch.ones_like(u)], dim=-1).to(device) # H*W*3
        m_sub = m[:,:,None,:] + subpixel_offsets[None,None,:,:] # H*W*M*3
        inv_proj = torch.inverse(proj_matrices)[:,:3,:3]

        m_sub = m_sub[None,:,:,:,None,:]

        if jitter:
            m_sub = m_sub.repeat(len(proj_matrices),1,1,1,1,1)
            jitter_ofs = (torch.rand_like(m_sub, device=device) - 0.5) / subpixel_resolution
            jitter_ofs[...,-1] = 0.0
            m_sub = m_sub + jitter_ofs

        ray_dir = torch.sum(inv_proj[:,None,None,None,:,:] * m_sub, dim=-1)
        ray_dir = ray_dir / torch.sqrt(torch.sum(ray_dir**2, dim=-1, keepdim=True) + 1e-6)

        cam_centers = torch.inverse(extrinsics)[:,:3,3][:,None,None,:].repeat(1,image_height,image_width,1)
        ray_origin = cam_centers[:,:,:,None,:].repeat(1,1,1,subpixel_resolution**2,1)

        x,n,is_valid, dist_coef = self.sphere_tracing(
            ray_origin.view(-1,3),
            ray_dir.view(-1,3),
            max_depth=max_depth
        )

        x = x.view(-1,image_height,image_width,subpixel_resolution**2,3)
        n = n.view(-1,image_height,image_width,subpixel_resolution**2,3)
        is_valid = is_valid.view(-1,image_height,image_width,subpixel_resolution**2,1).float()
        dist_coef = dist_coef.view(-1,image_height,image_width,subpixel_resolution**2,1)

        d = torch.sum((x - ray_origin) * extrinsics[:,None,None,None,2,:3], dim=-1, keepdim=True)
        n = torch.sum(n[...,None,:] * extrinsics[:,None,None,None,:3,:3], dim=-1)
        n = n * torch.tensor([1,-1,-1], dtype=n.dtype, device=device)

        u_h = torch.sum((x) * proj_matrices[:,None,None,None,0,:3], dim=-1) + proj_matrices[:,None,None,None,0,3]
        v_h = torch.sum((x) * proj_matrices[:,None,None,None,1,:3], dim=-1) + proj_matrices[:,None,None,None,1,3]
        w_h = torch.sum((x) * proj_matrices[:,None,None,None,2,:3], dim=-1) + proj_matrices[:,None,None,None,2,3]
        u_ = u_h / torch.clamp(w_h, 1e-5, None)
        v_ = v_h / torch.clamp(w_h, 1e-5, None)

        p_ = u_ - m[:,:,0][None,:,:,None] - 0.5
        q_ = v_ - m[:,:,1][None,:,:,None] - 0.5
        r2 = p_**2 + q_**2
        weight = torch.exp(-10*r2)[...,None] # n_view*h*w**n_sample*1
        weight = weight * dist_coef

        if shading_function is None:
            img = None
        else:
            shading = shading_function(
                x.view(len(proj_matrices),-1,3),
                n.view(len(proj_matrices),-1,3),
                -ray_dir.view(len(proj_matrices),-1,3),
                proj_matrices
            ).view(-1,image_height,image_width,subpixel_resolution**2,3)
            img = torch.sum(weight * shading, dim=-2) / (torch.sum(weight, dim=-2) + 1e-6)

        x = torch.sum(weight * x, dim=-2) / (torch.sum(weight, dim=-2) + 1e-6)
        normal = torch.sum(weight * n, dim=-2) / (torch.sum(weight, dim=-2) + 1e-6)
        mask = torch.sum(weight * is_valid, dim=-2) / (torch.sum(weight, dim=-2) + 1e-6)
        depth = torch.sum(weight * d, dim=-2) / (torch.sum(weight, dim=-2) + 1e-6)

        if not (img is None):
            img = img.transpose(-1,-2).transpose(-2,-3)
        x = x.transpose(-1,-2).transpose(-2,-3)
        depth = depth.transpose(-1,-2).transpose(-2,-3)
        normal = normal.transpose(-1,-2).transpose(-2,-3)
        mask = mask.transpose(-1,-2).transpose(-2,-3)

        return img, depth, normal, mask, x        

    def recover_rmap(
        self, 
        ref_imgs, 
        proj_matrices, 
        extrinsics, 
        image_width=128, 
        image_height=128, 
        jitter=True, 
        max_depth=100.0,
        out_size=128,
        projection_mode = 'sphere',
        subpixel_resolution=3
        ):
        device=proj_matrices.device

        v, u = torch.meshgrid(torch.arange(image_height), torch.arange(image_width))
        m = torch.stack([u,v], dim=-1).to(device) + 0.5 # H*W*3
        pixel_coords = m[None].view(1,-1,2).repeat(len(proj_matrices),1,1) # N*(H*W)*2

        if subpixel_resolution > 1:
            sub_v, sub_u = torch.meshgrid(torch.arange(subpixel_resolution), torch.arange(subpixel_resolution))
            sub_m = (torch.stack([sub_u,sub_v], dim=-1).to(device).view(-1,2) + 0.5) / subpixel_resolution # (Hs*Ws)*3
            pixel_coords = pixel_coords[:,:,None,:] - 0.5 + sub_m[None,None,:,:]
            pixel_coords = pixel_coords.view(len(proj_matrices),-1,2)

        if jitter:
            jitter_ofs = (torch.rand_like(pixel_coords, device=device) - 0.5)
            pixel_coords = pixel_coords + jitter_ofs / subpixel_resolution

        pixel_coords = torch.cat([pixel_coords, torch.ones_like(pixel_coords[...,:1])], dim=-1)
        
        inv_proj = torch.inverse(proj_matrices)[:,:3,:3]

        ray_dir = torch.sum(inv_proj[:,None,:,:] * pixel_coords[...,None,:], dim=-1)
        ray_dir = ray_dir / torch.sqrt(torch.sum(ray_dir**2, dim=-1, keepdim=True) + 1e-6)

        ray_origin = torch.inverse(extrinsics)[:,:3,3][:,None,:].repeat(1,ray_dir.size(1),1)

        if True:#with torch.no_grad():
            x,n,is_valid, dist_coef = self.sphere_tracing(
                ray_origin.view(-1,3),
                ray_dir.view(-1,3),
                max_depth=max_depth
            )

            is_valid = is_valid.view(-1,1)

        #f, grad_f = self.forward(x)
        #gDg = torch.sum(grad_f.detach()**2, dim=-1, keepdim=True)
        #x = x - f * grad_f.detach() / (gDg + 1e-20)

        #grad_f = self.forward(x)[1]
        #n = grad_f / torch.sqrt(torch.sum(grad_f**2, dim=-1, keepdim=True) + 1e-9)

        #n = n * is_valid

        if False:
            f, grad_f = self.forward(x, mask=(is_valid.view(-1) > 0.0))
            with torch.no_grad():
                gDl = torch.sum(grad_f * ray_dir.view(-1,3), dim=1, keepdim=True)
                gDl = torch.clamp(gDl, None, 0) - 1e-2

            x = x - f / gDl * ray_dir.view(-1,3) * is_valid

            grad_f = self.forward(x, mask=(is_valid.view(-1) > 0.0))[1]
            n = grad_f / torch.sqrt(torch.sum(grad_f**2, dim=-1, keepdim=True) + 1e-5) * is_valid

        x = x.view(len(proj_matrices),-1,3)
        n = n.view(len(proj_matrices),-1,3)
        is_valid = is_valid.view(len(proj_matrices),-1,1)
        dist_coef = dist_coef.view(len(proj_matrices),-1,1)

        n = torch.sum(n[...,None,:] * extrinsics[:,None,:3,:3], dim=-1)
        n = n * torch.tensor([1,-1,-1], dtype=n.dtype, device=device)

        m_ = (proj_matrices[:,None,:3,:3] @ x[...,:,None])[...,0] + proj_matrices[:,None,:3,3]
        m = m_[...,:2] / m_[...,2:3]
        grid = 2 * m / torch.tensor([ref_imgs.size(-1), ref_imgs.size(-2)], dtype=m.dtype, device=m.device) - 1
        pixel_vals = torch.nn.functional.grid_sample(
            ref_imgs,
            grid[:,:,None,:],
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )[:,:,:,0].transpose(-1,-2) # num_view * num_pixels * 3
        pixel_vals = torch.clamp(pixel_vals, 0, None)
        ref_masks = torch.any(ref_imgs > 0.0, dim=1, keepdim=True).float()
        pixel_masks = torch.nn.functional.grid_sample(
            ref_masks,
            grid[:,:,None,:],
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )[:,:,:,0].transpose(-1,-2) # num_view * num_pixels * 1

        if True:
            is_valid = (pixel_masks > 0.99).float()


        recon_rmaps = self.img2rmap(
            pixel_vals.transpose(-1,-2)[...,:,:,None], 
            n.transpose(-1,-2)[...,:,:,None], 
            is_valid.transpose(-1,-2)[...,:,:,None],
            kappa = 1000,
            out_size = out_size,
            hemispherical = True,
            projection_mode = projection_mode,
            distortion_coef=dist_coef
        )

        refl_dir = ray_dir.view(len(n),-1,3) - 2 * n * torch.sum(n * ray_dir.view(len(n),-1,3), dim=-1, keepdim=True)
        recon_ref_rmaps = self.img2rmap(
            pixel_vals.transpose(-1,-2)[...,:,:,None], 
            refl_dir.transpose(-1,-2)[...,:,:,None], 
            is_valid.transpose(-1,-2)[...,:,:,None],
            kappa = 1000,
            out_size = out_size,
            projection_mode = projection_mode,
            distortion_coef=dist_coef
        )

        if True:
            rmap_weight = create_rmap_visibility_weight(out_size, projection_mode).to(recon_rmaps.device)[None,None]
            rmap_weight = rmap_weight / torch.max(rmap_weight)
            recon_rmaps = rmap_weight * recon_rmaps + (1 - rmap_weight) * recon_rmaps.detach()

            ref_rmap_weight = create_rmap_visibility_weight(out_size, projection_mode, reflection_map=True).to(recon_ref_rmaps.device)[None,None]
            ref_rmap_weight = ref_rmap_weight / torch.max(ref_rmap_weight)
            recon_ref_rmaps = ref_rmap_weight * recon_ref_rmaps + (1 - ref_rmap_weight) * recon_ref_rmaps.detach()

        return recon_rmaps, recon_ref_rmaps

    def recover_rmap_2(
        self, 
        ref_imgs, 
        proj_matrices, 
        extrinsics, 
        image_width=128, 
        image_height=128, 
        jitter=True, 
        max_depth=100.0,
        out_size=128,
        projection_mode = 'sphere',
        subpixel_resolution=3,
        grad_weighting=False,
        reparam_mode=None,
        kappa=1000,
    ):
        grid_length = (2 * self.range) / self.resolution
        ref_masks = torch.any(ref_imgs > 0.0, dim=1, keepdim=True).float()
        # dense sampling on SDF surface
        w,v,u = torch.meshgrid(torch.arange(self.resolution),torch.arange(self.resolution),torch.arange(self.resolution))
        x = self.range * (2 * (u.float() + 0.5) / self.resolution - 1)
        y = self.range * (2 * (v.float() + 0.5) / self.resolution - 1)
        z = self.range * (2 * (w.float() + 0.5) / self.resolution - 1)
        grid_coords = torch.stack([x,y,z], dim=-1).to(self.grid.device)
        if jitter:
            jitter_ofs = (torch.rand_like(grid_coords, device=grid_coords.device) - 0.5)
            grid_coords = grid_coords + jitter_ofs * grid_length
        grids_on_surface = torch.abs(self.grid.detach()) < (2 * grid_length)
        surf_points = grid_coords[grids_on_surface] # N * 3
        with torch.no_grad():
            f, grad_f = self.forward(surf_points)
            surf_points = surf_points - f * grad_f

        recon_rmaps = []
        recon_ref_rmaps = []
        for idx_view in range(extrinsics.size(0)):
            # visibility
            ray_origin = torch.inverse(extrinsics)[idx_view,:3,3][None,:].repeat(surf_points.size(0),1)
            ray_direction = surf_points - ray_origin
            ray_direction = ray_direction / torch.sqrt(torch.sum(ray_direction**2, dim=-1, keepdim=True))
            surf_points_st, surf_normal_st, st_is_valid, dist_coef_st = self.sphere_tracing(ray_origin, ray_direction)
            mask_vis = (torch.sum((surf_points_st - surf_points)**2, dim=-1) < (2*grid_length)) * (st_is_valid > 0)

            surf_points_vis = surf_points_st[mask_vis]
            normal_vis = surf_normal_st[mask_vis]
            ray_direction_vis = surf_points_vis - ray_origin[mask_vis]
            dist_coef_vis = dist_coef_st[mask_vis]

            # eval surface normal
            if reparam_mode is None:
                surf_points_vis = surf_points_vis.detach()
                ray_direction_vis = ray_direction_vis.detach()
                normal_vis = self.forward(surf_points_vis)[1] # N * 3
                dist_coef_vis = dist_coef_vis.detach()
            else:
                assert reparam_mode == 'vicini'

            normal_vis = normal_vis / torch.sqrt(torch.sum(normal_vis**2, dim=-1, keepdim=True) + 1e-24)


            normal_vis = (extrinsics[idx_view,:3,:3] @ normal_vis.T).T
            normal_vis = normal_vis * torch.tensor([1,-1,-1], dtype=normal_vis.dtype, device=normal_vis.device)

            # sample projected coords
            m_h = (proj_matrices[idx_view][:3,:3] @ surf_points_vis.T + proj_matrices[idx_view][:3,3:4]).T
            grid2d_x = 2 * m_h[:,0] / m_h[:,2] / image_width - 1
            grid2d_y = 2 *  m_h[:,1] / m_h[:,2] / image_height - 1
            grid2d = torch.stack([grid2d_x, grid2d_y], dim=-1) # N * 2

            # eval pixel value
            pixel_val_vis = torch.clamp(F.grid_sample(
                ref_imgs[idx_view:idx_view+1],
                grid2d[None,None,:,:],
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            ), 0, None)[0,:,0,:].T # N * 3

            pixel_mask_vis = torch.nn.functional.grid_sample(
                ref_masks[idx_view:idx_view+1],
                grid2d[None,None,:,:],
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )[0,:,0,:].T # N * 3

            pixel_mask_vis = (pixel_mask_vis > 0.99).float()

            # sampling on the image plane
            pixel_mask_vis = (pixel_mask_vis > 0.99).float()

            normal_map, rendered_mask, surf_point_map = self.render(
                proj_matrices[idx_view:idx_view+1],
                extrinsics[idx_view:idx_view+1],
                image_width=image_width,
                image_height=image_height,
                subpixel_resolution=1
            )[2:5]

            mask = ref_masks[idx_view:idx_view+1] * (torch.sum(normal_map**2, dim=1, keepdim=True) > 0.8).float()
            ray_direction_map = surf_point_map - ray_origin[0][None,:,None,None]
            ray_direction_map = ray_direction_map / torch.sqrt(torch.sum(ray_direction_map**2, dim=1, keepdim=True) + 1e-20)

            pixel_val_img = ref_imgs[idx_view:idx_view+1].view(3,image_height*image_width).transpose(-1,-2)
            normal_img = normal_map[idx_view:idx_view+1].view(3,image_height*image_width).transpose(-1,-2)
            pixel_mask_img = mask.view(1,image_height*image_width).transpose(-1,-2)
            ray_direction_img = ray_direction_map.view(3,image_height*image_width).transpose(-1,-2)

            pixel_val_vis = torch.cat([pixel_val_vis, pixel_val_img], dim=0)
            normal_vis = torch.cat([normal_vis, normal_img], dim=0)
            pixel_mask_vis = torch.cat([pixel_mask_vis, pixel_mask_img], dim=0)
            ray_direction_vis = torch.cat([ray_direction_vis, ray_direction_img], dim=0)

            # interpolation
            recon_rmap = self.img2rmap(
                pixel_val_vis.transpose(-1,-2)[None,:,:,None], 
                normal_vis.transpose(-1,-2)[None,:,:,None], 
                pixel_mask_vis.transpose(-1,-2)[None,:,:,None],
                kappa = kappa,
                out_size = out_size,
                hemispherical = True,
                projection_mode = projection_mode,
                #distortion_coef=dist_coef_vis.view(1,-1,1)
            )
            recon_rmaps.append(recon_rmap)

            reflection_direction = ray_direction_vis - 2 * normal_vis * torch.sum(ray_direction_vis * normal_vis, dim=-1, keepdim=True)
            recon_ref_rmap = self.img2rmap(
                pixel_val_vis.transpose(-1,-2)[None,:,:,None],
                reflection_direction.transpose(-1,-2)[None,:,:,None], 
                pixel_mask_vis.transpose(-1,-2)[None,:,:,None],
                kappa = kappa,
                out_size = out_size,
                projection_mode = projection_mode,
                #distortion_coef=dist_coef_vis.view(1,-1,1)
            )
            recon_ref_rmaps.append(recon_ref_rmap)
        recon_rmaps = torch.cat(recon_rmaps, dim=0)
        recon_ref_rmaps = torch.cat(recon_ref_rmaps, dim=0)

        if grad_weighting:
            rmap_weight = create_rmap_visibility_weight(out_size, projection_mode).to(recon_rmaps.device)[None,None]
            rmap_weight = rmap_weight / torch.max(rmap_weight)
            recon_rmaps = rmap_weight * recon_rmaps + (1 - rmap_weight) * recon_rmaps.detach()

            ref_rmap_weight = create_rmap_visibility_weight(out_size, projection_mode, reflection_map=True).to(recon_ref_rmaps.device)[None,None]
            ref_rmap_weight = ref_rmap_weight / torch.max(ref_rmap_weight)
            recon_ref_rmaps = ref_rmap_weight * recon_ref_rmaps + (1 - ref_rmap_weight) * recon_ref_rmaps.detach()

        return recon_rmaps, recon_ref_rmaps

    def render_gradient(
        self, 
        proj_matrices, 
        extrinsics, 
        image_width=128, 
        image_height=128, 
        subpixel_resolution=3, 
        max_depth=100.0, 
        jitter=False        
    ):
        def gradient_shader(x, n, v, proj_matrices):
            res = self.sample_grid_grad(x.view(-1,3))
            return res.view(x.size(0),x.size(1),-1).repeat(1,1,3)

        return self.render(
            proj_matrices, 
            extrinsics, 
            image_width, 
            image_height, 
            max_depth=max_depth, 
            subpixel_resolution=subpixel_resolution, 
            jitter=jitter,
            shading_function=gradient_shader
        )

laplacian_filter = torch.nn.Conv3d(
    in_channels=1, 
    out_channels=1, 
    kernel_size=3, 
    stride=1, 
    padding=0, 
    bias=False
)
laplacian_filter.weight.requires_grad = False
grad_filter = torch.nn.Conv3d(
    in_channels=1, 
    out_channels=3, 
    kernel_size=3, 
    stride=1, 
    padding=0, 
    bias=False
)
grad_filter.weight.requires_grad = False
for i in range(3):
    for j in range(3):
        for k in range(3):
            d = abs(i - 1) + abs(j - 1) + abs(k - 1)
            if d == 0:
                laplacian_filter.weight[0,0,i,j,k] = -6.0
                for c in range(3):
                    grad_filter.weight[c,0,i,j,k] = 0.0
            elif d == 1:
                laplacian_filter.weight[0,0,i,j,k] = 1.0

                grad_filter.weight[0,0,i,j,k] = 0.5 * (1.0 if i > 1 else -1.0 if i < 1 else 0.0)
                grad_filter.weight[1,0,i,j,k] = 0.5 * (1.0 if j > 1 else -1.0 if j < 1 else 0.0)
                grad_filter.weight[2,0,i,j,k] = 0.5 * (1.0 if k > 1 else -1.0 if k < 1 else 0.0)
            else:
                laplacian_filter.weight[0,0,i,j,k] = 0.0

                for c in range(3):
                    grad_filter.weight[c,0,i,j,k] = 0.0


# grid: D*H*W
def compute_discrete_laplacian_3d(grid):
    global laplacian_filter
    laplacian_filter = laplacian_filter.to(grid.device)
    return laplacian_filter(grid[None,None])[0,0]
    
def compute_discrete_grad_3d(grid):
    global grad_filter
    grad_filter = grad_filter.to(grid.device)
    return grad_filter(grid[None,None])[0]


# depth grid
@cuda.jit
def compute_depth_value_and_grad_forward_kernel(
    # inputs
    grid, x,
    # parameters
    grid_range,
    # sampling parameters
    out_value, out_grad
):
    grid_resolution = grid.shape[-1]

    idx_sample = cuda.blockIdx.x

    idx_thread = cuda.threadIdx.x
    blockdim = cuda.blockDim.x

    x = x[idx_sample]

    u = 0.5 * (x[0] / grid_range + 1) * grid_resolution
    v = 0.5 * (x[1] / grid_range + 1) * grid_resolution

    u = clip_device(u, 1, grid_resolution - 1 - 1e-4)
    v = clip_device(v, 1, grid_resolution - 1 - 1e-4)

    depth_value_tmp = cuda.shared.array(shape=(128,), dtype=numba.float32)
    depth_grad_tmp = cuda.shared.array(shape=(128,2), dtype=numba.float32)

    depth_value_tmp[idx_thread] = 0.0
    depth_grad_tmp[idx_thread][0] = depth_grad_tmp[idx_thread][1] = 0.0

    idx_ofs = idx_thread
    while idx_ofs < (basis_radius)**2:
                ofs_u = (idx_ofs % basis_radius) - 2 * basis_scale
                ofs_v = ((idx_ofs // basis_radius) % basis_radius) - 2 * basis_scale

                u_err = u - (int(u) + ofs_u + 0.5)
                v_err = v - (int(v) + ofs_v + 0.5)

                wu = compute_weight_device(u_err / basis_scale) / basis_scale
                wv = compute_weight_device(v_err / basis_scale) / basis_scale

                if (wu == 0) or (wv == 0):
                    pass
                else:

                    val = grid[
                        clip_device(int(v) + ofs_v, 0, grid_resolution-1), 
                        clip_device(int(u) + ofs_u, 0, grid_resolution-1)
                    ]

                    depth_value_tmp[idx_thread] += wu * wv * val

                    grad_u = compute_grad_weight_device(u_err / basis_scale) / (basis_scale**2)
                    grad_v = compute_grad_weight_device(v_err / basis_scale) / (basis_scale**2)

                    depth_grad_tmp[idx_thread][0] += wv * val * grad_u
                    depth_grad_tmp[idx_thread][1] += wu * val * grad_v

                idx_ofs += blockdim

    cuda.syncthreads()
    num_threads_valid = blockdim
    while num_threads_valid > 1:
        curr_ofs = ((num_threads_valid + 1) // 2)
        if idx_thread < curr_ofs:
            if (idx_thread + curr_ofs) < num_threads_valid:
                j = idx_thread + curr_ofs
                depth_value_tmp[idx_thread] += depth_value_tmp[j]
                depth_grad_tmp[idx_thread,0] += depth_grad_tmp[j,0]
                depth_grad_tmp[idx_thread,1] += depth_grad_tmp[j,1]
        num_threads_valid = curr_ofs
        cuda.syncthreads()

    c = 0.5 * grid_resolution / grid_range # gradient of uvw w.r.t. x
    if idx_thread == 0:
        out_value[idx_sample,0] = depth_value_tmp[0]
        out_grad[idx_sample,0] = depth_grad_tmp[0][0] * c
        out_grad[idx_sample,1] = depth_grad_tmp[0][1] * c

@cuda.jit
def compute_depth_value_and_grad_backward_kernel(
    # inputs
    grid, x, grad_out_value, grad_out_grad,
    # parameters
    grid_range,
    # sampling parameters
    grad_grid, grad_x
):
    grid_resolution = grid.shape[0]

    idx_sample = cuda.blockIdx.x

    idx_thread = cuda.threadIdx.x
    blockdim = cuda.blockDim.x

    x = x[idx_sample]
    grad_out_value = grad_out_value[idx_sample,0]
    grad_out_grad = grad_out_grad[idx_sample]

    grad_x_tmp = cuda.shared.array(shape=(125,2), dtype=numba.float32)
    grad_x_tmp[idx_thread][0] = grad_x_tmp[idx_thread][1] = 0.0

    if (grad_out_value == 0) and (grad_out_grad[0] == 0) and (grad_out_grad[1] == 0):
        pass
    else:
        u = 0.5 * (x[0] / grid_range + 1) * grid_resolution
        v = 0.5 * (x[1] / grid_range + 1) * grid_resolution

        u = clip_device(u, 1, grid_resolution - 1 - 1e-4)
        v = clip_device(v, 1, grid_resolution - 1 - 1e-4)

        idx_ofs = idx_thread
        while idx_ofs < (basis_radius)**2:
                    ofs_u = (idx_ofs % basis_radius) - 2 * basis_scale
                    ofs_v = ((idx_ofs // basis_radius) % basis_radius) - 2 * basis_scale

                    u_err = u - (int(u) + ofs_u + 0.5)
                    v_err = v - (int(v) + ofs_v + 0.5)

                    wu = compute_weight_device(u_err / basis_scale) / basis_scale
                    wv = compute_weight_device(v_err / basis_scale) / basis_scale

                    if (wu == 0) or (wv == 0):
                        pass
                    else:

                        val = grid[
                            clip_device(int(v) + ofs_v, 0, grid_resolution-1), 
                            clip_device(int(u) + ofs_u, 0, grid_resolution-1)
                        ]

                        grad_u = compute_grad_weight_device(u_err / basis_scale) / (basis_scale**2)
                        grad_v = compute_grad_weight_device(v_err / basis_scale) / (basis_scale**2)

                        c = 0.5 * grid_resolution / grid_range

                        grad_val = 0.0
                        if grad_out_value != 0.0:
                            grad_val += wu * wv * grad_out_value

                            grad_x_tmp[idx_thread][0] += wv * val * grad_u * c * grad_out_value
                            grad_x_tmp[idx_thread][1] += wu * val * grad_v * c * grad_out_value

                        if (grad_out_grad[0] != 0) or (grad_out_grad[1] != 0) or (grad_out_grad[2] != 0):
                            grad_val += wv * grad_u * c * grad_out_grad[0]
                            grad_val += wu * grad_v * c * grad_out_grad[1]

                            grad2_u = compute_grad_grad_weight_device(u_err / basis_scale) / (basis_scale**3)
                            grad2_v = compute_grad_grad_weight_device(v_err / basis_scale) / (basis_scale**2)

                            grad_x_tmp[idx_thread][0] += grad2_u * wv * val * c**2 * grad_out_grad[0]
                            grad_x_tmp[idx_thread][0] += grad_u * grad_v * val * c**2 * grad_out_grad[1]

                            grad_x_tmp[idx_thread][1] += grad_u * grad_v * val * c**2 * grad_out_grad[0]
                            grad_x_tmp[idx_thread][1] += wu * grad2_v * val * c**2 * grad_out_grad[1]

                        if grad_val != 0:
                            cuda.atomic.add(grad_grid, (
                                clip_device(int(v) + ofs_v, 0, grid_resolution-1), 
                                clip_device(int(u) + ofs_u, 0, grid_resolution-1)
                            ), grad_val)

                    idx_ofs += blockdim

    cuda.syncthreads()
    num_threads_valid = blockdim
    while num_threads_valid > 1:
        curr_ofs = ((num_threads_valid + 1) // 2)
        if idx_thread < curr_ofs:
            if (idx_thread + curr_ofs) < num_threads_valid:
                j = idx_thread + curr_ofs
                grad_x_tmp[idx_thread,0] += grad_x_tmp[j,0]
                grad_x_tmp[idx_thread,1] += grad_x_tmp[j,1]
        num_threads_valid = curr_ofs
        cuda.syncthreads()

    if idx_thread == 0:
        #for j in range(1,blockdim):
        #    grad_x_tmp[0,0] += grad_x_tmp[j,0]
        #    grad_x_tmp[0,1] += grad_x_tmp[j,1]
        #    grad_x_tmp[0,2] += grad_x_tmp[j,2]
        grad_x[idx_sample,0] = grad_x_tmp[0][0]
        grad_x[idx_sample,1] = grad_x_tmp[0][1]

class b_spline_interpolation_2d(torch.autograd.Function):
    # grid : S * S
    # x    : N * 2
    @staticmethod
    def forward(ctx, grid, x, grid_range):
        dtype = x.dtype
        device = x.device

        depth_value = torch.zeros_like(x[:,0:1], dtype=dtype, device=device)
        depth_grad = torch.zeros_like(x, dtype=dtype, device=device)

        compute_depth_value_and_grad_forward_kernel[(x.size(0),),(25,)](
            # inputs
            grid.detach(), x.detach(),
            # parameters
            grid_range.item(),
            # sampling parameters
            depth_value, depth_grad
        )

        ctx.save_for_backward(
            grid, x, grid_range
        )
        return depth_value, depth_grad

    @staticmethod
    def backward(ctx, grad_out_value, grad_out_grad):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grid, x, grid_range, = ctx.saved_tensors

        dtype = x.dtype
        device = x.device

        grad_grid = torch.zeros_like(grid, dtype=dtype, device=device)
        grad_x = torch.zeros_like(x, dtype=dtype, device=device)

        compute_depth_value_and_grad_backward_kernel[(x.size(0),),(25,)](
            # inputs
            grid.detach(), x.detach(), grad_out_value.detach(), grad_out_grad.detach(),
            # parameters
            grid_range.item(),
            # sampling parameters
            grad_grid, grad_x
        )

        return grad_grid, grad_x, None
    
class DepthGrid(nn.Module):
    def __init__(self, resolution=128, range=1.0, initial_radius=1):
        super(DepthGrid, self).__init__()

        self.range = range
        self.resolution = resolution
        self.threshold = 1e-4

        self.grid = nn.Parameter(torch.zeros(
            (resolution, resolution), 
            dtype=torch.float32,
            requires_grad=True
        ))

        self.grid_sample = b_spline_interpolation_2d.apply

    def resize_grid(self, resolution):
        self.resolution = resolution
        with torch.no_grad():
            new_grid = F.interpolate(self.grid[None,None], resolution, mode='bilinear')[0,0]
            new_grid.requires_grad=True
        self.grid = nn.Parameter(new_grid)

    def compute_weight(self, t):
        #return torch.clamp(1-torch.abs(t),0,1)
        t = 2 - torch.clamp(torch.abs(t), 0, 2)
        b1 = 1 / 6.0 * t**3
        m1 = (t < 1.0).float()
        u = t - 1.0
        b2 = 1 / 6.0 * (-3 * u**3 + 3 * u**2 + 3 * u + 1)
        m2 = (t >= 1.0).float()
        return b1 * m1 + b2 * m2

    def compute_grad_weight(self, t):
        #return torch.sgn(t) * (torch.abs(t) < 1.0).float()
        sgn = -torch.sgn(t)
        t = 2 - torch.clamp(torch.abs(t), 0, 2)
        gb1 = 1 / 2.0 * t**2
        m1 = (t < 1.0).float()
        u = t - 1.0
        gb2 = 1 / 2.0 * (-3.0 * u**2 + 2.0 * u + 1.0)
        m2 = (t >= 1.0).float()

        return sgn * (gb1 * m1 + gb2 * m2)

    # x: N * 2
    # is_valid: N
    # out: N * 1, N * 2
    def forward(self, x, backward_to_grid=True, mask=None):
        if not (mask is None):
            out_val = torch.zeros_like(x[...,:1], dtype=x.dtype, device=x.device)
            out_grad = torch.zeros_like(x, dtype=x.dtype, device=x.device)
            val, grad = self.forward(x[mask], backward_to_grid=backward_to_grid)
            out_val[mask] = val
            out_grad[mask] = grad
            return out_val, out_grad

        grid = self.grid
        if not backward_to_grid:
            grid = grid.detach()

        if x.is_cuda:
            return self.grid_sample(grid, x, torch.tensor(self.range))

        u = 0.5 * (x[:,0] / self.range + 1) * self.resolution
        v = 0.5 * (x[:,1] / self.range + 1) * self.resolution

        u = torch.clamp(u, 1, self.resolution - 1 - 1e-4)
        v = torch.clamp(v, 1, self.resolution - 1 - 1e-4)


        sdf_value = torch.zeros_like(u, dtype=x.dtype, device=x.device)
        sdf_grad = torch.zeros_like(x, dtype=x.dtype, device=x.device)
        for ofs_v in [-2,-1,0,1,2]:
            for ofs_u in [-2,-1,0,1,2]:
                val = grid[
                    torch.clamp(v.long() + ofs_v, 0, self.resolution-1), 
                    torch.clamp(u.long() + ofs_u, 0, self.resolution-1)
                ]
                wu = self.compute_weight(u - (u.int() + ofs_u + 0.5))
                wv = self.compute_weight(v - (v.int() + ofs_v + 0.5))
                sdf_value = sdf_value + wu * wv * val

                grad_u = self.compute_grad_weight(u - (u.int() + ofs_u + 0.5))
                grad_v = self.compute_grad_weight(v - (v.int() + ofs_v + 0.5))

                c = 0.5 * self.resolution / self.range
                sdf_grad[:,0] = sdf_grad[:,0] + wv * val * grad_u * c
                sdf_grad[:,1] = sdf_grad[:,1] + wu * val * grad_v * c

        return sdf_value[:,None], sdf_grad
    

    def get_grid_coords(self, upsampling_ratio=1, resolution=None, device=None):
        if np.isscalar(resolution):
            resolution = (resolution, resolution)
        elif resolution is None:
            resolution = (
                int(upsampling_ratio * self.grid.size(1)),
                int(upsampling_ratio * self.grid.size(0)),
            )

        v,u = torch.meshgrid(
            torch.arange(resolution[1]),
            torch.arange(resolution[0])
        )

        x = (2 * (u + 0.5) / resolution[0] - 1) * self.range
        y = -(2 * (v + 0.5) / resolution[1] - 1) * self.range

        if device is None:
            device = self.grid.device

        return torch.stack([x,y], dim=-1).to(device)

    def create_depth_and_normal_maps(self, resolution=256):
        x = self.get_grid_coords(resolution=resolution).view(-1,2)
        d , grad_d = self.forward(x)
        depth_map = d.view(1,1,resolution,resolution)
        p,q = grad_d.unbind(-1)

        l = torch.sqrt(p**2+q**2+1)
        normal_map = torch.stack([
            p / l,
            q / l,
            1 / l
        ], dim=0).view(1,3,resolution,resolution)

        return depth_map, normal_map
    
    def save_mesh(self, out_file, resolution=256, mask=None, use_rangegrid_format=False):
        with torch.no_grad():
            depth_map, normal_map = self.create_depth_and_normal_maps(resolution)
            x_ = self.get_grid_coords(resolution=resolution).view(-1,2)
            x = torch.cat([x_,-depth_map.view(-1,1)], dim=-1) # N*3
            n = normal_map.view(3,-1).transpose(0,1) # N*3
        faces = []
        if not (mask is None):
            mask = mask.view(-1).cpu().numpy()

        if use_rangegrid_format:
            x = torch.cat([x[v*resolution:(v+1)*resolution] for v in range(resolution-1,-1,-1)], dim=0)
            mask = np.concatenate([mask[v*resolution:(v+1)*resolution] for v in range(resolution-1,-1,-1)], axis=0)

            verts = torch.stack([x[i] for i in range(resolution**2) if (mask[i] > 0)], dim=0).cpu().numpy()
            vert_exists = [int(bool(mask[i])) for i in range(resolution**2)]
            vert_indices = np.cumsum(vert_exists, dtype=int) - 1
            lines = [
                'ply',
                'format ascii 1.0',
                'obj_info is_cyberware_data 1',
                'obj_info is_mesh 0',
                'obj_info is_warped 0',
                'obj_info is_interlaced 1',
                'obj_info num_cols '+str(resolution),
                'obj_info num_rows '+str(resolution),
                'element vertex '+str(len(verts)),
                'property float x',
                'property float y',
                'property float z',
                'element range_grid '+str(resolution**2),
                'property list uchar int vertex_indices',
                'end_header',
            ]
            for vert in verts:
                lines.append(' '.join([str(val) for val in vert]))
            for i in range(resolution**2):
                if vert_exists[i] == 0:
                    lines.append('0')
                else:
                    lines.append('1 '+str(vert_indices[i]))
            with open(out_file, 'w') as f:
                f.write('\n'.join(lines))
            return

        for v in range(resolution-1):
            for u in range(resolution-1):
                i1 = v * resolution + u
                i2 = v * resolution + (u + 1)
                i3 = (v + 1) * resolution + u
                i4 = (v + 1) * resolution + (u + 1)
                if not (mask is None):
                    if (mask[i1] * mask[i2] * mask[i3] * mask[i4]) == 0:
                        continue
                if mask[i1] * mask[i2] * mask[i3] > 0:
                    faces.append([i1,i3,i2])
                if mask[i2] * mask[i3] * mask[i4] > 0:
                    faces.append([i3,i4,i2])
        faces = np.array(faces)

        mesh = trimesh.Trimesh(vertices=x.cpu().numpy(), faces=faces)
        mesh.export(out_file)
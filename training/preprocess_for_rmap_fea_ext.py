import os
import sys
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from core.dataset import nLMVSSynthDataset
from core.sfs_utils import *
from core.brdf_utils import compute_brdf_reflectance, LoadMERL

import numpy as np
import cv2
import matplotlib.pyplot as plt

import os

from glob import glob
from tqdm import tqdm

np.random.seed(16)

def sample_rmap(rmap, normal_map, projection_mode = 'stereographic'):
    if projection_mode == 'stereographic':
        p = normal_map[:,0] / (1 + normal_map[:,2])
        q = normal_map[:,1] / (1 + normal_map[:,2])
        u = 2 * (0.5 + 0.25 * p) - 1 
        v = 2 * (0.5 - 0.25 * q) - 1
    elif projection_mode == 'sphere':
        r = torch.sqrt(torch.sum(normal_map**2, dim=1) + 1e-4)
        u = normal_map[:,0] / r
        v = -normal_map[:,1] / r

    grid = torch.stack([u,v], dim=-1)

    img = F.grid_sample(rmap, grid, mode='bicubic', padding_mode='border', align_corners=False)
    mask = (torch.sum(normal_map**2, dim=1, keepdim=True) > (0.25**2)).float()
    return img * mask

device = torch.device("cpu")
numdepth = 192
out_dir = os.environ['HOME']+'/data/tmp/rmap-fea-ext'
os.makedirs(out_dir, exist_ok=True)

# dataset settings
dataset_options = {
    'num_neighbors': 9,
    'use_crop': True,
    'img_size': (128,128),
    'rmap_mode': 'probe',
    'rmap_size': (256,256),
    'mask_img': True,
    #'use_diffuse_illum': True,
    #'use_brdf_code': True,
    'use_diffuse_img': True,
    'use_diffuse_img_r05': True,
    'use_diffuse_rmap': True,
}
dataset = nLMVSSynthDataset(os.environ['HOME']+'/data/mvs_train/rendered', **dataset_options)

brdf_files = sorted(glob('/home/kyamashita/data/BRDF/synthetic-brdf/brdfs/*.binary'))

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

with torch.no_grad():
    bar = tqdm(dataloader)
    bar.set_description('preprocessing')
    for idx_minbatch, minbatch in enumerate(bar):
        imgs = minbatch['hdr_images']# [BS*C*H*W]
        #gt_diffuse_imgs = minbatch['hdr_diffuse_images_r05'] # [BS*C*H*W]
        gt_rmaps = minbatch['hdr_rmaps']
        #gt_diffuse_rmaps = minbatch['hdr_diffuse_rmaps']

        #masks = minbatch['masks']
        #intrinsics = minbatch['intrinsics']
        extrinsics = minbatch['extrinsics']
        proj_matrices = minbatch['proj_matrices']
        #depth_ranges = minbatch['depth_ranges']

        #gt_diffuse_illum_map = minbatch['diffuse_illum_map'] * minbatch['diffuse_reflectance'][:,:,None,None]
        #gt_diffuse_illum_map = F.interpolate(gt_diffuse_illum_map, scale_factor=0.25, mode='area')

        gt_depths = minbatch['gt_depths']
        gt_normals = minbatch['gt_normals']
        gt_normals = gt_normals / torch.sqrt(torch.sum(gt_normals**2, dim=2, keepdim=True) + 1e-9)

        # adjust scale
        brdf = LoadMERL(brdf_files[idx_minbatch // 10])
        reflectance = compute_brdf_reflectance(brdf)
        imgs = 0.5 * imgs / reflectance[:,:,None,None]
        gt_rmaps = 0.5 * gt_rmaps / reflectance[:,:,None,None]
        #gt_diffuse_rmaps = 0.5 * gt_diffuse_rmaps


        rot_matrices = extrinsics[:,:,:3,:3]
        #depth_values = get_depth_values(
        #    torch.mean(depth_ranges[:,0], dim=1), 
        #    imgs.size()[3:5], 
        #    intrinsics[:,0], 
        #    numdepth=numdepth
        #)

        idx_scan = minbatch['idx_scan']

        rmap1 = gt_rmaps[:,0,:,64:192,64:192]
        rot1 = rot_matrices[:,0]
        img1 = imgs[:,0]
        normal1 = gt_normals[:,0]
        depth1 = gt_depths[:,0]

        idx_src = np.random.randint(1,gt_rmaps.size(1))
        rmap2 = gt_rmaps[:,idx_src,:,64:192,64:192]
        rot2 = rot_matrices[:,idx_src]
        img2 = imgs[:,idx_src]
        normal2 = gt_normals[:,idx_src]
        depth2 = gt_depths[:,idx_src]

        P12 = proj_matrices[:,idx_src] @ torch.inverse(proj_matrices[:,0])
        R12 = (extrinsics[:,idx_src] @ torch.inverse(extrinsics[:,0]))[:,:3,:3]

        # compute gt normal map matching
        depth_threshold = 0.01
        normal_cosine_threshold = 0.95
        v, u = torch.meshgrid(torch.arange(img1.size(-2)), torch.arange(img1.size(-1)))
        m_ref = torch.stack([u,v,torch.ones_like(u)], dim=0)[None] * depth1 # BS*3*H*W
        n_ref_ = normal1 * torch.tensor([1.,-1.,-1.])[:,None,None]
        m_src_ = torch.sum(P12[:,:3,:3,None,None] * m_ref[:,None,:,:,:], dim=2) + P12[:,:3,3,None,None]
        d_src = m_src_[:,2:3] # BS*1*H*W
        m_src = m_src_[:,:2] / d_src
        n_src = torch.sum(R12[:,:,:,None,None] * n_ref_[:,None,:,:,:], dim=2) * torch.tensor([1.,-1.,-1.])[:,None,None]
        d_src_sampled = F.grid_sample(
            depth2,
            torch.stack([
                2 * m_src[:,0] / img1.size(-1) - 1.,
                2 * m_src[:,1] / img1.size(-2) - 1.,
            ], dim=-1),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )
        n_src_sampled = F.grid_sample(
            normal2,
            torch.stack([
                2 * m_src[:,0] / img1.size(-1) - 1.,
                2 * m_src[:,1] / img1.size(-2) - 1.,
            ], dim=-1),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )
        n_src_sampled = n_src_sampled / torch.sqrt(torch.sum(n_src_sampled**2, dim=1, keepdim=True) + 1e-4)
        vis_mask = (torch.abs(d_src_sampled - d_src) < depth_threshold)
        vis_mask *= (d_src_sampled > 0.) * (d_src > 0.)
        vis_mask *= torch.sum(n_src * n_src_sampled, dim=1, keepdim=True) > normal_cosine_threshold

        m_src = m_src * vis_mask.float() + (-1) * (1 - vis_mask.float())
        m_src = (m_src + 0.5).int()

        if False:
            plt.figure(figsize=(16,4))
            plt.subplot(1,4,1)
            plot_normal_map(normal1)
            plt.xlabel('Normal1')
            plt.subplot(1,4,2)
            plot_normal_map(normal2)
            plt.xlabel('Normal2')
            plt.subplot(1,4,3)
            plt.imshow(m_src[0,0])
            plt.xlabel('Corr u')
            plt.subplot(1,4,4)
            plt.imshow(m_src[0,1])
            plt.xlabel('Corr v')
            plt.show()




        if False:
            plt.subplot(1,2,1)
            plot_hdr(rmap1)
            plt.subplot(1,2,2)
            plot_hdr(rmap2)
            plt.show()


        data_dict = {
            'idx_scan': idx_scan[0],
            'img1': img1[0].clone().cpu(),
            'img2': img2[0].clone().cpu(),
            'normal1': normal1[0].clone().cpu(),
            'normal2': normal2[0].clone().cpu(),
            'corr_map': m_src[0].clone().cpu(),
            'rmap1': rmap1[0].clone().cpu(),
            'rmap2': rmap2[0].clone().cpu(),
            'rot1': rot1[0].clone().cpu(),
            'rot2': rot2[0].clone().cpu(),
        }

        torch.save(data_dict, out_dir+'/'+str(idx_minbatch).zfill(8)+'.pt')
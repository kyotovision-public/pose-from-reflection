import os
import sys
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
os.environ["MPLBACKEND"]="WebAgg"

os.environ['MKL_NUM_THREADS']='1'
os.environ['NUMEXPR_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset

from core.dataset import nLMVSSynthDataset, DrexelMultiNatGeomDataset, TwoViewRealImageDataset
from core.rm_net import ReflectanceMapNet
from core.sfs_models import SimpleSfSNet
from core.sfs_utils import *
from core.rmap_utils import sample_rmap
from core.brdf_utils import LoadMERL, compute_brdf_reflectance
from core.criterion import ImageLogL1Loss, ImageGradientLoss, VGG16PerceptualLoss, OccludingBoundaryLoss
from core.training_utils import normal_loss
from core.geometry import DepthGrid

import numpy as np
import matplotlib.pyplot as plt

import json

import sys
from tqdm import tqdm
import glob
import subprocess

import argparse
import json

torch.manual_seed(16)
np.random.seed(16)

def get_gpu_memory_size():
    cmd = '/usr/bin/nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits'
    mem_sizes = [int(s.strip()) for s in subprocess.check_output(cmd, shell=True).decode().split('\n') if s.strip() != '']
    return min(mem_sizes)

def compute_depth_error(est_depth, gt_depth, mask, diagonal_length):
    mask = mask * (gt_depth > 0.0).float()
    error_map = torch.abs(est_depth-gt_depth) * mask / diagonal_length
    return torch.sum(error_map) / torch.sum(mask)

def compute_log_l1_error(est_img, tar_img):
    mean_color = torch.mean(tar_img, dim=(2,3))
    est_img = torch.log1p(100 * est_img / mean_color[:,:,None,None])
    tar_img = torch.log1p(100 * tar_img / mean_color[:,:,None,None])
    error_map = torch.abs(est_img - tar_img)
    return torch.mean(error_map), error_map

# inspired by Johnson & Adelson
def compute_curl_loss(normal):
    mask = (torch.sum(normal**2, dim=1, keepdim=True) > 0.25).float()
    p = -normal[:,0:1] / (normal[:,2:3] + 1e-20)
    q = -normal[:,1:2] / (normal[:,2:3] + 1e-20)
    qu = torch.roll(q,-1,-1) - q
    pv = torch.roll(p, 1,-2) - p
    mask = mask * torch.roll(mask, -1, -1) * torch.roll(mask, 1, -2)
    curl_map = (qu - pv) * mask
    if False:
        plt.imshow(torch.abs(curl_map)[0,0].detach().cpu(), vmin=0, vmax=1)
        plt.show()
    return torch.sum(torch.abs(curl_map)) / torch.sum(mask)

grad_filter = torch.nn.Conv2d(
    in_channels=1, 
    out_channels=2, 
    kernel_size=3, 
    stride=1, 
    padding=1, 
    bias=False
)
grad_filter.weight.requires_grad = False
grad_filter.weight[:] = 0.0
grad_filter.weight[0,0,0,0] = grad_filter.weight[0,0,2,0] = 0.125
grad_filter.weight[0,0,1,0] = 0.25
grad_filter.weight[0,0,0,2] = grad_filter.weight[0,0,2,2] = -0.125
grad_filter.weight[0,0,1,2] = -0.25
grad_filter.weight[1,0,0,0] = grad_filter.weight[1,0,0,2] = -0.125
grad_filter.weight[1,0,0,1] = -0.25
grad_filter.weight[1,0,2,0] = grad_filter.weight[1,0,2,2] = 0.125
grad_filter.weight[1,0,2,1] = 0.25

#print(grad_filter.weight[0,0])
#print(grad_filter.weight[1,0])

def compute_boundary_loss(normal):
    # detect boundary pixels
    mask0 = (torch.sum(normal**2, dim=1, keepdim=True) > 0.25)
    mask_all = 1.0
    for dv in [-1,0,1]:
        for du in [-1,0,1]:
            mask_all = mask_all * torch.roll(torch.roll(mask0, du, -1), dv, -2)
    mask = ((mask0 == True) * (mask_all == False)).float()

    # compute boundary orientation
    global grad_filter
    grad_filter = grad_filter.to(normal.device)
    grad = grad_filter(mask0.float())
    grad = grad / torch.sqrt(torch.sum(grad**2, dim=1) + 1e-9)
    grad = torch.cat([grad, torch.zeros_like(grad[:,:1])], dim=1) * mask

    if False:
        plt.subplot(1,3,1)
        plt.imshow(mask0[0,0].cpu())
        plt.subplot(1,3,2)
        plt.imshow(mask[0,0].cpu())
        plt.subplot(1,3,3)
        plot_normal_map(grad)
        plt.show()

    angle_error = torch.acos(torch.clamp(torch.sum(grad * normal, dim=1, keepdim=True), -0.99999, 0.99999)) * mask 
    if False:
        plt.imshow(angle_error[0,0].detach().cpu())
        plt.show()
    return torch.sum(angle_error**2 * mask) / torch.sum(mask)

def compute_smootheness_loss(normal):
    mask = (torch.sum(normal**2, dim=1, keepdim=True) > 0.25).float()
    mask_any = 0
    error = 0.0
    for dv in [0,0,-1,1]:
        for du in [-1,1,0,0]:
            mask_ = torch.roll(torch.roll(mask, du, -1), dv, -2)
            normal_ = torch.roll(torch.roll(normal, du, -1), dv, -2)
            e = torch.acos(torch.clamp(torch.sum(normal * normal_, dim=1, keepdim=True), -0.99999, 0.99999))
            error = error + 0.25 * (e**2 * mask * mask_)
            mask_any = mask_any + mask_
    mask_any = (mask_any > 0).float()
    mask = mask * mask_any
    error = error * mask
    #plt.imshow(error[0,0].detach().cpu())
    #plt.show()
    return torch.sum(error * mask) / torch.sum(mask)

if True:
    parser = argparse.ArgumentParser()
    parser.add_argument('object_id', type=int)
    parser.add_argument('--config', default=f'{project_dir}/confs/test_depth_grid_nlmvss10.json')
    parser.add_argument('--depth-anything', action='store_true')
    #parser.add_argument('--wo-photo-cost', action='store_true')
    args = parser.parse_args()

    with open(args.config,'r') as f:
        confs = json.load(f)

    if not 'refine_verts' in confs:
        confs['refine_verts'] = False
    if not ('vhull_loss_weight' in confs['loss_confs']):
        confs['loss_confs']['vhull_loss_weight'] = 0
    if not ('shading_reg_loss_weight' in confs['loss_confs']):
        confs['loss_confs']['shading_reg_loss_weight'] = 0
    if not ('color_reg_loss_weight' in confs['loss_confs']):
        confs['loss_confs']['color_reg_loss_weight'] = 0
    if not ('view_start' in confs):
        confs['view_start'] = confs['view_skip'] // 2
    if not ('view_end' in confs):
        confs['view_end'] = 10000
    if not ('num_views_per_chunk' in confs):
        confs['num_views_per_chunk'] = 2# if (get_gpu_memory_size() < 32000) else 4
    if not ('use_nlmvs_sfsnet' in confs):
        confs['use_nlmvs_sfsnet'] = False
    if not ('reg_normal' in confs['loss_confs']):
        confs['loss_confs']['reg_normal'] = [0., 0., 6.9846964e-01]
        

    object_id = args.object_id
    confs['object_id'] = object_id
    max_mesh_subdivision_level = 2

    torch.manual_seed(8)
    np.random.seed(8)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_path = confs['dataset_path'] #os.environ['HOME']+'/data/mvs_eval/rendered'
    if confs['dataset_type'] == 'nlmvss':
        instance_dirs = sorted(glob.glob(dataset_path+'/?????'))
        subset_ofs = int(np.sum([len(glob.glob(instance_dirs[i]+'/images/*.exr')) for i in range(object_id)]))
        subset_len = len(glob.glob(instance_dirs[object_id]+'/images/*.exr'))

        #material_id = object_id % 6
        #shape_id = (object_id // 6) % 6
        #illum_id = object_id // (6 * 6)

        #brdf_files = sorted(glob.glob(dataset_path+'/../assets/material/*.binary'))
        #material_id = object_id % len(brdf_files)
        #brdf_file = brdf_files[material_id]
        brdf_file = None

        dataset = nLMVSSynthDataset(dataset_path, **confs['dataset_confs'])
    elif confs['dataset_type'] == 'nlmvsr':
        instance_dir = [f for f in sorted(glob.glob(dataset_path+'/*/*')) if os.path.isdir(f)][object_id]
        illum_name, shape_name = instance_dir.split('/')[-2:]
        subset_ofs = 0
        subset_len = len(glob.glob(instance_dir+'/view-??.exr'))

        brdf_file = None

        dataset = DrexelMultiNatGeomDataset(dataset_path, illum_name, shape_name, **confs['dataset_confs'])
    elif confs['dataset_type'] == 'real':
        dataset = TwoViewRealImageDataset(dataset_path, object_id, **confs['dataset_confs'])
        subset_ofs = 0
        subset_len = 2

        brdf_file = None
    else:
        print('Error: Invalid dataset type')
        exit()

    out_dir = './run/'+confs['exp_name']
    if args.depth_anything:
        out_dir += '_da'
    out_dir += '/'+str(object_id).zfill(3)
    os.makedirs(out_dir, exist_ok=True)

    with open(out_dir+'/confs.json', 'w') as f:
        json.dump(confs, f, indent=2)



    list_split = np.arange(len(dataset))
    test_subset_indices =  list_split[subset_ofs:subset_ofs+subset_len][int(confs['view_start']):int(confs['view_end']):int(confs['view_skip'])]
    test_dataset = Subset(dataset, test_subset_indices)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    reg_normal = torch.tensor(confs['loss_confs']['reg_normal'], device=device)
    
    # load model
    rm_net = ReflectanceMapNet()
    rm_net.load_state_dict(torch.load(project_dir+'/weights/rm-net-da/best.ckpt')['rm_net_state_dict'])
    for p in rm_net.parameters():
        p.requires_grad = False
    rm_net.eval()
    rm_net.to(device)

    sfsnet = SimpleSfSNet(wo_lambertian=True)
    weight_dir = project_dir+'/weights/simple-sfsnet-da'
    loss_files = glob.glob(weight_dir+'/???.ckpt')
    checkpoint_path = None
    min_val_loss = 1e12
    for f in loss_files:
        val_loss = torch.load(f)['val_loss']
        if val_loss < min_val_loss:
            checkpoint_path = f
            min_val_loss = val_loss
    checkpoint = torch.load(checkpoint_path)
    sfsnet.load_state_dict(checkpoint['sfsnet_state_dict'])
    print(checkpoint_path, 'loaded')
    for p in sfsnet.parameters():
        p.requires_grad = False
    sfsnet.eval()
    sfsnet.to(device)

    if 'reference_view_id' in confs:
        idx_ref_view = confs['reference_view_id']
        num_neighbiring_views = confs['num_neighboring_views']

        view_dirs = []
        for idx_img, minbatch in enumerate(testloader):
            view_dirs.append(minbatch['extrinsics'].to(device)[:,0,2,:3])
        view_dirs = torch.cat(view_dirs, dim=0)
        view_cosines = torch.sum(view_dirs[None,:,:] * view_dirs[:,None,:], dim=-1)
        neighboring_view_indices = torch.argsort(view_cosines[idx_ref_view],descending=True)[::2][:num_neighbiring_views+1]
        print('selected views:', neighboring_view_indices.cpu())
    else:
        neighboring_view_indices = None

    if args.depth_anything:
        from depth_anything_v2.dpt import DepthAnythingV2
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

        depth_model = DepthAnythingV2(**model_configs[encoder])
        depth_model.load_state_dict(torch.load(f'{project_dir}/pretrained-models/depth-anything-v2/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        depth_model = depth_model.to(DEVICE).eval()
        for p in depth_model.parameters():
            p.requires_grad = False

    # data loading
    bar = tqdm(testloader)
    bar.set_description('Data Loading')
    imgs = []
    masks = []
    soft_masks = []
    proj_matrices = []
    intrinsics = []
    extrinsics = []
    gt_rmaps = []
    gt_diffuse_rmaps = []
    gt_diffuse_imgs = []
    gt_depths = []
    gt_normals = []
    for idx_img, minbatch in enumerate(bar):
        view_id = idx_img# % num_images_per_object
        if not (neighboring_view_indices is None):
            if not (view_id in neighboring_view_indices):
                continue

        def load_if_exists(name):
            if name in minbatch:
                return minbatch[name].to(device)

        img = minbatch['hdr_images'].to(device)[:,0] # [BS*C*H*W]
        normal_map = minbatch['gt_normals'].to(device)[:,0] # [BS*3*H*W]
        gt_depth = minbatch['gt_depths'].to(device)[:,0] # [BS*1*H*W]

        gt_diffuse_img = minbatch['hdr_diffuse_images'].to(device)[:,0] if ('hdr_diffuse_images' in minbatch) else None # [BS*C*H*W]
        soft_mask = minbatch['masks'].to(device)[:,0] # [BS*1*H*W]
        gt_rmap = minbatch['hdr_rmaps'].to(device)[:,0] if ('hdr_rmaps' in minbatch) else None # [BS*N*1*H*W]
        gt_diffuse_rmap = minbatch['hdr_diffuse_rmaps'].to(device)[:,0] if ('hdr_diffuse_rmaps' in minbatch) else None # [BS*N*1*H*W]
        diffuse_reflectance = minbatch['diffuse_reflectance'].to(device) if ('diffuse_reflectance' in minbatch) else None

        proj_matrix = minbatch['proj_matrices'].to(device)[:,0] 
        intrinsic_matrix = minbatch['intrinsics'].to(device)[:,0] 
        extrinsic_matrix = minbatch['extrinsics'].to(device)[:,0] 

        normal_map = normal_map / torch.sqrt(torch.clamp(torch.sum(normal_map**2, dim=1, keepdim=True), 1e-6, None))
        gt_normal = normal_map

        mask = torch.any(img > 0, dim=1, keepdim=True).float()

        if not (brdf_file is None):
            brdf = LoadMERL(brdf_file).to(device)
            reflectance = compute_brdf_reflectance(brdf)

            img = img
            gt_rmap = gt_rmap
            if not (gt_diffuse_rmap is None):
                gt_diffuse_rmap = gt_diffuse_rmap * reflectance[:,:,None,None]
            if not (gt_diffuse_img is None):
                gt_diffuse_img = gt_diffuse_img / diffuse_reflectance[:,:,None,None] * reflectance[:,:,None,None]

        if not (gt_rmap is None):
            sampled_img = sample_rmap(gt_rmap, normal_map)
        if not (gt_diffuse_rmap is None):
            sampled_diffuse_img = sample_rmap(gt_diffuse_rmap, normal_map)

        imgs.append(img)
        masks.append(mask)
        soft_masks.append(soft_mask)
        proj_matrices.append(proj_matrix)
        intrinsics.append(intrinsic_matrix)
        extrinsics.append(extrinsic_matrix)
        gt_rmaps.append(gt_rmap)
        gt_diffuse_rmaps.append(gt_diffuse_rmap)
        gt_diffuse_imgs.append(gt_diffuse_img)
        gt_normals.append(gt_normal)
        gt_depths.append(gt_depth)

        #nf = torch.quantile(img[img > 0], 0.9)#torch.clamp(torch.max(gt_rmap[0]), None, 1)
        #save_hdr_as_ldr(out_dir+'/in_img.png', img / nf)
        #save_hdr_as_ldr(out_dir+'/gt_rmap.png', gt_rmap[...,64:192,64:192] / nf)
        #save_hdr_as_ldr(out_dir+'/gt_diffuse_rmap.png', gt_diffuse_rmap[...,64:192,64:192] / nf)
        #save_hdr_as_ldr(out_dir+'/gt_diffuse_img.png', gt_diffuse_img / nf)
        #save_normal_map(out_dir+'/gt_normal.png', gt_normal)

    world_scale = minbatch['world_scale'][0].item() if 'world_scale' in minbatch else 1.0
    world_offset = minbatch['world_offset'][0].numpy() if 'world_offset' in minbatch else [0.0, 0.0, 0.0]

    imgs = torch.cat(imgs, dim=0)
    masks = torch.cat(masks, dim=0)
    soft_masks = torch.cat(soft_masks, dim=0)
    proj_matrices = torch.cat(proj_matrices, dim=0)
    intrinsics = torch.cat(intrinsics, dim=0)
    extrinsics = torch.cat(extrinsics, dim=0)
    gt_rmaps = torch.cat(gt_rmaps, dim=0) if not (gt_rmaps[0] is None) else None
    gt_diffuse_rmaps = torch.cat(gt_diffuse_rmaps, dim=0) if not (gt_diffuse_rmaps[0] is None) else None
    gt_diffuse_imgs = torch.cat(gt_diffuse_imgs, dim=0) if not (gt_diffuse_imgs[0] is None) else None
    gt_normals = torch.cat(gt_normals, dim=0)
    gt_depths = torch.cat(gt_depths, dim=0)

    # normalize scale
    nf = torch.quantile(imgs[imgs > 0], 0.9)
    imgs = imgs / nf
    gt_rmaps = gt_rmaps / nf if not (gt_rmaps is None) else None
    gt_diffuse_rmaps = gt_diffuse_rmaps / nf if not (gt_diffuse_rmaps is None) else None
    gt_diffuse_imgs = gt_diffuse_imgs / nf if not (gt_diffuse_imgs is None) else None

    nf = torch.quantile(imgs[imgs > 0], 0.9)#torch.clamp(torch.max(gt_rmap[0]), None, 1)
    save_hdr_as_ldr(out_dir+'/in_img.png', torch.cat(imgs.unbind(0), dim=-1)[None] / nf)
    save_hdr(out_dir+'/in_img.exr', torch.cat(imgs.unbind(0), dim=-1)[None])
    if not (gt_rmaps is None):
        save_hdr_as_ldr(out_dir+'/gt_rmap.png', torch.cat(gt_rmaps[...,64:192,64:192].unbind(0), dim=-1)[None] / nf)
        save_hdr(out_dir+'/gt_rmap.exr', torch.cat(gt_rmaps[...,64:192,64:192].unbind(0), dim=-1)[None])
    if not (gt_diffuse_rmaps is None):
        save_hdr_as_ldr(out_dir+'/gt_diffuse_rmap.png', torch.cat(gt_diffuse_rmaps[...,64:192,64:192].unbind(0), dim=-1)[None] / nf)
        save_hdr(out_dir+'/gt_diffuse_rmap.exr', torch.cat(gt_diffuse_rmaps[...,64:192,64:192].unbind(0), dim=-1)[None])
    if not (gt_diffuse_imgs is None):
        save_hdr_as_ldr(out_dir+'/gt_diffuse_img.png', torch.cat(gt_diffuse_imgs.unbind(0), dim=-1)[None] / nf)
        save_hdr(out_dir+'/gt_diffuse_img.exr', torch.cat(gt_diffuse_imgs.unbind(0), dim=-1)[None])

    if not (gt_normals is None):
        save_normal_map(out_dir+'/gt_normal.png', torch.cat(gt_normals.unbind(0), dim=-1))
        torch.save(gt_normals.detach().cpu(), out_dir+'/gt_normal.pt')


    torch.save(extrinsics.cpu(), out_dir+'/gt_extrinsic_matrices.pt')
    torch.save(intrinsics.cpu(), out_dir+'/gt_intrinsic_matrices.pt')

    os.makedirs(out_dir+'/gt_geometry', exist_ok=True)
    for i in range(len(gt_depths)):
        gt_depth_grid = DepthGrid(gt_depths.size(-1))
        with torch.no_grad():
            focal_length_pix = intrinsics[i,0,0]
            gt_depth_ = torch.stack(gt_depths[i].unbind(-2)[::-1],-2)# * focal_length_pix / (2 * gt_depth_grid.range / gt_depths.size(-1))
            mask_ = masks[i] * (gt_depths[i] > 0).float()
            mask_ = mask_ * torch.roll(mask_,-1,-1) * torch.roll(mask_,1,-1)
            mask_ = mask_ * torch.roll(mask_,-1,1) * torch.roll(mask_,1,1)
            mean_depth = torch.sum(gt_depths[i] * mask_) / torch.sum(mask_)
            pixel_size_sc = 2 / gt_depths.size(-1) * gt_depth_grid.range
            pixel_size = 1. / focal_length_pix * mean_depth
            gt_depth_grid.grid[:] = (gt_depth_ - mean_depth) * pixel_size_sc / pixel_size
            gt_depth_grid.save_mesh(
                out_dir+'/gt_geometry/view-'+str(i).zfill(2)+'.ply',
                confs['dataset_confs']['img_size'][0],
                mask=mask_,
                use_rangegrid_format=True
            )

            gt_normal_rendered = gt_depth_grid.create_depth_and_normal_maps(imgs.size(-1))[1] * mask_.cpu()
            save_normal_map(   
                out_dir+'/gt_geometry/view-'+str(i).zfill(2)+'_n.png', 
                gt_normal_rendered
            )

    # initialize depth_grid
    initial_depth_grid_size = 16
    depth_grids = [DepthGrid(initial_depth_grid_size) for _ in range(len(imgs))]
    #depth_grid = torch.zeros((len(imgs),1,initial_depth_grid_size,initial_depth_grid_size), device=device, requires_grad=True)
    with torch.no_grad():
        v,u = torch.meshgrid(torch.arange(initial_depth_grid_size),torch.arange(initial_depth_grid_size))
        ny = -((v.to(device) + 0.5) / (initial_depth_grid_size / 2) - 1.0)
        nx = (u.to(device) + 0.5) / (initial_depth_grid_size / 2) - 1.0
        nz = torch.sqrt(torch.clamp(1 - nx**2 - ny**2,1e-2,1))
        #r = torch.sqrt(nx**2+ny**2+nz**2 + 1e-3)
        #nx /= r
        #ny /= r
        #nz /= r
        for i in range(len(imgs)):
            depth_grids[i].to(device)
            depth_grids[i].grid[:] = 0.5 - nz
    def save_mesh(out_dir):
        for i in range(len(depth_grids)):
            depth_grids[i].save_mesh(out_dir+'/view-'+str(i).zfill(2)+'.ply',confs['dataset_confs']['img_size'][0],mask=masks[i],use_rangegrid_format=True)
    os.makedirs(out_dir+'/initial_geometry', exist_ok=True)
    save_mesh(out_dir+'/initial_geometry')

    if args.depth_anything:
        bar = tqdm(range(len(imgs)))
        bar.set_description('depth anything v2')
        est_depths_da = []
        est_normals_da = []
        for idx_view in bar:
            img_np = torch.stack(imgs[idx_view].unbind(0), dim=-1).cpu().numpy()
            img_np = (255 * np.clip(img_np, 0, 1)**(1/2.2)).astype(np.uint8)
            depth_np = depth_model.infer_image(img_np[...,::-1])
            depth_np = np.max(depth_np) / depth_np
            mask_np = masks[idx_view,0].cpu().numpy()
            depth_np[mask_np == 0] = np.max(depth_np[mask_np > 0])
            da_depth = torch.from_numpy(depth_np).float().to(device)[None]
            est_depths_da.append(da_depth)

            depth_grid_ = DepthGrid(imgs.size(-1))
            depth_grid_.to(device)
            with torch.no_grad():
                da_depth_grid = torch.from_numpy(depth_np[::-1].copy()).float().to(device)
                da_depth_grid = F.interpolate(da_depth_grid[None,None], size=imgs.size(-1), mode='area')[0,0]
                depth_grid_.grid[:] = da_depth_grid
                est_normal_da = depth_grid_.create_depth_and_normal_maps(imgs.size(-1))[1]
            est_normals_da.append(est_normal_da)

        est_depths_da = torch.stack(est_depths_da, dim=0) * masks
        est_normals_da = torch.cat(est_normals_da, dim=0) * masks

        da_bl_params = torch.zeros((len(imgs),3), dtype=imgs.dtype, device=imgs.device, requires_grad=True)

        def correct_da_normals(est_normals_da, da_bl_params):
            log_lam, mu, nu = da_bl_params.unbind(-1)
            new_nx = est_normals_da[:,0] + mu[:,None,None] * est_normals_da[:,2]
            new_ny = est_normals_da[:,1] + nu[:,None,None] * est_normals_da[:,2]
            new_nz = log_lam.exp()[:,None,None] * est_normals_da[:,2]
            est_normals_da_new =  torch.stack([new_nx, new_ny, new_nz], dim=1)
            est_normals_da_new = est_normals_da_new / torch.sqrt(torch.sum(est_normals_da_new**2, dim=1, keepdim=True) + 1e-6)
            return est_normals_da_new

    bar = tqdm(range(4000))
    bar.set_description('Geometry Optimization')
    def setup_optimizer():
        list_params = []
        for depth_grid in depth_grids:
            list_params += list(depth_grid.parameters())
        return torch.optim.Adam(list_params, lr=1e-2 * (16/depth_grid.resolution), betas=(0.9, 0.99))
    optimizer = setup_optimizer()
    if args.depth_anything:
        optimizer_da = torch.optim.Adam([
            {'params': da_bl_params, 'lr': 1e-3},
        ])

    image_loss = ImageLogL1Loss()
    image_loss.to(device)
    image_grad_loss = ImageGradientLoss()
    image_grad_loss.to(device)
    perceptual_loss = VGG16PerceptualLoss()
    perceptual_loss.to(device)
    occluding_boundary_loss = OccludingBoundaryLoss()
    occluding_boundary_loss.to(device)
    #est_normal = dnp(mask=mask).detach()

    list_normal_error = []
    list_loss = []
    list_sfs_loss = []
    list_img_loss = []
    list_occ_loss = []
    list_normal_reg_loss = []
    idx_itr_ofs = 0
    best_loss = 1e20
    for idx_itr in bar:
        # create depth & normal maps
        with torch.no_grad():
            est_normals = []
            for i in range(len(imgs)):
                est_depth, est_normal = depth_grids[i].create_depth_and_normal_maps(imgs.size(-1))
                est_normals.append(est_normal)
            est_normals = torch.cat(est_normals, dim=0) * masks
            
            est_normals.requires_grad = True

            if args.depth_anything:
                est_normals_da_corrected = correct_da_normals(est_normals_da, da_bl_params)

        optimizer.zero_grad()
        if args.depth_anything:
            optimizer_da.zero_grad()

        num_views_per_chunk = confs['num_views_per_chunk']
        est_rmaps = []
        est_imgs = []
        est_normals_sfs = []
        loss=0.0
        sfs_loss = 0.0
        img_loss = 0.0
        occ_loss = 0.0
        normal_error = 0.0
        for idx_chunk in range((len(proj_matrices) - 1) // num_views_per_chunk + 1):
            soc = idx_chunk*num_views_per_chunk
            eoc = (idx_chunk+1)*num_views_per_chunk

            est_normals_chunk = est_normals[soc:eoc]

            rmap_result = rm_net(
                imgs[soc:eoc],
                est_normals_chunk.detach(),
                3
            )[-1]

            est_rmaps_chunk = rmap_result['rmap']
            #est_rmaps_wo_mask_chunk = rmap_result['rmap_wo_mask']
            #est_shadow_masks_chunk = rmap_result['est_mask']
            est_imgs_chunk = rmap_result['est_img']

            # SfS
            for _ in range(1):
                sfs_result = sfsnet(
                    imgs[soc:eoc], 
                    imgs[soc:eoc],#est_diffuse_imgs_chunk, 
                    est_rmaps_chunk.detach(), 
                    est_rmaps_chunk.detach(),#est_diffuse_rmaps_chunk
                    num_itr= 10,
                    initial_normal = None,
                )

            est_normals_sfs_chunk = sfs_result['est_normal']

            # SfS loss
            if True:#not confs['use_nlmvs_sfsnet']:
                sfs_loss_chunk = normal_loss(est_normals_chunk, est_normals_sfs_chunk, masks[soc:eoc])
            else:
                sfs_loss_chunk = normal_loss(
                    est_normals_chunk, 
                    est_normals_sfs_chunk, 
                    masks[soc:eoc] * est_confidences_chunk / torch.mean(est_confidences_chunk,dim=(2,3),keepdim=True)
                )

            img_loss_chunk = 0 * image_loss(est_imgs_chunk.detach(), imgs[soc:eoc])

            occ_loss_chunk = occluding_boundary_loss(est_normals_chunk, masks[soc:eoc])

            if args.depth_anything:
                est_normals_da_corrected_chunk = correct_da_normals(est_normals_da[soc:eoc], da_bl_params[soc:eoc])
                da_loss_chunk = normal_loss(est_normals_da_corrected_chunk, est_normals_chunk, masks[soc:eoc])

            loss_chunk = confs['loss_confs']['sfs_loss_weight'] * sfs_loss_chunk
            loss_chunk = loss_chunk + confs['loss_confs']['nerf_loss_weight'] * img_loss_chunk
            loss_chunk = loss_chunk + confs['loss_confs']['occluding_boundary_loss_weight'] * occ_loss_chunk
            
            if args.depth_anything:
                loss_chunk = loss_chunk + .25 * confs['loss_confs']['sfs_loss_weight'] * da_loss_chunk


            normal_error_chunk = normal_loss(
                est_normals_chunk, 
                gt_normals[soc:eoc], 
                masks[soc:eoc]
            ).item()

            loss_ = loss_chunk * num_views_per_chunk / len(proj_matrices)
            loss_.backward()

            loss = loss + loss_chunk.detach() * num_views_per_chunk / len(proj_matrices)
            sfs_loss = sfs_loss + sfs_loss_chunk.detach() * num_views_per_chunk / len(proj_matrices)
            img_loss = img_loss + img_loss_chunk.detach() * num_views_per_chunk / len(proj_matrices)
            occ_loss = occ_loss + occ_loss_chunk.detach() * num_views_per_chunk / len(proj_matrices)
            normal_error = normal_error + normal_error_chunk * num_views_per_chunk / len(proj_matrices)

            est_normals_sfs.append(est_normals_sfs_chunk.detach())
            est_rmaps.append(est_rmaps_chunk.detach())
            est_imgs.append(est_imgs_chunk.detach())

        est_normals_sfs = torch.cat(est_normals_sfs, dim=0)
        est_imgs = torch.cat(est_imgs, dim=0)
        est_rmaps = torch.cat(est_rmaps, dim=0)            

        # normal reg loss
        est_mean_normals = torch.sum(est_normals * masks, dim=(2,3)) / torch.sum(masks, dim=(2,3))
        normal_reg_loss = torch.mean(torch.sum((est_mean_normals - reg_normal)**2, dim=1))

        loss_ = confs['loss_confs']['normal_reg_loss_weight'] * normal_reg_loss
        loss_.backward()
        loss = loss + loss_.detach()


        # backward from est_normals to depth_grid
        est_normals_ = []
        for i in range(len(imgs)):
            est_depth, est_normal_ = depth_grids[i].create_depth_and_normal_maps(imgs.size(-1))
            est_normals_.append(est_normal_)
        est_normals_ = torch.cat(est_normals_, dim=0) * masks
        est_normals_.backward(est_normals.grad)


        if loss.item() < best_loss:
            best_loss = loss.item()
            best_depth_grids = [depth_grid.grid.detach() for depth_grid in depth_grids]

        optimizer.step()
        if args.depth_anything:
            optimizer_da.step()

        bar.set_postfix(
            sfs_loss=sfs_loss.item(),
            img_loss=img_loss.item(),
            occ_loss=occ_loss.item(),
            normal_reg_loss=normal_reg_loss.item(),
            normal_error=np.degrees(normal_error),
        )
        list_normal_error.append(normal_error)
        list_loss.append(loss.item())
        list_sfs_loss.append(sfs_loss.item())
        list_img_loss.append(img_loss.item())
        list_occ_loss.append(occ_loss.item())
        list_normal_reg_loss.append(normal_reg_loss.item())

        def save_images():
            os.makedirs(out_dir+'/est_normal', exist_ok=True)
            save_normal_map(out_dir+'/est_normal/'+str(idx_itr).zfill(3)+'.png', torch.cat(est_normals.unbind(0), dim=-1))
            os.makedirs(out_dir+'/est_normal_sfs', exist_ok=True)
            save_normal_map(out_dir+'/est_normal_sfs/'+str(idx_itr).zfill(3)+'.png', torch.cat(est_normals_sfs.unbind(0), dim=-1))

            if args.depth_anything:
                os.makedirs(out_dir+'/est_normal_da', exist_ok=True)
                save_normal_map(out_dir+'/est_normal_da/'+str(idx_itr).zfill(3)+'.png', torch.cat(est_normals_da_corrected.unbind(0), dim=-1))


            nf = torch.quantile(imgs[imgs > 0], 0.9)#torch.clamp(torch.max(gt_rmap[0]), None, 1)
            os.makedirs(out_dir+'/est_img', exist_ok=True)
            save_hdr_as_ldr(out_dir+'/est_img/'+str(idx_itr).zfill(3)+'.png', torch.cat(est_imgs.unbind(0), dim=-1)[None] / nf)
            os.makedirs(out_dir+'/est_rmap', exist_ok=True)
            save_hdr_as_ldr(out_dir+'/est_rmap/'+str(idx_itr).zfill(3)+'.png', torch.cat(est_rmaps[...,64:192,64:192].unbind(0), dim=-1)[None] / nf)

            #est_confidences_gray = torch.cat(est_confidences.unbind(0), dim=-1)[0]
            #est_confidences_gray = est_confidences_gray / torch.max(est_confidences_gray)
            #est_confidences_gray = (255 * np.clip(est_confidences_gray.cpu().numpy(),0,1)).astype(np.uint8)
            #est_confidences_cmap = cv2.applyColorMap(est_confidences_gray, cv2.COLORMAP_VIRIDIS)
            #os.makedirs(out_dir+'/est_confidence', exist_ok=True)
            #cv2.imwrite(out_dir+'/est_confidence/'+str(idx_itr).zfill(3)+'.png', est_confidences_cmap)

        if (idx_itr % 10) == 0:
            save_images()

        def plot_loss_lists():
            plt.plot(np.degrees(list_normal_error))
            plt.ylabel('Normal Error [deg]')
            plt.xlabel('Iteration')
            plt.grid()
            plt.savefig(out_dir+'/normal_error.png')
            plt.close()

            loss_lists = [
                ['sfs loss', list_sfs_loss],
                ['img loss', list_img_loss],
                ['occ loss', list_occ_loss],
                ['nreg loss', list_normal_reg_loss],
                ['total loss', list_loss]
            ]
            plt.figure(figsize=(6.4 * len(loss_lists),4.8))
            for i in range(len(loss_lists)):
                plt.subplot(1, len(loss_lists),1+i)
                plt.plot(loss_lists[i][1])
                plt.xlabel('Iteration')
                plt.ylabel(loss_lists[i][0])
                plt.grid()
            plt.savefig(out_dir+'/loss.png')
            plt.close()

        if (idx_itr % 50) == 0:
            plot_loss_lists()
        if (idx_itr % 100) == 0:
            os.makedirs(out_dir+'/est_mesh/'+str(idx_itr).zfill(4), exist_ok=True)
            save_mesh(out_dir+'/est_mesh/'+str(idx_itr).zfill(4))



        if (idx_itr - idx_itr_ofs) > 200:
            is_converged = False
            m1 = np.mean(list_loss[-200:-100])
            m2 = np.mean(list_loss[-100:])
            if m2 > (m1 * 0.98):
                is_converged = True
            if is_converged:
                with torch.no_grad():
                    for i in range(len(imgs)):
                        depth_grids[i].grid[:] = best_depth_grids[i]

                plot_loss_lists()
                idx_itr_ofs = idx_itr + 1
                if depth_grids[0].resolution >= imgs.size(-1):
                    break
                else:
                    with torch.no_grad():
                        for depth_grid in depth_grids:
                            depth_grid.resize_grid(2 * depth_grid.resolution)
                    print('grid resolution if updated from', depth_grid.resolution//2, 'to', depth_grid.resolution)

                optimizer = setup_optimizer()
                best_loss = 1e20
                best_depth_grids_state_dicts = [depth_grid.state_dict() for depth_grid in depth_grids]

    save_normal_map(out_dir+'/est_normal_final.png', torch.cat(est_normals.unbind(0), dim=-1))
    torch.save(est_normals.detach().cpu(), out_dir+'/est_normal_final.pt')
    save_normal_map(out_dir+'/est_normal_sfs_final.png', torch.cat(est_normals_sfs.unbind(0), dim=-1))
    torch.save(est_normals_sfs.detach().cpu(), out_dir+'/est_normal_sfs_final.pt')

    nf = torch.quantile(imgs[imgs > 0], 0.9)#torch.clamp(torch.max(gt_rmap[0]), None, 1)
    save_hdr_as_ldr(out_dir+'/est_img_final.png', torch.cat(est_imgs.unbind(0), dim=-1)[None] / nf)
    save_hdr(out_dir+'/est_img_final.exr', torch.cat(est_imgs.unbind(0), dim=-1)[None])
    save_hdr_as_ldr(out_dir+'/est_rmap_final.png', torch.cat(est_rmaps[...,64:192,64:192].unbind(0), dim=-1)[None] / nf)
    save_hdr(out_dir+'/est_rmap_final.exr', torch.cat(est_rmaps[...,64:192,64:192].unbind(0), dim=-1)[None])

    est_rmaps_sfs = []
    for idx_chunk in range((len(proj_matrices) - 1) // num_views_per_chunk + 1):
        soc = idx_chunk*num_views_per_chunk
        eoc = (idx_chunk+1)*num_views_per_chunk

        rmap_result_sfs = rm_net(
            imgs[soc:eoc],
            est_normals_sfs[soc:eoc],
            3
        )[-1]
        est_rmaps_sfs_chunk = rmap_result_sfs['rmap']
        est_rmaps_sfs.append(est_rmaps_sfs_chunk)
    est_rmaps_sfs = torch.cat(est_rmaps_sfs, dim=0)
    save_hdr_as_ldr(out_dir+'/est_rmap_sfs_final.png', torch.cat(est_rmaps_sfs[...,64:192,64:192].unbind(0), dim=-1)[None] / nf)
    save_hdr(out_dir+'/est_rmap_sfs_final.exr', torch.cat(est_rmaps_sfs[...,64:192,64:192].unbind(0), dim=-1)[None])

os.makedirs(out_dir+'/est_mesh_final', exist_ok=True)
save_mesh(out_dir+'/est_mesh_final')
import os
import sys
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
#os.environ["MPLBACKEND"]="WebAgg"

os.environ['MKL_NUM_THREADS']='1'
os.environ['NUMEXPR_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('object_id', type=int)
parser.add_argument('--exp-name', default=None)
parser.add_argument('--config', default='./confs/test_joint_opt_nlmvss2.json')
parser.add_argument('--cam-file', default=None)
parser.add_argument('--normal-file', default=None)
parser.add_argument('--rmap-file', default=None)
parser.add_argument('--surf-pcd-file', default=None)
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--depth-anything', action='store_true')
#parser.add_argument('--wo-photo-cost', action='store_true')
args = parser.parse_args()

if not (args.gpu is None):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(args.gpu)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset

from largesteps.geometry import compute_matrix
from largesteps.parameterize import to_differential, from_differential
from largesteps.optimize import AdamUniform

from core.dataset import nLMVSSynthDataset, DrexelMultiNatGeomDataset, TwoViewRealImageDataset
from core.rm_net import ReflectanceMapNet
from core.sfs_models import SimpleSfSNet
from core.sfs_utils import *
from core.rmap_utils import sample_rmap, create_normal_grid, rotate_rmap
from core.brdf_utils import LoadMERL, compute_brdf_reflectance
from core.criterion import ImageLogL1Loss, ImageGradientLoss, VGG16PerceptualLoss, compute_mask_loss
from core.training_utils import normal_loss
from core.geometry import SDFGrid
from core.mesh_renderer import MeshRenderer
from core.rmap_utils import ReflectanceMapShader, rvec2rotmat
from core.nie_utils import compute_face_normals, compute_vertex_normals, compute_edges, laplacian_simple
from core.appearance_model import NeuralRadianceField
from core.rmap_fea_ext import ReflectanceMapFeatureExtractor

import numpy as np
import matplotlib.pyplot as plt

import trimesh
import json

import sys
from tqdm import tqdm
import glob
import subprocess

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

    with open(args.config,'r') as f:
        confs = json.load(f)
    if not (args.exp_name is None):
        confs['exp_name'] = args.exp_name
    else:
        if args.depth_anything:
            confs['exp_name'] += '_da'


    if not 'refine_verts' in confs:
        confs['refine_verts'] = False
    if not ('vhull_loss_weight' in confs['loss_confs']):
        confs['loss_confs']['vhull_loss_weight'] = 0
    if not ('shading_reg_loss_weight' in confs['loss_confs']):
        confs['loss_confs']['shading_reg_loss_weight'] = 0
    if not ('color_reg_loss_weight' in confs['loss_confs']):
        confs['loss_confs']['color_reg_loss_weight'] = 0
    if not ('rmap_loss_weight' in confs['loss_confs']):
        confs['loss_confs']['rmap_loss_weight'] = 0
    if not ('view_start' in confs):
        confs['view_start'] = confs['view_skip'] // 2
    if not ('view_end' in confs):
        confs['view_end'] = 10000
    if not ('num_views_per_chunk' in confs):
        confs['num_views_per_chunk'] = 1# if (get_gpu_memory_size() < 32000) else 2
    if not ('use_nlmvs_sfsnet' in confs):
        confs['use_nlmvs_sfsnet'] = False
    if not ('initial_radius' in confs):
        confs['initial_radius'] = 0.1
    object_id = args.object_id
    confs['object_id'] = object_id
    max_sdf_resolution = confs['max_sdf_resolution']
    max_mesh_subdivision_level = 1

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
    #if args.wo_photo_cost:
    #    out_dir += '_wo_photo_cost'
    out_dir += '/'+str(object_id).zfill(3)
    os.makedirs(out_dir, exist_ok=True)

    with open(out_dir+'/confs.json', 'w') as f:
        json.dump(confs, f, indent=2)



    list_split = np.arange(len(dataset))
    test_subset_indices =  list_split[subset_ofs:subset_ofs+subset_len][int(confs['view_start']):int(confs['view_end']):int(confs['view_skip'])]
    test_dataset = Subset(dataset, test_subset_indices)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

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

    rmap_fea_ext = ReflectanceMapFeatureExtractor()
    rmap_fea_ext.load_state_dict(torch.load(project_dir+'/weights/rmap-fea-ext-da-ms/best.ckpt')['rmap_fea_ext_state_dict'])
    for p in rmap_fea_ext.parameters():
        p.requires_grad = False
    rmap_fea_ext.eval()
    rmap_fea_ext.to(device)

    sdf = SDFGrid(
        resolution=64, 
        initial_radius=0.4
    )
    sdf = sdf.to(device)

    vhull_sdf = SDFGrid(
        resolution=max_sdf_resolution, 
        initial_radius=0.8
    )
    vhull_sdf = vhull_sdf.to(device)
    for p in vhull_sdf.parameters():
        p.requires_grad = False

    mesh_subdivision_level = -1

    radiance_field = NeuralRadianceField(
        homogeneous=False,
        view_independent=False,
        ref_nerf=True,
        use_encoding=False
    )
    radiance_field.to(device)

    mesh_renderer = MeshRenderer(device)

    rmap_shader = ReflectanceMapShader('probe')
    rmap_shader.to(device)

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

    torch.save(extrinsics.cpu(), out_dir+'/gt_extrinsic_matrices.pt')
    torch.save(intrinsics.cpu(), out_dir+'/gt_intrinsic_matrices.pt')

    gt_extrinsics = extrinsics.clone()

    if not (args.cam_file is None):
        cam_data = np.load(args.cam_file)
        for idx_view in range(len(extrinsics)):
            Kk = cam_data['K'+str(idx_view+1)]
            Pk = cam_data['P'+str(idx_view+1)]
            Rk = cam_data['R'+str(idx_view+1)]
            tk = cam_data['t'+str(idx_view+1)]
            extrinsics[idx_view,:3,:3] = torch.from_numpy(Rk.astype(np.float32)) 
            extrinsics[idx_view,:3,3] = torch.from_numpy(tk.astype(np.float32)) 
            proj_matrices[idx_view,:,:] = torch.from_numpy(Pk.astype(np.float32))
            intrinsics[idx_view,:,:] = torch.from_numpy(Kk.astype(np.float32))

            print('T:')
            print(extrinsics[idx_view])
            print(torch.det(extrinsics[idx_view]))
            print('P:')
            print(proj_matrices[idx_view])
            #print(intrinsics[idx_view,:3,:3]@extrinsics[idx_view,:3,:4])
            print('K:')
            print(intrinsics[idx_view])

            world_scale = 1.
            world_offset = [0., 0., 0.,]

    if not (args.normal_file is None):
        est_normals_sfs_initial = torch.load(args.normal_file).to(device)
    else:
        est_normals_sfs_initial = None
    if not (args.rmap_file is None):
        est_rmaps_initial = torch.load(args.rmap_file).to(device)
        if est_rmaps_initial.size(-1) == 128:
            est_rmaps_initial = F.pad(est_rmaps_initial, (64,64,64,64))
    else:
        est_rmaps_initial = None

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

    def init_sdf_with_points(sdf, points):
        init_sdf(sdf, view_indices=None)
        return
        grid_coords = sdf.get_grid_coords()
        grid_mask = torch.zeros(grid_coords.size()[:3], dtype=sdf.grid.dtype, device=sdf.grid.device)
        for x in points:
            u = int(0.5 * (x[0].item() / sdf.range + 1) * sdf.resolution)
            v = int(0.5 * (x[1].item() / sdf.range + 1) * sdf.resolution)
            w = int(0.5 * (x[2].item() / sdf.range + 1) * sdf.resolution)

            if (min(u,v,w) < 0) or (max(u,v,w) >= sdf.resolution):
                continue

            for w_ in range(w-1,w+2):
                for v_ in range(v-1,v+2):
                    for u_ in range(u-1,u+2):
                        if (min(u_,v_,w_) < 0) or (max(u_,v_,w_) >= sdf.resolution):
                            continue
                        grid_mask[w_,v_,u_] = 1.

        if torch.any(grid_mask > 0.):
            sdf.grid[:] = -(grid_mask - 0.5)
        else:
            init_sdf(sdf, view_indices=None)
        sdf.validate()


    def init_sdf(sdf, view_indices=None):
        if view_indices is None:
            view_indices = range(len(proj_matrices))
        with torch.no_grad():
            grid_coords = sdf.get_grid_coords()
            grid_mask = torch.ones(grid_coords.size()[:3], dtype=sdf.grid.dtype, device=sdf.grid.device)
            for idx_view in view_indices:
                grid_proj_coords = (proj_matrices[idx_view:idx_view+1,None,:3,:3] @ grid_coords.view(-1,3)[None,:,:,None] + proj_matrices[idx_view:idx_view+1,None,:3,3:4])[...,0]
                grid_proj_coords = grid_proj_coords[...,:2] / grid_proj_coords[...,2:3] # nview * nvoxel * 2
                grid_proj_coords_n = 2 * grid_proj_coords / torch.tensor([masks.size(-1), masks.size(-2)], device=device) - 1
                grid_mask *= (torch.prod(torch.nn.functional.grid_sample(
                    masks[idx_view:idx_view+1],
                    grid_proj_coords_n[:,:,None,:],
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=False
                )[:,0,:,0], dim=0).view(grid_coords.size()[:3]) > 1e-4).float()
            if len(view_indices) < 5:
                grid_mask *= (torch.sum(grid_coords**2, dim=-1) < confs['initial_radius']**2).float()

            sdf.grid[:] = -(grid_mask - 0.5)
            sdf.validate()

    # initialization using a visual hull
    sdf_grid_vals_sphere = sdf.grid.detach().clone()
    for i in range(3):
        with torch.no_grad():
            if not (args.surf_pcd_file is None):
                surf_pts = torch.from_numpy(trimesh.load(args.surf_pcd_file).vertices).float().to(device)
                #import open3d as o3d
                #surf_pcd = o3d.io.read_point_cloud(args.surf_pcd_file)
                #surf_pts = torch.from_numpy(np.asarray(surf_pcd.points)).float().to(device)
                init_sdf_with_points(sdf, surf_pts)
            else:
                init_sdf(sdf)
            break

            if i == 2:
                break
            if len(proj_matrices) < 2:
                continue

            # adjust camera parameters
            verts, faces = sdf.extract_mesh(
                resolution=sdf.resolution, 
                refine_verts=True, 
                use_random_offset=False,
                use_random_rotation=False,
                use_remeshing=False
            )

            bbox_min = torch.min(verts, dim=0)[0]
            bbox_max = torch.max(verts, dim=0)[0]
            bbox_center = 0.5 * (bbox_min + bbox_max)
            r = torch.max(torch.sqrt(torch.sum((verts - bbox_center)**2, dim=-1)))
            scale = 0.8 / r
            M = torch.tensor([
                [1. / scale, 0., 0., bbox_center[0]],
                [0., 1. / scale, 0., bbox_center[1]],
                [0., 0., 1. / scale, bbox_center[2]],
                [0., 0., 0., 1.],
            ], device=device)
            proj_matrices = proj_matrices @ M
            extrinsics = extrinsics @ M


            M0 = torch.tensor([
                [1. / world_scale, 0., 0., world_offset[0]],
                [0., 1. / world_scale, 0., world_offset[1]],
                [0., 0., 1. / world_scale, world_offset[2]],
                [0., 0., 0., 1.],
            ], device=device)
            M_ = M0 @ M
            world_scale = 1. / M_[0,0].item()
            world_offset = M_[:3,3].detach().cpu().numpy()

    print('world_scale:', world_scale)
    print('world_offset:', world_offset)

    #init_sdf(sdf, view_indices=[len(extrinsics)//2,])

    sdf.save_mesh(
        out_dir+'/initial_mesh.ply', 
        resolution=sdf.resolution, 
        scale_factor = 1.0 / world_scale,
        offset=world_offset
    )

    init_sdf(vhull_sdf)

    tvecs = extrinsics[:,:3,3]
    rvecs = torch.stack([torch.tensor(cv2.Rodrigues(R.cpu().numpy())[0][:,0]).to(device) for R in extrinsics[:,:3,:3]], dim=0).float()

    if args.cam_file is None:#with torch.no_grad():
        idx_mid_view = len(rvecs) // 2
        rvecs[:] = rvecs[idx_mid_view:idx_mid_view+1,:]
        tvecs[:] = tvecs[idx_mid_view:idx_mid_view+1,:]

    if confs['refine_translation']:
        tvecs.requires_grad = True
    if confs['refine_rotation']:
        rvecs.requires_grad = True

    if args.depth_anything:
        from core.geometry import DepthGrid
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
    optimize_verts = False
    def setup_optimizer():
        if optimize_verts:
            return AdamUniform([verts_u,], 3e-2 / 1.41421356**(mesh_subdivision_level))#torch.optim.Adam([verts_u,], lr=1e-2, betas=(0.9, 0.99))
        return torch.optim.Adam(sdf.parameters(), lr=1e-2 * (64 / sdf.resolution), betas=(0.9, 0.99))

    optimizer = setup_optimizer()
    optimizer_nerf = torch.optim.Adam([
        {'params': radiance_field.parameters(), 'lr': 1e-3}, 
    ])
    optimizer_cam = torch.optim.Adam([
        {'params': rvecs, 'lr': 1e-3},
        {'params': tvecs, 'lr': 1e-3},
    ])
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
    #est_normal = dnp(mask=mask).detach()

    list_normal_error = []
    list_pose_error = []
    list_loss = []
    list_sfs_loss = []
    list_depth_loss = []
    list_mask_loss = []
    list_nerf_diffuse_img_loss = []
    list_nerf_img_loss = []
    list_laplacian_loss = []
    list_shading_reg_loss = []
    list_color_reg_loss = []
    list_vhull_loss = []
    list_rmap_loss = []
    idx_itr_ofs = 0
    best_loss = 1e20
    best_sdf_grid = sdf.grid.detach()
    for idx_itr in bar:
        # extract mesh
        if not optimize_verts:
            verts, faces = sdf.extract_mesh(
                resolution=sdf.resolution, 
                refine_verts=True, 
                use_random_offset=True,
                use_random_rotation=True,
                use_remeshing=True
            )
        else:
            verts_ = from_differential(M, verts_u, method='Cholesky')
            verts = verts_.detach()

        face_normals = compute_face_normals(verts.detach(), faces)
        vert_normals = compute_vertex_normals(verts.detach(), faces, face_normals)
        vert_normals = vert_normals

        if True:#not optimize_verts:
            verts.requires_grad = True
        vert_normals.requires_grad = True

        with torch.no_grad():
            if args.depth_anything:
                est_normals_da_corrected = correct_da_normals(est_normals_da, da_bl_params)

        optimizer_nerf.zero_grad()
        optimizer.zero_grad()
        optimizer_cam.zero_grad()
        if args.depth_anything:
            optimizer_da.zero_grad()

        num_views_per_chunk = confs['num_views_per_chunk']
        est_rmaps = []
        est_imgs = []
        est_imgs_nerf = []
        est_normals = []
        est_normals_sfs = []
        rmap_feas = []
        loss=0.0
        mask_loss = 0.0
        img_loss = 0.0
        diffuse_img_loss = 0.0
        sfs_loss = 0.0
        depth_loss = 0.0
        normal_error = 0.0
        rotation_error = 0.0
        curl_loss = 0.0
        boundary_loss = 0.0
        smooth_loss=0.0
        nerf_img_loss = 0.0
        nerf_diffuse_img_loss = 0.0
        for idx_chunk in range((len(proj_matrices) - 1) // num_views_per_chunk + 1):
            soc = idx_chunk*num_views_per_chunk
            eoc = (idx_chunk+1)*num_views_per_chunk

            rot_matrices_chunk = rvec2rotmat(rvecs[soc:eoc])
            extrinsics_chunk = torch.cat([rot_matrices_chunk, tvecs[soc:eoc,:,None]], dim=-1)
            extrinsics_chunk = torch.cat([
                extrinsics_chunk,
                torch.tensor([0.,0.,0.,1.], dtype=extrinsics_chunk.dtype, device=device)[None,None].repeat(extrinsics_chunk.size(0),1,1)
            ], dim=1)

            intrinsics_chunk = intrinsics[soc:eoc]
            intrinsics_chunk = torch.cat([
                torch.cat([intrinsics_chunk, torch.zeros_like(intrinsics_chunk[:,:3,0:1])], dim=-1),
                torch.tensor([0.,0.,0.,1.], dtype=extrinsics_chunk.dtype, device=device)[None,None].repeat(extrinsics_chunk.size(0),1,1)
            ], dim=1)
            proj_matrices_chunk = intrinsics_chunk @ extrinsics_chunk

            z_near_chunk = max(1e-3, -5 + torch.min(torch.sqrt(torch.sum(extrinsics_chunk.detach()[:,:3,3]**2, dim=1))).item())
            z_far_chunk = max(10, 5 + torch.max(torch.sqrt(torch.sum(extrinsics_chunk.detach()[:,:3,3]**2, dim=1))).item())

            # render images and normal maps
            est_imgs_nerf_chunk, est_depths_chunk, est_normals_chunk, est_masks_chunk = mesh_renderer.render(
                verts, vert_normals, faces, 
                proj_matrices_chunk, 
                extrinsics_chunk, 
                radiance_field,
                rspp=3,
                orthographic_camera= False,#not wo_mv_loss,
                level_set_fn=None,
                resolution=(imgs.size(-2),imgs.size(-1)),
                z_near=z_near_chunk,
                z_far=z_far_chunk
            )[:4]

            if True:
                est_diffuse_imgs_nerf_chunk = est_imgs_nerf_chunk.detach()

            if confs['debug_with_gt']:
                est_normals_sfs_chunk = gt_normals[soc:eoc]
                est_diffuse_rmaps_chunk = gt_diffuse_rmaps[soc:eoc]
                est_rmaps_chunk = gt_rmaps[soc:eoc]
                est_diffuse_imgs_chunk = gt_diffuse_imgs[soc:eoc]
                est_imgs_chunk = imgs[soc:eoc]
            else:
                # rmap recovery
                rmap_result = rm_net(
                    imgs[soc:eoc],
                    est_normals_chunk.detach(),
                    3
                )[-1]

                est_rmaps_chunk = rmap_result['rmap']
                est_rmaps_wo_mask_chunk = rmap_result['rmap_wo_mask']
                est_shadow_masks_chunk = rmap_result['est_mask']
                est_imgs_chunk = rmap_result['est_img']

                if (not (est_rmaps_initial is None))and (idx_itr < 100):
                    est_rmaps_chunk = est_rmaps_initial[soc:eoc]

                # SfS
                if (not (est_normals_sfs_initial is None))and (idx_itr < 100):
                    est_normals_sfs_chunk = est_normals_sfs_initial[soc:eoc]
                else:
                    for _ in range(1):
                        sfs_result = sfsnet(
                            imgs[soc:eoc], 
                            imgs[soc:eoc],#est_diffuse_imgs_chunk, 
                            est_rmaps_chunk.detach(), 
                            est_rmaps_chunk.detach(),#est_diffuse_rmaps_chunk
                            num_itr=10,
                            initial_normal = None,
                        )
                        est_normals_sfs_chunk = sfs_result['est_normal']
                        #est_normals_sfs_norm_chunk = sfs_result['est_normal_norm']
                        #est_confidences_chunk = est_normals_sfs_norm_chunk * (3. - est_normals_sfs_norm_chunk**2) / (1. + 1e-20 - est_normals_sfs_norm_chunk**2)


            rmap_feas_chunk = rmap_fea_ext(est_rmaps_chunk.detach()[:,:,64:192,64:192])

            # SfS loss
            sfs_loss_chunk = normal_loss(est_normals_chunk, est_normals_sfs_chunk, masks[soc:eoc])

            # depth loss
            est_depths_mask_chunk = (est_depths_chunk < z_far_chunk).float()
            mean_depth_chunk = torch.sum(est_depths_chunk * est_depths_mask_chunk, dim=(1,2,3)) / torch.sum(est_depths_mask_chunk, dim=(1,2,3))
            depth_loss_chunk = torch.mean(torch.abs(mean_depth_chunk - extrinsics[soc:eoc,2,3]))
            #print(mean_depth_chunk, extrinsics[soc:eoc,2,3])
            #depth_loss_chunk = torch.sum(torch.abs(est_depths_chunk - extrinsics[soc:eoc,2,3,None,None,None]) * est_depths_mask_chunk) / torch.sum(torch.abs(est_depths_mask_chunk))

            if args.depth_anything:
                est_normals_da_corrected_chunk = correct_da_normals(est_normals_da[soc:eoc], da_bl_params[soc:eoc])
                da_loss_chunk = normal_loss(est_normals_da_corrected_chunk, est_normals_chunk, masks[soc:eoc])

            # consistency loss
            #img_loss_chunk = image_loss(est_imgs_sfs_chunk, est_imgs_chunk) + image_loss(est_imgs_chunk, imgs[soc:eoc])
            #img_grad_loss_chunk = image_grad_loss(est_imgs_sfs_chunk, est_imgs_chunk) + image_grad_loss(est_imgs_chunk, imgs[soc:eoc])
            #img_perceptual_loss_chunk = perceptual_loss(est_imgs_sfs_chunk, est_imgs_chunk) + perceptual_loss(est_imgs_chunk, imgs[soc:eoc])
            #diffuse_img_loss_chunk = image_loss(est_diffuse_imgs_sfs_chunk, est_diffuse_imgs_chunk)
            #diffuse_img_grad_loss_chunk = image_grad_loss(est_diffuse_imgs_sfs_chunk, est_diffuse_imgs_chunk)
            #diffuse_img_perceptual_loss_chunk = perceptual_loss(est_diffuse_imgs_sfs_chunk, est_diffuse_imgs_chunk)

            # silhouette loss
            mask_loss_chunk = compute_mask_loss(est_masks_chunk, soft_masks[soc:eoc])

            # neural rendering loss
            nerf_img_loss_chunk = image_loss(est_imgs_nerf_chunk, imgs[soc:eoc])
            nerf_diffuse_img_loss_chunk = 0 * nerf_img_loss_chunk#image_loss(est_diffuse_imgs_nerf_chunk, est_diffuse_imgs_chunk)

            #curl_loss_chunk = compute_curl_loss(est_normals_chunk)
            #boundary_loss_chunk = compute_boundary_loss(est_normals_chunk)
            #smooth_loss_chunk = compute_smootheness_loss(est_normals_chunk)

            loss_chunk = confs['loss_confs']['sfs_loss_weight'] * sfs_loss_chunk
            #loss_chunk = loss_chunk + img_loss_chunk + 1e-1 * img_grad_loss_chunk + 1e-2 * img_perceptual_loss_chunk
            #loss_chunk = loss_chunk + 0.1 * (diffuse_img_loss_chunk + 1e-1 * diffuse_img_grad_loss_chunk + 1e-2 * diffuse_img_perceptual_loss_chunk)
            #loss_chunk = loss_chunk + 1e-2 * boundary_loss_chunk + 1e-2 * curl_loss_chunk# + 1e1 * smooth_loss_chunk
            loss_chunk = loss_chunk + confs['loss_confs']['mask_loss_weight'] * mask_loss_chunk
            loss_chunk = loss_chunk + confs['loss_confs']['nerf_loss_weight'] * nerf_img_loss_chunk
            loss_chunk = loss_chunk + confs['loss_confs']['nerf_diffuse_loss_weight'] * nerf_diffuse_img_loss_chunk

            if args.depth_anything:
                loss_chunk = loss_chunk + .25 * confs['loss_confs']['sfs_loss_weight'] * da_loss_chunk

            if len(imgs) == 1:
                loss_chunk = loss_chunk + 1e-0 * depth_loss_chunk
            #loss_chunk = img_loss_chunk + 1e-1 * img_grad_loss_chunk + 1e-2 * img_perceptual_loss_chunk

            normal_error_chunk = normal_loss(
                est_normals_chunk, 
                gt_normals[soc:eoc], 
                masks[soc:eoc]
            ).item()

            loss_ = loss_chunk * num_views_per_chunk / len(proj_matrices)
            loss_.backward()

            loss = loss + loss_chunk.detach() * num_views_per_chunk / len(proj_matrices)
            #img_loss = img_loss + img_loss_chunk.detach() * num_views_per_chunk / len(proj_matrices)
            #diffuse_img_loss = diffuse_img_loss + diffuse_img_loss_chunk.detach() * num_views_per_chunk / len(proj_matrices)
            sfs_loss = sfs_loss + sfs_loss_chunk.detach() * num_views_per_chunk / len(proj_matrices)
            depth_loss = depth_loss + depth_loss_chunk.detach() * num_views_per_chunk / len(proj_matrices)
            normal_error = normal_error + normal_error_chunk * num_views_per_chunk / len(proj_matrices)
            #curl_loss = curl_loss + curl_loss_chunk.detach() * num_views_per_chunk / len(proj_matrices)
            #boundary_loss = boundary_loss + boundary_loss_chunk.detach() * num_views_per_chunk / len(proj_matrices)
            #smooth_loss = smooth_loss + smooth_loss_chunk.detach() * num_views_per_chunk / len(proj_matrices)

            mask_loss = mask_loss + mask_loss_chunk.detach() * num_views_per_chunk / len(proj_matrices)

            nerf_img_loss = nerf_img_loss + nerf_img_loss_chunk.detach() * num_views_per_chunk / len(proj_matrices)
            nerf_diffuse_img_loss = nerf_diffuse_img_loss + nerf_diffuse_img_loss_chunk.detach() * num_views_per_chunk / len(proj_matrices)

            est_normals.append(est_normals_chunk.detach())
            est_normals_sfs.append(est_normals_sfs_chunk.detach())
            est_rmaps.append(est_rmaps_chunk.detach())
            est_imgs.append(est_imgs_chunk.detach())
            est_imgs_nerf.append(est_imgs_nerf_chunk.detach())
            rmap_feas.append(rmap_feas_chunk.detach())

        est_normals = torch.cat(est_normals, dim=0)
        est_normals_sfs = torch.cat(est_normals_sfs, dim=0)
        est_imgs = torch.cat(est_imgs, dim=0)
        est_imgs_nerf = torch.cat(est_imgs_nerf, dim=0)
        est_rmaps = torch.cat(est_rmaps, dim=0)
        rmap_feas = torch.cat(rmap_feas, dim=0)

        with torch.no_grad():
            relative_rot = rvec2rotmat(rvecs)[1:] @ rvec2rotmat(rvecs[:1]).transpose(-1,-2)
            gt_relative_rot = gt_extrinsics[1:,:3,:3] @ gt_extrinsics[:1,:3,:3].transpose(-1,-2)

            def compute_pose_error(R_est, R_gt):
                return torch.arccos(torch.clamp(0.5 * (torch.stack([torch.trace(mat) for mat in R_est @ R_gt.transpose(-1,-2)], dim=0) - 1.), -1., 1.))
            pose_error = torch.mean(compute_pose_error(relative_rot, gt_relative_rot)).item()


        # rmap
        est_rmaps_aligned = []
        rmap_feas_aligned = []
        for idx_view in range(len(est_rmaps)):
            ref_rot_chunk = rvec2rotmat(rvecs[:1])#torch.eye(3)[None].to(device)
            src_rot_chunk = rvec2rotmat(rvecs[idx_view:idx_view+1])
            est_rmaps_aligned_chunk = rotate_rmap(
                est_rmaps[idx_view:idx_view+1,:,64:192,64:192], 
                ref_rot_chunk, 
                src_rot_chunk, 
                projection_mode='probe'
            )
            rmap_feas_aligned_chunk = rotate_rmap(
                rmap_feas[idx_view:idx_view+1],
                ref_rot_chunk, 
                src_rot_chunk, 
                projection_mode='probe'
            )
            est_rmaps_aligned.append(est_rmaps_aligned_chunk)
            rmap_feas_aligned.append(rmap_feas_aligned_chunk)
        est_rmaps_aligned = torch.cat(est_rmaps_aligned, dim=0)
        rmap_feas_aligned = torch.cat(rmap_feas_aligned, dim=0)

        normal_grid = create_normal_grid(rmap_feas_aligned.size(-1),'probe').to(device)
        solid_angle = torch.sinc(torch.arccos(torch.clamp(normal_grid[...,2], -1, 1)) / np.pi) # solid angle
        solid_angle = (solid_angle * (torch.sum(normal_grid**2,dim=-1) > 0.9**2))[None,None]
        rmap_loss = 0.
        for idx_scale in range(5):
            rmap_feas_aligned_ = F.interpolate(rmap_feas_aligned, scale_factor=0.5**(idx_scale), mode='bilinear', antialias=True)
            solid_angle_ = F.interpolate(solid_angle, scale_factor=0.5**(idx_scale), mode='bilinear', antialias=True)
            rmap_loss_map_ = torch.mean(rmap_feas_aligned_**2, dim=0) - torch.mean(rmap_feas_aligned_, dim=0)**2
            rmap_loss = rmap_loss + torch.mean(rmap_loss_map_ * solid_angle_)**2**(-idx_scale)
        loss_ = confs['loss_confs']['rmap_loss_weight'] * rmap_loss
        if not (loss_.grad_fn is None):
            loss_.backward()
        loss = loss + loss_.detach()

        if False:#idx_itr % 50 == 0:
            plt.subplot(2,2,1)
            plot_hdr(torch.cat(est_rmaps[:,:,64:192,64:192].unbind(0), dim=-1)[None])
            plt.subplot(2,2,2)
            plt.imshow(torch.cat(rmap_feas.unbind(0), dim=-1)[0].cpu().numpy())
            plt.subplot(2,2,3)
            plot_hdr(torch.cat(est_rmaps_aligned.unbind(0), dim=-1)[None])
            plt.subplot(2,2,4)
            plt.imshow(torch.cat(rmap_feas.unbind(0), dim=-1)[0].cpu().numpy())
            plt.show()

        #print(rmap_feas.size())
        #exit()

        # laplacian loss
        edges = compute_edges(verts, faces)
        L = laplacian_simple(verts, edges.long())
        laplacian_loss = torch.trace(((L @ verts).T @ verts))
        loss_ = confs['loss_confs']['laplacian_loss_weight'] * laplacian_loss
        if not optimize_verts:
            loss_.backward()
            loss = loss + loss_.detach()

        optimizer_nerf.step()
        if args.depth_anything:
            optimizer_da.step()

        # vhull loss
        vhull_loss = torch.mean(torch.clamp(vhull_sdf(verts)[0],0,None)**2)
        loss_ = confs['loss_confs']['vhull_loss_weight'] * vhull_loss
        if not optimize_verts:
            loss_.backward()
            loss = loss + loss_.detach()

        if False:#len(proj_matrices) < 2:
            verts_center = torch.mean(verts, dim=0)
            center_loss = torch.sum(verts_center**2)
            loss_ = center_loss
            loss_.backward()
            loss = loss + loss_.detach()

        # backward from face_normals to verts
        face_normals = compute_face_normals(verts, faces)
        vert_normals_ = compute_vertex_normals(verts, faces, face_normals)
        vert_normals_.backward(vert_normals.grad)

        if not optimize_verts:
            # backward to SDF
            surf_sdf_vals, surf_sdf_grads = sdf(verts.detach())
            surf_sdf_vals.backward(-torch.sum(surf_sdf_grads.detach() * verts.grad, dim=-1, keepdim=True))
        else:
            with torch.no_grad():
                face_normals = compute_face_normals(verts, faces)
                vert_normals_ = compute_vertex_normals(verts, faces, face_normals)
                vert_normals_ = vert_normals_ / torch.sqrt(torch.sum(vert_normals_**2, dim=1, keepdim=True) + 1e-9)
                verts.grad[:] = torch.sum(verts.grad[:] * vert_normals_, dim=1, keepdim=True) * vert_normals_
            verts_.backward(verts.grad)

        if loss.item() < best_loss:
            best_loss = loss.item()
            if not optimize_verts:
                best_sdf_grid = sdf.grid.detach()
            else:
                best_verts = verts.detach()
        with torch.no_grad():
            if confs['refine_rotation']:
                rvecs.grad[len(rvecs)//2][:] = 0.
                #if idx_itr < 100:
                #    rvecs.grad[:] = 0.
            if confs['refine_translation']:
                tvecs.grad[len(rvecs)//2][:] = 0.
                #if idx_itr < 100:
                #    tvecs.grad[:] = 0.
        optimizer.step()
        optimizer_cam.step()
        if not optimize_verts:
            sdf.validate()
        #else:
        #    with torch.no_grad():
        #        surf_dists = torch.sqrt(torch.sum(verts.detach()**2, dim=-1, keepdim=True))
        #        surf_sphere_sdf_vals = surf_dists - 1.0
        #        surf_sphere_sdf_grads = verts.detach() / (surf_dists + 1e-6)
        #        verts[:] -= torch.clamp(surf_sphere_sdf_vals, 0.0, None) * surf_sphere_sdf_grads

        #        surf_vhull_sdf_vals, surf_vhull_sdf_grads = vhull_sdf(verts.detach())
        #        surf_vhull_sdf_normals = surf_vhull_sdf_grads / torch.clamp(torch.sum(surf_vhull_sdf_grads**2, dim=-1, keepdim=True), 0.4, None)
        #        verts[:] -= 0.95 * torch.clamp(surf_vhull_sdf_vals - 0.05, 0.0, None) * surf_vhull_sdf_normals
        #est_normal = est_normal_sfs

        bar.set_postfix(
            #img_loss=img_loss.item(),
            #dimg_loss=diffuse_img_loss.item(),
            sfs_loss=sfs_loss.item(),
            depth_loss=depth_loss.item(),
            normal_error=np.degrees(normal_error),
            #curl_loss=curl_loss.item(),
            #boundary_loss=boundary_loss.item(),
            #smooth_loss=smooth_loss.item(),
            mask_loss=mask_loss.item(),
            nerf_loss=nerf_img_loss.item(),
            nerf_diffuse_loss=nerf_diffuse_img_loss.item(),
            laplacian_loss=laplacian_loss.item(),
            #vhull_loss=vhull_loss.item(),
        )
        list_normal_error.append(normal_error)
        list_pose_error.append(pose_error)
        list_loss.append(loss.item())
        list_sfs_loss.append(sfs_loss.item())
        list_depth_loss.append(depth_loss.item())
        list_mask_loss.append(mask_loss.item())
        list_nerf_img_loss.append(nerf_img_loss.item())
        list_nerf_diffuse_img_loss.append(nerf_diffuse_img_loss.item())
        list_laplacian_loss.append(laplacian_loss.item())
        list_vhull_loss.append(vhull_loss.item())
        list_rmap_loss.append(rmap_loss.item())

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
            os.makedirs(out_dir+'/est_img_nerf', exist_ok=True)
            save_hdr_as_ldr(out_dir+'/est_img_nerf/'+str(idx_itr).zfill(3)+'.png', torch.cat(est_imgs_nerf.unbind(0), dim=-1)[None] / nf)
            os.makedirs(out_dir+'/est_rmap', exist_ok=True)
            save_hdr_as_ldr(out_dir+'/est_rmap/'+str(idx_itr).zfill(3)+'.png', torch.cat(est_rmaps[...,64:192,64:192].unbind(0), dim=-1)[None] / nf)

            #os.makedirs(out_dir+'/est_img', exist_ok=True)
            #save_hdr_as_ldr(out_dir+'/est_img/'+str(idx_itr).zfill(3)+'.png', est_img / nf)

        if (idx_itr % 10) == 0:
            save_images()

        def plot_loss_lists():
            plt.plot(np.degrees(list_normal_error))
            plt.ylabel('Normal Error [deg]')
            plt.xlabel('Iteration')
            plt.grid()
            plt.savefig(out_dir+'/normal_error.png')
            plt.close()

            plt.plot(np.degrees(list_pose_error))
            plt.ylabel('Pose Error [deg]')
            plt.xlabel('Iteration')
            plt.grid()
            plt.savefig(out_dir+'/pose_error.png')
            plt.close()

            loss_lists = [
                ['sfs loss', list_sfs_loss],
                ['mask loss', list_mask_loss],
                ['nerf image loss', list_nerf_img_loss],
                #['nerf diffuse image loss', list_nerf_diffuse_img_loss],
                ['laplacian loss', list_laplacian_loss],
                #['shading reg loss', list_shading_reg_loss],
                #['color reg loss', list_color_reg_loss],
                ['depth loss', list_depth_loss],
                ['rmap loss', list_rmap_loss],
                #['vhull loss', list_vhull_loss],
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

        def save_mesh(out_file, resolution):
            if not optimize_verts:
                sdf.save_mesh(
                    out_file, 
                    resolution=resolution, 
                    scale_factor = 1.0 / world_scale,
                    offset=world_offset
                )
            else:
                mesh = trimesh.Trimesh(
                    vertices=verts.detach().cpu().numpy() / world_scale + world_offset, 
                    faces=faces.cpu().numpy()
                )
                mesh.export(out_file)

        if (idx_itr % 50) == 0:
            plot_loss_lists()
            os.makedirs(out_dir+'/est_mesh', exist_ok=True)
            save_mesh(out_dir+'/est_mesh/'+str(idx_itr).zfill(3)+'.ply', sdf.resolution)



        if (idx_itr - idx_itr_ofs) > 200:
            is_converged = False
            m1 = np.mean(list_loss[-200:-100])
            m2 = np.mean(list_loss[-100:])
            if m2 > (m1 * 0.98):
                is_converged = True
            if is_converged:
                with torch.no_grad():
                    if not optimize_verts:
                        sdf.grid[:] = best_sdf_grid
                    else:
                        verts = best_verts
                plot_loss_lists()
                idx_itr_ofs = idx_itr + 1
                if sdf.resolution >= max_sdf_resolution:
                    if confs['refine_verts'] == False:
                        break
                    elif mesh_subdivision_level >= max_mesh_subdivision_level:
                        break
                    elif mesh_subdivision_level < 0:
                        verts, faces = sdf.extract_mesh(
                            resolution= sdf.resolution, 
                            refine_verts=True, 
                            use_random_offset=False,
                            use_random_rotation=False,
                            use_remeshing=True
                        )
                        mesh_subdivision_level = 0
                        M = compute_matrix(verts, faces, lambda_=19)
                        verts_u = to_differential(M, verts)
                        verts_u.requires_grad = True
                        optimize_verts = True
                        print('Direct mesh optimization started')
                    else:
                        # subdivide mesh
                        import open3d as o3d
                        mesh = o3d.geometry.TriangleMesh()
                        mesh.vertices = o3d.utility.Vector3dVector(verts.detach().cpu().numpy())
                        mesh.triangles = o3d.utility.Vector3iVector(faces.detach().cpu().numpy())
                        mesh.compute_vertex_normals()
                        mesh = mesh.subdivide_loop(number_of_iterations=1)

                        verts = np.asarray(mesh.vertices).astype(np.float32)
                        faces = np.asarray(mesh.triangles).astype(np.int32)

                        #mesh = pymesh.form_mesh(
                        #    verts.detach().cpu().numpy(), 
                        #    faces.detach().cpu().numpy()
                        #)
                        #mesh = pymesh.subdivide(mesh, order=1, method='simple')
                        #verts = mesh.vertices.astype(np.float32)
                        #faces = mesh.faces.astype(np.int32)
    
                        verts = torch.from_numpy(verts.copy()).to(device)
                        faces = torch.from_numpy(faces.copy()).to(device)

                        M = compute_matrix(verts, faces, lambda_=19)
                        verts_u = to_differential(M, verts)
                        verts_u.requires_grad = True

                        mesh_subdivision_level += 1
                        print('mesh is subdivided')
                else:
                    with torch.no_grad():
                        sdf.resize_grid(2 * sdf.resolution)
                    print('SDF resolution if updated from', sdf.resolution//2, 'to', sdf.resolution)

                optimizer = setup_optimizer()
                best_loss = 1e20
                if not optimize_verts:
                    best_sdf_grid = sdf.grid.detach()
                else:
                    best_verts = verts.detach()

    save_mesh(out_dir+'/est_mesh_final.ply', 2 * sdf.resolution)

    save_normal_map(out_dir+'/est_normal_final.png', torch.cat(est_normals.unbind(0), dim=-1))
    torch.save(est_normals.detach().cpu(), out_dir+'/est_normal_final.pt')

    save_normal_map(out_dir+'/est_normal_sfs_final.png', torch.cat(est_normals_sfs.unbind(0), dim=-1))
    torch.save(est_normals_sfs.detach().cpu(), out_dir+'/est_normal_sfs_final.pt')

    nf = torch.quantile(imgs[imgs > 0], 0.9)#torch.clamp(torch.max(gt_rmap[0]), None, 1)
    save_hdr_as_ldr(out_dir+'/est_img_final.png', torch.cat(est_imgs.unbind(0), dim=-1)[None] / nf)
    save_hdr(out_dir+'/est_img_final.exr', torch.cat(est_imgs.unbind(0), dim=-1)[None])
    save_hdr_as_ldr(out_dir+'/est_img_nerf_final.png', torch.cat(est_imgs_nerf.unbind(0), dim=-1)[None] / nf)
    save_hdr(out_dir+'/est_img_nerf_final.exr', torch.cat(est_imgs_nerf.unbind(0), dim=-1)[None])
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

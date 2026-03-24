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

from largesteps.geometry import compute_matrix
from largesteps.parameterize import to_differential, from_differential
from largesteps.optimize import AdamUniform

from core.dataset import nLMVSSynthDataset, DrexelMultiNatGeomDataset
from core.rm_net import ReflectanceMapNet
from core.sfs_models import SimpleSfSNet
from core.sfs_utils import save_hdr, save_hdr_as_ldr, save_normal_map
from core.rmap_utils import sample_rmap
from core.brdf_utils import LoadMERL, compute_brdf_reflectance
from core.criterion import ImageLogL1Loss, compute_mask_loss
from core.training_utils import normal_loss
from core.geometry import SDFGrid
from core.mesh_renderer import MeshRenderer
from core.nie_utils import compute_face_normals, compute_vertex_normals, compute_edges, laplacian_simple
from core.appearance_model import NeuralRadianceField

import numpy as np
import matplotlib.pyplot as plt

import trimesh
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('object_id', type=int)
    parser.add_argument('--config', default='./confs/test_sdf_sarm_nlmvss.json')
    args = parser.parse_args()

    with open(args.config,'r') as f:
        confs = json.load(f)

    if not 'refine_verts' in confs:
        confs['refine_verts'] = False
    if not ('vhull_loss_weight' in confs['loss_confs']):
        confs['loss_confs']['vhull_loss_weight'] = 0
    if not ('view_start' in confs):
        confs['view_start'] = confs['view_skip'] // 2
    if not ('view_end' in confs):
        confs['view_end'] = 10000
    if not ('num_views_per_chunk' in confs):
        confs['num_views_per_chunk'] = 2 if (get_gpu_memory_size() < 32000) else 4

    object_id = args.object_id
    confs['object_id'] = object_id
    max_sdf_resolution = confs['max_sdf_resolution']
    max_mesh_subdivision_level = 2

    torch.manual_seed(8)
    np.random.seed(8)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_path = confs['dataset_path']
    if confs['dataset_type'] == 'nlmvss':
        instance_dirs = sorted(glob.glob(dataset_path+'/?????'))
        subset_ofs = int(np.sum([len(glob.glob(instance_dirs[i]+'/images/*.exr')) for i in range(object_id)]))
        subset_len = len(glob.glob(instance_dirs[object_id]+'/images/*.exr'))

        material_id = object_id % 6
        shape_id = (object_id // 6) % 6
        illum_id = object_id // (6 * 6)

        # brdf_files = sorted(glob.glob(dataset_path+'/../assets/material/*.binary'))
        # brdf_file = brdf_files[material_id]
        brdf_file = None

        dataset = nLMVSSynthDataset(dataset_path, **confs['dataset_confs'])
    elif confs['dataset_type'] == 'nlmvsr':
        instance_dir = [f for f in sorted(glob.glob(dataset_path+'/*/*')) if os.path.isdir(f)][object_id]
        illum_name, shape_name = instance_dir.split('/')[-2:]
        subset_ofs = 0
        subset_len = len(glob.glob(instance_dir+'/view-??.exr'))

        brdf_file = None

        dataset = DrexelMultiNatGeomDataset(dataset_path, illum_name, shape_name, **confs['dataset_confs'])
    else:
        print('Error: Invalid dataset type')
        exit()

    out_dir = './run/'+confs['exp_name']
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
    weight_dir = './weights/simple-sfsnet-da'
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

    sdf = SDFGrid(
        resolution=64, 
        initial_radius=0.8
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

    # data loading
    bar = tqdm(testloader)
    bar.set_description('Data Loading')
    imgs = []
    masks = []
    soft_masks = []
    proj_matrices = []
    extrinsics = []
    gt_rmaps = []
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

        soft_mask = minbatch['masks'].to(device)[:,0] # [BS*1*H*W]
        gt_rmap = minbatch['hdr_rmaps'].to(device)[:,0] if ('hdr_rmaps' in minbatch) else None # [BS*N*1*H*W]

        proj_matrix = minbatch['proj_matrices'].to(device)[:,0] 
        extrinsic_matrix = minbatch['extrinsics'].to(device)[:,0] 

        normal_map = normal_map / torch.sqrt(torch.clamp(torch.sum(normal_map**2, dim=1, keepdim=True), 1e-6, None))
        gt_normal = normal_map

        mask = torch.any(img > 0, dim=1, keepdim=True).float()

        if not (brdf_file is None):
            brdf = LoadMERL(brdf_file).to(device)
            reflectance = compute_brdf_reflectance(brdf)

            img = img
            gt_rmap = gt_rmap

        if not (gt_rmap is None):
            sampled_img = sample_rmap(gt_rmap, normal_map)

        imgs.append(img)
        masks.append(mask)
        soft_masks.append(soft_mask)
        proj_matrices.append(proj_matrix)
        extrinsics.append(extrinsic_matrix)
        gt_rmaps.append(gt_rmap)
        gt_normals.append(gt_normal)

    world_scale = minbatch['world_scale'][0].item() if 'world_scale' in minbatch else 1.0
    world_offset = minbatch['world_offset'][0].numpy() if 'world_offset' in minbatch else [0.0, 0.0, 0.0]

    imgs = torch.cat(imgs, dim=0)
    masks = torch.cat(masks, dim=0)
    soft_masks = torch.cat(soft_masks, dim=0)
    proj_matrices = torch.cat(proj_matrices, dim=0)
    extrinsics = torch.cat(extrinsics, dim=0)
    gt_rmaps = torch.cat(gt_rmaps, dim=0) if not (gt_rmaps[0] is None) else None
    gt_normals = torch.cat(gt_normals, dim=0)

    # normalize scale
    nf = torch.quantile(imgs[imgs > 0], 0.9)
    imgs = imgs / nf
    gt_rmaps = gt_rmaps / nf if not (gt_rmaps is None) else None

    nf = torch.quantile(imgs[imgs > 0], 0.9)#torch.clamp(torch.max(gt_rmap[0]), None, 1)
    save_hdr_as_ldr(out_dir+'/in_img.png', torch.cat(imgs.unbind(0), dim=-1)[None] / nf)
    save_hdr(out_dir+'/in_img.exr', torch.cat(imgs.unbind(0), dim=-1)[None])
    if not (gt_rmaps is None):
        save_hdr_as_ldr(out_dir+'/gt_rmap.png', torch.cat(gt_rmaps[...,64:192,64:192].unbind(0), dim=-1)[None] / nf)
        save_hdr(out_dir+'/gt_rmap.exr', torch.cat(gt_rmaps[...,64:192,64:192].unbind(0), dim=-1)[None])

    if not (gt_normals is None):
        save_normal_map(out_dir+'/gt_normal.png', torch.cat(gt_normals.unbind(0), dim=-1))

    def init_sdf(sdf):
        with torch.no_grad():
            grid_coords = sdf.get_grid_coords()
            grid_mask = torch.ones(grid_coords.size()[:3], dtype=sdf.grid.dtype, device=sdf.grid.device)
            for idx_view in range(len(proj_matrices)):
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
            if len(proj_matrices) < 2:
                grid_mask *= (torch.sum(grid_coords**2, dim=-1) < 0.5**2).float()

            sdf.grid[:] = -(grid_mask - 0.5)
            sdf.validate()

    # initialization using a visual hull
    for i in range(3):
        with torch.no_grad():
            init_sdf(sdf)

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

    sdf.save_mesh(
        out_dir+'/initial_mesh.ply', 
        resolution=sdf.resolution, 
        scale_factor = 1.0 / world_scale,
        offset=world_offset
    )

    init_sdf(vhull_sdf)

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
    image_loss = ImageLogL1Loss()
    image_loss.to(device)

    list_normal_error = []
    list_loss = []
    list_sfs_loss = []
    list_mask_loss = []
    list_nerf_img_loss = []
    list_laplacian_loss = []
    list_vhull_loss = []
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

        optimizer_nerf.zero_grad()
        optimizer.zero_grad()


        num_views_per_chunk = confs['num_views_per_chunk']
        est_rmaps = []
        est_imgs = []
        est_imgs_nerf = []
        est_normals = []
        est_normals_sfs = []
        loss=0.0
        mask_loss = 0.0
        img_loss = 0.0
        sfs_loss = 0.0
        normal_error = 0.0
        curl_loss = 0.0
        boundary_loss = 0.0
        smooth_loss=0.0
        nerf_img_loss = 0.0
        for idx_chunk in range((len(proj_matrices) - 1) // num_views_per_chunk + 1):
            soc = idx_chunk*num_views_per_chunk
            eoc = (idx_chunk+1)*num_views_per_chunk

            # render images and normal maps
            est_imgs_nerf_chunk, _, est_normals_chunk, est_masks_chunk = mesh_renderer.render(
                verts, vert_normals, faces, 
                proj_matrices[soc:eoc], 
                extrinsics[soc:eoc], 
                radiance_field,
                rspp=3,
                orthographic_camera= False,#not wo_mv_loss,
                level_set_fn=None,
                resolution=(imgs.size(-2),imgs.size(-1))
            )[:4]

            if confs['debug_with_gt']:
                est_normals_sfs_chunk = gt_normals[soc:eoc]
                est_rmaps_chunk = gt_rmaps[soc:eoc]
                est_imgs_chunk = imgs[soc:eoc]
            else:
                # RM estimation
                rmap_result = rm_net(
                    imgs[soc:eoc],
                    est_normals_chunk.detach(),
                    3
                )[-1]

                est_rmaps_chunk = rmap_result['rmap']
                est_rmaps_wo_mask_chunk = rmap_result['rmap_wo_mask']
                est_shadow_masks_chunk = rmap_result['est_mask']
                est_imgs_chunk = rmap_result['est_img']

                # SfS
                for _ in range(1):
                    sfs_result = sfsnet(
                        imgs[soc:eoc], 
                        0*imgs[soc:eoc], # deprecated input
                        est_rmaps_chunk, 
                        0 * est_rmaps_chunk, # deprecated input
                    )
                    est_normals_sfs_chunk = sfs_result['est_normal']
                    est_imgs_sfs_chunk = sample_rmap(est_rmaps_chunk, est_normals_sfs_chunk, 'probe', 'bilinear')


            # SfS loss
            sfs_loss_chunk = normal_loss(est_normals_chunk, est_normals_sfs_chunk, masks[soc:eoc])

            # silhouette loss
            mask_loss_chunk = compute_mask_loss(est_masks_chunk, soft_masks[soc:eoc])

            # neural rendering loss
            nerf_img_loss_chunk = image_loss(est_imgs_nerf_chunk, imgs[soc:eoc])

            loss_chunk = confs['loss_confs']['sfs_loss_weight'] * sfs_loss_chunk
            loss_chunk = loss_chunk + confs['loss_confs']['mask_loss_weight'] * mask_loss_chunk
            loss_chunk = loss_chunk + confs['loss_confs']['nerf_loss_weight'] * nerf_img_loss_chunk

            normal_error_chunk = normal_loss(
                est_normals_chunk, 
                gt_normals[soc:eoc], 
                masks[soc:eoc]
            ).item()

            loss_ = loss_chunk * num_views_per_chunk / len(proj_matrices)
            loss_chunk.backward()

            loss = loss + loss_chunk.detach() * num_views_per_chunk / len(proj_matrices)
            sfs_loss = sfs_loss + sfs_loss_chunk.detach() * num_views_per_chunk / len(proj_matrices)
            normal_error = normal_error + normal_error_chunk * num_views_per_chunk / len(proj_matrices)

            mask_loss = mask_loss + mask_loss_chunk.detach() * num_views_per_chunk / len(proj_matrices)

            nerf_img_loss = nerf_img_loss + nerf_img_loss_chunk.detach() * num_views_per_chunk / len(proj_matrices)

            est_normals.append(est_normals_chunk.detach())
            est_normals_sfs.append(est_normals_sfs_chunk.detach())
            est_rmaps.append(est_rmaps_chunk.detach())
            est_imgs.append(est_imgs_chunk.detach())
            est_imgs_nerf.append(est_imgs_nerf_chunk.detach())

        est_normals = torch.cat(est_normals, dim=0)
        est_normals_sfs = torch.cat(est_normals_sfs, dim=0)
        est_imgs = torch.cat(est_imgs, dim=0)
        est_imgs_nerf = torch.cat(est_imgs_nerf, dim=0)
        est_rmaps = torch.cat(est_rmaps, dim=0)

        # laplacian loss
        edges = compute_edges(verts, faces)
        L = laplacian_simple(verts, edges.long())
        laplacian_loss = torch.trace(((L @ verts).T @ verts))
        loss_ = confs['loss_confs']['laplacian_loss_weight'] * laplacian_loss
        if not optimize_verts:
            loss_.backward()
            loss = loss + loss_.detach()

        optimizer_nerf.step()

        # vhull loss
        vhull_loss = torch.mean(torch.clamp(vhull_sdf(verts)[0],0,None)**2)
        loss_ = confs['loss_confs']['vhull_loss_weight'] * vhull_loss
        if not optimize_verts:
            loss_.backward()
            loss = loss + loss_.detach()

        if len(proj_matrices) < 2:
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
            verts_.backward(verts.grad)

        if loss.item() < best_loss:
            best_loss = loss.item()
            if not optimize_verts:
                best_sdf_grid = sdf.grid.detach()
            else:
                best_verts = verts.detach()
        optimizer.step()
        if not optimize_verts:
            sdf.validate()
        else:
            with torch.no_grad():
                surf_dists = torch.sqrt(torch.sum(verts.detach()**2, dim=-1, keepdim=True))
                surf_sphere_sdf_vals = surf_dists - 1.0
                surf_sphere_sdf_grads = verts.detach() / (surf_dists + 1e-6)
                verts[:] -= torch.clamp(surf_sphere_sdf_vals, 0.0, None) * surf_sphere_sdf_grads

                surf_vhull_sdf_vals, surf_vhull_sdf_grads = vhull_sdf(verts.detach())
                surf_vhull_sdf_normals = surf_vhull_sdf_grads / torch.clamp(torch.sum(surf_vhull_sdf_grads**2, dim=-1, keepdim=True), 0.4, None)
                verts[:] -= 0.95 * torch.clamp(surf_vhull_sdf_vals - 0.05, 0.0, None) * surf_vhull_sdf_normals
        #est_normal = est_normal_sfs

        bar.set_postfix(
            sfs_loss=sfs_loss.item(),
            normal_error=np.degrees(normal_error),
            mask_loss=mask_loss.item(),
            nerf_loss=nerf_img_loss.item(),
        )
        list_normal_error.append(normal_error)
        list_loss.append(loss.item())
        list_sfs_loss.append(sfs_loss.item())
        list_mask_loss.append(mask_loss.item())
        list_nerf_img_loss.append(nerf_img_loss.item())
        list_laplacian_loss.append(laplacian_loss.item())
        list_vhull_loss.append(vhull_loss.item())

        def save_images():
            os.makedirs(out_dir+'/est_normal', exist_ok=True)
            save_normal_map(out_dir+'/est_normal/'+str(idx_itr).zfill(3)+'.png', torch.cat(est_normals.unbind(0), dim=-1))
            os.makedirs(out_dir+'/est_normal_sfs', exist_ok=True)
            save_normal_map(out_dir+'/est_normal_sfs/'+str(idx_itr).zfill(3)+'.png', torch.cat(est_normals_sfs.unbind(0), dim=-1))

            nf = torch.quantile(imgs[imgs > 0], 0.9)#torch.clamp(torch.max(gt_rmap[0]), None, 1)
            os.makedirs(out_dir+'/est_img', exist_ok=True)
            save_hdr_as_ldr(out_dir+'/est_img/'+str(idx_itr).zfill(3)+'.png', torch.cat(est_imgs.unbind(0), dim=-1)[None] / nf)
            os.makedirs(out_dir+'/est_img_nerf', exist_ok=True)
            save_hdr_as_ldr(out_dir+'/est_img_nerf/'+str(idx_itr).zfill(3)+'.png', torch.cat(est_imgs_nerf.unbind(0), dim=-1)[None] / nf)
            os.makedirs(out_dir+'/est_rmap', exist_ok=True)
            save_hdr_as_ldr(out_dir+'/est_rmap/'+str(idx_itr).zfill(3)+'.png', torch.cat(est_rmaps[...,64:192,64:192].unbind(0), dim=-1)[None] / nf)

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
                ['mask loss', list_mask_loss],
                ['nerf image loss', list_nerf_img_loss],
                ['laplacian loss', list_laplacian_loss],
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
    save_normal_map(out_dir+'/est_normal_sfs_file.png', torch.cat(est_normals_sfs.unbind(0), dim=-1))

    nf = torch.quantile(imgs[imgs > 0], 0.9)#torch.clamp(torch.max(gt_rmap[0]), None, 1)
    save_hdr_as_ldr(out_dir+'/est_img_final.png', torch.cat(est_imgs.unbind(0), dim=-1)[None] / nf)
    save_hdr(out_dir+'/est_img_final.exr', torch.cat(est_imgs.unbind(0), dim=-1)[None])
    save_hdr_as_ldr(out_dir+'/est_img_nerf_final.png', torch.cat(est_imgs_nerf.unbind(0), dim=-1)[None] / nf)
    save_hdr(out_dir+'/est_img_nerf_final.exr', torch.cat(est_imgs_nerf.unbind(0), dim=-1)[None])
    save_hdr_as_ldr(out_dir+'/est_rmap_final.png', torch.cat(est_rmaps[...,64:192,64:192].unbind(0), dim=-1)[None] / nf)
    save_hdr(out_dir+'/est_rmap_final.exr', torch.cat(est_rmaps[...,64:192,64:192].unbind(0), dim=-1)[None])

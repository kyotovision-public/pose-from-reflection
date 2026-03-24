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

from core.dataset import nLMVSSynthDataset, DrexelMultiNatGeomDataset
from core.rm_net import ReflectanceMapNet
from core.sfs_utils import *
from core.rmap_utils import sample_rmap, create_normal_grid, img2rmap
from core.brdf_utils import LoadMERL, compute_brdf_reflectance

import numpy as np
import matplotlib.pyplot as plt

import json

import sys
from tqdm import tqdm
import glob

import argparse
import json

def compute_depth_error(est_depth, gt_depth, mask, diagonal_length):
    mask = mask * (gt_depth > 0.0).float()
    error_map = torch.abs(est_depth-gt_depth) * mask / diagonal_length
    return torch.sum(error_map) / torch.sum(mask)

def compute_log_l1_error(est_img, tar_img, normalize_scale=False, weight_img=None):
    mean_color = torch.mean(tar_img, dim=(2,3))
    est_img = torch.log1p(100 * est_img / mean_color[:,:,None,None])
    tar_img = torch.log1p(100 * tar_img / mean_color[:,:,None,None])
    m = torch.any(tar_img > 0.0, dim=-3)
    error_map = []
    for idx_ch in range(est_img.size(-3)):
        y = est_img[...,idx_ch,:,:][m].view(-1)
        t = tar_img[...,idx_ch,:,:][m].view(-1)
        if weight_img is None:
            w = 1.0
        else:
            w = weight_img[...,0,:,:][m].view(-1)
        mid_ofs = 0.0
        if normalize_scale:
            min_ofs = torch.min(y - t)
            max_ofs = torch.max(y - t)
            for _ in range(20):
                mid_ofs = 0.5 * (min_ofs + max_ofs)
                grad = torch.sum(w * torch.sign(y - t - mid_ofs))
                if grad.item() >= 0:
                    min_ofs = mid_ofs
                else:
                    max_ofs = mid_ofs
                #e = torch.sum(w * torch.abs(y - t - mid_ofs)) / torch.sum(w * torch.ones_like(y - t - mid_ofs))
            mid_ofs = 0.5 * (min_ofs + max_ofs)
        error_map.append(torch.abs(est_img[...,idx_ch,:,:] - tar_img[...,idx_ch,:,:] - mid_ofs) * m)
    error_map = torch.mean(torch.stack(error_map, dim=-3), dim=-3, keepdim=True)
    if weight_img is None:
        weight_img = torch.ones_like(error_map)
    return torch.sum(weight_img * error_map) / torch.sum(weight_img), error_map

if True:
    parser = argparse.ArgumentParser()
    parser.add_argument('object_id', type=int)
    parser.add_argument('--config', default='./confs/test_rm_net.json')
    #parser.add_argument('--wo-photo-cost', action='store_true')
    args = parser.parse_args()

    with open(args.config,'r') as f:
        confs = json.load(f)

    object_id = args.object_id

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_path = confs['dataset_path'] #os.environ['HOME']+'/data/mvs_eval/rendered'
    if confs['dataset_type'] == 'nlmvss':
        instance_dirs = sorted(glob.glob(dataset_path+'/?????'))
        subset_ofs = int(np.sum([len(glob.glob(instance_dirs[i]+'/images/*.exr')) for i in range(object_id)]))
        subset_len = len(glob.glob(instance_dirs[object_id]+'/images/*.exr'))

        material_id = object_id % 6
        shape_id = (object_id // 6) % 6
        illum_id = object_id // (6 * 6)

        #brdf_files = sorted(glob.glob(dataset_path+'/../assets/material/*.binary'))
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
    else:
        print('Error: Invalid dataset type')
        exit()

    out_dir = './run/'+confs['exp_name']
    #if args.wo_photo_cost:
    #    out_dir += '_wo_photo_cost'
    out_dir += '/'+str(object_id).zfill(3)
    os.makedirs(out_dir, exist_ok=True)



    list_split = np.arange(len(dataset))
    test_subset_indices =  list_split[subset_ofs:subset_ofs+subset_len]
    test_dataset = Subset(dataset, test_subset_indices)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    # load model
    rm_net = ReflectanceMapNet()
    rm_net.load_state_dict(torch.load('./weights/rm-net-da/best.ckpt')['rm_net_state_dict'])
    for p in rm_net.parameters():
        p.requires_grad = False
    rm_net.eval()
    rm_net.to(device)

    rm_net_wo_mask = ReflectanceMapNet(wo_mask=True)
    rm_net_wo_mask.load_state_dict(torch.load('./weights/rm-net-wo-mask-da/best.ckpt')['rm_net_state_dict'])
    for p in rm_net_wo_mask.parameters():
        p.requires_grad = False
    rm_net_wo_mask.eval()
    rm_net_wo_mask.to(device)

    rm_net_wo_img_filtering = ReflectanceMapNet(wo_img_filtering=True)
    rm_net_wo_img_filtering.load_state_dict(torch.load('./weights/rm-net-wo-img-filtering-da/best.ckpt')['rm_net_state_dict'])
    for p in rm_net_wo_img_filtering.parameters():
        p.requires_grad = False
    rm_net_wo_img_filtering.eval()
    rm_net_wo_img_filtering.to(device)

    rm_net_naive = ReflectanceMapNet(wo_img_filtering=True)
    rm_net_naive.load_state_dict(torch.load('./weights/rm-net-wo-mask-wo-img-filtering-da/best.ckpt')['rm_net_state_dict'])
    for p in rm_net_naive.parameters():
        p.requires_grad = False
    rm_net_naive.eval()
    rm_net_naive.to(device)

    per_object_results = []
    total_depth_error = 0.0
    total_num = 0.0
    bar = tqdm(testloader)
    imgs = []
    normal_maps = []
    est_rmaps = []
    gt_rmaps = []
    est_masks = []
    est_normals = []
    for idx_img, minbatch in enumerate(bar):
        view_id = idx_img# % num_images_per_object

        def load_if_exists(name):
            if name in minbatch:
                return minbatch[name].to(device)


        img = minbatch['hdr_images'].to(device)[:,0] # [BS*C*H*W]
        normal_map = minbatch['gt_normals'].to(device)[:,0] # [BS*3*H*W]

        gt_rmap = minbatch['hdr_rmaps'].to(device)[:,0] if ('hdr_rmaps' in minbatch) else None # [BS*N*1*H*W]

        normal_map = normal_map / torch.sqrt(torch.clamp(torch.sum(normal_map**2, dim=1, keepdim=True), 1e-6, None))

        if not (brdf_file is None):
            brdf = LoadMERL(brdf_file).to(device)
            reflectance = compute_brdf_reflectance(brdf)

            img = img
            gt_rmap = gt_rmap

        rmap_result = rm_net(
            img,
            normal_map,
            3
        )[-1]

        est_rmap = rmap_result['rmap']
        est_img = rmap_result['est_img']
        est_mask = rmap_result['est_mask']


        if not (gt_rmap is None):
            sampled_img = sample_rmap(gt_rmap, normal_map)

        # ablation
        est_rmap_wo_mask = rm_net_wo_mask(img, normal_map)[-1]['rmap']
        est_rmap_wo_img_filtering = rm_net_wo_img_filtering(img, normal_map)[-1]['rmap']
        est_rmap_naive = rm_net_naive(img, normal_map)[-1]['rmap']

        sparse_rmap = img2rmap(img, normal_map, torch.any(img > 0, dim=1, keepdim=True).float(), 256, projection_mode='probe')

        view_result_dir = out_dir+'/'+str(idx_img).zfill(3)
        os.makedirs(view_result_dir, exist_ok=True)

        # quantitative evaluation
        if not (gt_rmap is None):
            nz = create_normal_grid(est_rmap.size(-1), projection_mode='probe')[...,2].to(device)
            solid_angle = (2 * np.pi)**2 / (est_rmap.size(-2) * est_rmap.size(-1)) * torch.sinc(torch.acos(torch.clamp(nz, -0.99999, 0.99999)) / np.pi)
            solid_angle = solid_angle * torch.any(gt_rmap > 0, dim=-3, keepdim=True)
            est_img_log_mae, est_img_error_map = compute_log_l1_error(est_img, sampled_img)

            est_rmap_log_mae, est_rmap_error_map = compute_log_l1_error(est_rmap, gt_rmap,weight_img=solid_angle)
            est_rmap_log_mae_wo_mask, est_rmap_error_map_wo_mask = compute_log_l1_error(est_rmap_wo_mask, gt_rmap,weight_img=solid_angle)
            est_rmap_log_mae_wo_img_filtering, est_rmap_error_map_wo_img_filtering = compute_log_l1_error(est_rmap_wo_img_filtering, gt_rmap,weight_img=solid_angle)
            est_rmap_log_mae_naive, est_rmap_error_map_naive = compute_log_l1_error(est_rmap_naive, gt_rmap,weight_img=solid_angle)

            error_dict = {
                'rmap_log_mae': est_rmap_log_mae.item(),
                'rmap_log_mae_wo_mask': est_rmap_log_mae_wo_mask.item(),
                'rmap_log_mae_wo_img_filtering': est_rmap_log_mae_wo_img_filtering.item(),
                'rmap_log_mae_naive': est_rmap_log_mae_naive.item(),
            }
            with open(view_result_dir+'/error.json', 'w') as f:
                json.dump(error_dict, f, indent=2)

        def save_rmap_as_ldr(dst, rmap, sparse=False):
            if  rmap.ndim == 4:
                rmap = rmap[0]
            rmap = rmap[...,64:192,64:192].detach().cpu().numpy().transpose(1,2,0)
            rmap = (255 * np.clip(rmap[...,::-1],0,1)**(1/2.2)).astype(np.uint8)
            m = 255 * np.any(rmap > 0, axis=-1, keepdims=True).astype(np.uint8)
            if not sparse:
                nz = create_normal_grid(est_rmap.size(-1), projection_mode='probe')[...,2:3].numpy()
                m = 255 * (nz[64:192,64:192] > 0).astype(np.uint8)
            rmap = np.concatenate([rmap, m], axis=-1)
            cv2.imwrite(dst, rmap)

        nf = torch.quantile(img[img > 0], 0.95)
        save_hdr(view_result_dir+'/input_image.exr', img)
        save_hdr_as_ldr(view_result_dir+'/input_image.png', img / nf)

        save_hdr(view_result_dir+'/recon_image.exr', est_img)
        save_hdr_as_ldr(view_result_dir+'/recon_image.png', est_img / nf)

        save_normal_map(view_result_dir+'/input_normal_map.png', normal_map)

        save_rmap_as_ldr(view_result_dir+'/est_rmap.png', est_rmap / nf)
        save_hdr(view_result_dir+'/est_rmap.exr', est_rmap[...,64:192,64:192])

        save_rmap_as_ldr(view_result_dir+'/est_rmap_wo_mask.png', est_rmap_wo_mask / nf)
        save_hdr(view_result_dir+'/est_rmap_wo_mask.exr', est_rmap_wo_mask[...,64:192,64:192])

        save_rmap_as_ldr(view_result_dir+'/est_rmap_wo_img_filtering.png', est_rmap_wo_img_filtering / nf)
        save_hdr(view_result_dir+'/est_rmap_wo_img_filtering.exr', est_rmap_wo_img_filtering[...,64:192,64:192])

        save_rmap_as_ldr(view_result_dir+'/est_rmap_naive.png', est_rmap_naive / nf)
        save_hdr(view_result_dir+'/est_rmap_naive.exr', est_rmap_naive[...,64:192,64:192])

        save_rmap_as_ldr(view_result_dir+'/est_rmap_sparse.png', sparse_rmap / nf, sparse=True)
        save_hdr(view_result_dir+'/est_rmap_sparse.exr', sparse_rmap[...,64:192,64:192])

        if not (gt_rmap is None):
            save_rmap_as_ldr(view_result_dir+'/gt_rmap.png', gt_rmap / nf)
            save_hdr(view_result_dir+'/gt_rmap.exr', gt_rmap[...,64:192,64:192])

        log_est_mask = ((est_mask+1e-20).log() - (-7))
        log_est_mask = log_est_mask / torch.max(log_est_mask)
        est_mask_gray = (255 * np.clip(log_est_mask[0,0].cpu().numpy(),0,1)).astype(np.uint8)
        est_mask_cmap = cv2.applyColorMap(est_mask_gray, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(view_result_dir+'/est_mask.png', est_mask_cmap)

        imgs.append(img)
        normal_maps.append(normal_map)
        est_rmaps.append(est_rmap)
        gt_rmaps.append(gt_rmap)
        est_masks.append(est_mask)
            
    with open(out_dir+'/000/error.json','r') as f:
        error_keys = [k for k in json.load(f)]
    error_dict = {}
    for k in error_keys:
        error_dict[k] = 0.0
    result_files = sorted(glob.glob(out_dir+'/???/error.json'))
    for result_file in result_files:
        with open(result_file,'r') as f:
            res = json.load(f)
            for k in error_keys:
                error_dict[k] += res[k] / len(result_files)
    for k in error_keys:
        print(k.ljust(30),':', error_dict[k])
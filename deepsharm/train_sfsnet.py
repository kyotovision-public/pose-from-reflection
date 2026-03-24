import os
import sys
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

os.environ['MKL_NUM_THREADS']='1'
os.environ['NUMEXPR_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset

from core.dataset import PreProcessedDataset
from core.sfs_models import SimpleSfSNet
from core.sfs_utils import plot_hdr, plot_normal_map
from core.training_utils import normal_loss

import numpy as np
import matplotlib.pyplot as plt

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import glob

from tqdm import tqdm
import argparse

#torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BS = 32

parser = argparse.ArgumentParser()
args = parser.parse_args()

weight_dir = './weights/simple-sfsnet-da'
os.makedirs(weight_dir, exist_ok=True)

dataset = PreProcessedDataset(os.environ['HOME']+'/data/tmp/deepsharm')

def worker_init_fn(worker_id):
    torch.manual_seed(worker_id)

list_split = np.arange(len(dataset))
train_subset_indices =  list_split[:int(0.8*len(list_split))]
train_dataset = Subset(dataset, train_subset_indices)
trainloader = DataLoader(train_dataset, batch_size=BS, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

val_subset_indices =  list_split[int(0.8*len(list_split)):int(0.9*len(list_split))]
val_dataset = Subset(dataset, val_subset_indices)
valloader = DataLoader(val_dataset, batch_size=BS, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

# load model
sfsnet = SimpleSfSNet(wo_lambertian=True)
sfsnet = nn.DataParallel(sfsnet)
sfsnet.to(device)

# optimizer
optimizer = torch.optim.Adam(sfsnet.parameters())

# img: BS*C*H*W
# theta: BS
def rotate_img(img, theta, scale):
    BS,C,H,W = img.size()
    v,u = torch.meshgrid(torch.arange(H), torch.arange(W))
    x = 2 * (u.to(img.device) + 0.5) / W - 1.0
    y = 2 * (v.to(img.device) + 0.5) / H - 1.0

    theta = theta[:,None,None]
    x_ = torch.cos(theta) * x - torch.sin(theta) * y
    y_ = torch.sin(theta) * x + torch.cos(theta) * y

    grid = scale[:,None,None,None] * torch.stack([x_, y_], dim=-1)
    img_ = F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=False)

    return img_

def rotate_normal_map(normal_map, theta, scale):
    normal_map_ = rotate_img(normal_map, theta, scale)

    nx, ny, nz = normal_map_.unbind(1)
    theta = theta[:,None,None]
    nx_ = torch.cos(theta) * nx - torch.sin(theta) * ny
    ny_ = torch.sin(theta) * nx + torch.cos(theta) * ny
    nz_ = nz

    normal_ = torch.stack([nx_, ny_, nz_], dim=1)
    normal_ = normal_ / torch.sqrt(torch.sum(normal_**2, dim=1, keepdim=True) + 1e-4)
    return normal_

# loss
def compute_nll_loss(sfs_result, gt_normal):
    mask = (torch.sum(gt_normal**2, dim=1, keepdim=True) > 0.25).float()
    pi, kappa, mu = sfs_result[:3]
    dp = torch.sum(mu * gt_normal[:,None], dim=2, keepdim=True)
    C_ = kappa / (2 * np.pi * (1.0 - torch.clamp(-2.0 * kappa,None,-1e-7).exp()))
    pdf = torch.sum(pi * C_ * torch.clamp(kappa * (dp - 1), None, 30).exp(), dim=1)
    nl_pdf = -torch.log(pdf + 1e-6)

    loss = torch.sum(nl_pdf * mask) / (torch.sum(mask) + 1e-9)
    return loss


list_ckpt = sorted(glob.glob(weight_dir+'/???.ckpt'))
idx_itr_ofs = 0
if len(list_ckpt) > 0:
    path = list_ckpt[-1]
    checkpoint = torch.load(path)
    print('existing checkpoint '+path+' loaded')
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    sfsnet.module.load_state_dict(checkpoint['sfsnet_state_dict'])
    idx_itr_ofs = len(list_ckpt)

for idx_itr in range(idx_itr_ofs, 200):
    # train
    bar = tqdm(trainloader)
    bar.set_description('Epoch '+str(idx_itr+1).zfill(2)+' (train)')
    sfsnet.train()
    total_loss = 0.0
    total_loss_pw = 0.0
    total_loss_final = 0.0
    total_loss_mean = 0.0
    for idx_minbatch, minbatch in enumerate(bar):
        img = minbatch['imgs'][:,0].to(device)
        diffuse_img = minbatch['diffuse_imgs'][:,0].to(device)
        rmap = minbatch['rmaps'][:,0].to(device)
        diffuse_rmap = minbatch['diffuse_rmaps'][:,0].to(device)
        #mask = minbatch['mask'].to(device)
        #rmap_diffuse = minbatch['rmap_diffuse'].to(device)
        #rmap_mirror = minbatch['rmap_mirror'].to(device)
        gt_normal = minbatch['normals'][:,0].to(device)

        # data aug
        if True:
            scale = 0.25 + 1.5 * torch.rand((img.size(0),), device=device)
            theta = 2.0 * np.pi * torch.rand((img.size(0),), device=device)
            img = rotate_img(img, theta, scale)
            diffuse_img = rotate_img(diffuse_img, theta, scale)
            rmap = rotate_img(rmap, theta, torch.ones_like(scale))
            diffuse_rmap = rotate_img(diffuse_rmap, theta, torch.ones_like(scale))
            gt_normal = rotate_normal_map(gt_normal, theta, scale)
            diffuse_img = diffuse_img * (0.1 * torch.randn_like(diffuse_img)).exp()

        mask = (torch.sum(gt_normal**2, dim=1, keepdim=True) > 0.1).float()
        mask = mask * torch.any(img > 0.0, dim=1, keepdim=True).float()

        results = sfsnet(img, diffuse_img, rmap, diffuse_rmap)
        est_normal = results['est_normal']

        loss_pw = compute_nll_loss(results['pixel_wise_sfs_result'], gt_normal)
        loss_final = compute_nll_loss(results['sfs_result'], gt_normal)
        list_loss_mean = []
        for est_normal in results['normal_results']:
            list_loss_mean.append(normal_loss(est_normal, gt_normal, mask))
        list_loss_mean = torch.stack(list_loss_mean)

        loss = loss_pw + loss_final + torch.sum(list_loss_mean)

        if (idx_minbatch % 100) == 0:
            nf = torch.quantile(img[0][img[0] > 0], 0.9)
            plt.subplot(1,9,1)
            plot_hdr(img / nf)
            plt.subplot(1,9,2)
            plot_hdr(diffuse_img / nf)
            plt.subplot(1,9,3)
            plot_hdr(rmap / nf)
            plt.subplot(1,9,4)
            plot_hdr(diffuse_rmap / nf)
            plt.subplot(1,9,5)
            plot_normal_map(results['pixel_wise_sfs_result'][-1])
            for j in range(3):
                plt.subplot(1,9,6+j)
                plot_normal_map(results['normal_results'][j])
            plt.subplot(1,9,9)
            plot_normal_map(gt_normal)
            plt.show()

        # backward
        optimizer.zero_grad()        
        loss.backward()

        with torch.no_grad():
            nan_flag = False
            inf_flag = False
            for p in sfsnet.parameters():
                if p.grad is None:
                    continue
                if torch.any(torch.isnan(p.grad)):
                    nan_flag = True
                    p.grad[torch.isnan(p.grad)] = 0.0
                if torch.any(torch.isinf(p.grad)):
                    inf_flag = True
                    p.grad[torch.isinf(p.grad)] = 0.0
            if nan_flag == True:
                print('Warning: NaN in backward')
            if inf_flag == True:
                print('Warning: Inf in backward')

        optimizer.step()        

        # update bar postfix
        total_loss += loss.item()
        total_loss_pw += loss_pw.item()
        total_loss_final += loss_final.item()
        total_loss_mean += list_loss_mean.detach().cpu().numpy()
        bar.set_postfix(
            loss=total_loss/(idx_minbatch+1),
            #loss_coarse=total_loss_coarse/(idx_minbatch+1),
            loss_pw=total_loss_pw/(idx_minbatch+1),
            loss_final=total_loss_final/(idx_minbatch+1),
            loss_mean=total_loss_mean/(idx_minbatch+1),
        )
    train_loss = total_loss/(idx_minbatch+1)
    train_loss_pw = total_loss_pw/(idx_minbatch+1)
    train_loss_final = total_loss_final/(idx_minbatch+1)
    train_loss_mean = total_loss_mean/(idx_minbatch+1)

    # val
    bar = tqdm(valloader)
    bar.set_description('Epoch '+str(idx_itr+1).zfill(2)+' (val) ')
    sfsnet.eval()
    total_loss = 0.0
    total_loss_pw = 0.0
    total_loss_final = 0.0
    total_loss_mean = 0.0
    with torch.no_grad():
        for idx_minbatch, minbatch in enumerate(bar):
            img = minbatch['imgs'][:,0].to(device)
            diffuse_img = minbatch['diffuse_imgs'][:,0].to(device)
            rmap = minbatch['rmaps'][:,0].to(device)
            diffuse_rmap = minbatch['diffuse_rmaps'][:,0].to(device)
            #mask = minbatch['mask'].to(device)
            #rmap_diffuse = minbatch['rmap_diffuse'].to(device)
            #rmap_mirror = minbatch['rmap_mirror'].to(device)
            gt_normal = minbatch['normals'][:,0].to(device)

            mask = (torch.sum(gt_normal**2, dim=1, keepdim=True) > 0.1).float()
            mask = mask * torch.any(img > 0.0, dim=1, keepdim=True).float()

            results = sfsnet(img, diffuse_img, rmap, diffuse_rmap)

            est_normal = results['est_normal']

            loss_pw = compute_nll_loss(results['pixel_wise_sfs_result'], gt_normal)
            loss_final = compute_nll_loss(results['sfs_result'], gt_normal)
            list_loss_mean = []
            for est_normal in results['normal_results']:
                list_loss_mean.append(normal_loss(est_normal, gt_normal, mask))
            list_loss_mean = torch.stack(list_loss_mean)

            loss = loss_pw + loss_final + torch.sum(list_loss_mean)

            # update bar postfix
            total_loss += loss.item()
            total_loss_pw += loss_pw.item()
            total_loss_final += loss_final.item()
            total_loss_mean += list_loss_mean.detach().cpu().numpy()

            bar.set_postfix(
                loss=total_loss/(idx_minbatch+1),
                #loss_coarse=total_loss_coarse/(idx_minbatch+1),
                loss_pw=total_loss_pw/(idx_minbatch+1),
                loss_final=total_loss_final/(idx_minbatch+1),
                loss_mean=total_loss_mean/(idx_minbatch+1),
            )
    val_loss = total_loss/(idx_minbatch+1)
    val_loss_pw = total_loss_pw/(idx_minbatch+1)
    val_loss_final = total_loss_final/(idx_minbatch+1)
    val_loss_mean = total_loss_mean/(idx_minbatch+1)

    # save weights
    torch.save({
            'sfsnet_state_dict': sfsnet.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_loss_pw': train_loss_pw,
            'val_loss_pw': val_loss_pw,
            'train_loss_final': train_loss_final,
            'val_loss_final': val_loss_final,
            'train_loss_mean': train_loss_mean,
            'val_loss_mean': val_loss_mean,
    }, weight_dir+'/'+str(idx_itr).zfill(3)+'.ckpt') 
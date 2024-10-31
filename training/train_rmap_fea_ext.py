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
from core.rm_net import ReflectanceMapNet
from core.rmap_fea_ext import ReflectanceMapFeatureExtractor
from core.sfs_utils import *
from core.rmap_utils import rotate_rmap, create_normal_grid, sample_rmap

import numpy as np

from info_nce_pytorch.info_nce import InfoNCE

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import glob
import argparse

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

#torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BS = 32

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-dir', type=str, default=os.environ['HOME']+'/data/tmp/rmap-fea-ext')
args = parser.parse_args()

weight_dir = './weights/rmap-fea-ext-da-ms'
os.makedirs(weight_dir, exist_ok=True)

logger = SummaryWriter(log_dir=weight_dir, flush_secs=20)

dataset = PreProcessedDataset(args.dataset_dir)

def worker_init_fn(worker_id):
    torch.manual_seed(worker_id)

list_split = np.arange(len(dataset))
train_subset_indices =  list_split[:int(0.8*len(list_split))]
train_dataset = Subset(dataset, train_subset_indices)
trainloader = DataLoader(train_dataset, batch_size=BS, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

val_subset_indices =  list_split[int(0.8*len(list_split)):int(0.9*len(list_split))]
val_dataset = Subset(dataset, val_subset_indices)
valloader = DataLoader(val_dataset, batch_size=BS, shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

# load model
rmap_fea_ext = ReflectanceMapFeatureExtractor()
rmap_fea_ext = nn.DataParallel(rmap_fea_ext)
rmap_fea_ext.to(device)


rm_net = ReflectanceMapNet()
rm_net.load_state_dict(torch.load('./weights/rm-net-da/best.ckpt')['rm_net_state_dict'])
rm_net.to(device)
for p in rm_net.parameters():
    p.requires_grad = False
rm_net.eval()

info_nce_loss = InfoNCE()

# optimizer
optimizer = torch.optim.Adam(rmap_fea_ext.parameters())

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
    img_ = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    return img_

list_ckpt = sorted(glob.glob(weight_dir+'/latest.ckpt'))
idx_itr_ofs = 0
min_val_loss = 1e12
if len(list_ckpt) > 0:
    path = list_ckpt[-1]
    checkpoint = torch.load(path)
    print('existing checkpoint '+path+' loaded')
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    rmap_fea_ext.module.load_state_dict(checkpoint['rmap_fea_ext_state_dict'])
    idx_itr_ofs = checkpoint['idx_itr'] + 1
    min_val_loss = checkpoint['min_val_loss']

for idx_itr in range(idx_itr_ofs, 200):
    loss_dict = {}
    def add_to_dict(key, value, global_step):
        if key in loss_dict:
            loss_dict[key].append(value)
        else:
            loss_dict[key] = [value,]

        logger.add_scalar(key, value, global_step)

    # train
    bar = tqdm(trainloader)
    bar.set_description('Epoch '+str(idx_itr+1).zfill(2)+' (train)')
    rmap_fea_ext.train()
    for idx_minbatch, minbatch in enumerate(bar):
        global_steps = idx_itr * len(trainloader) + idx_minbatch
        img1 = minbatch['img1'].to(device)
        img2 = minbatch['img2'].to(device)

        normal1 = minbatch['normal1'].to(device)
        normal2 = minbatch['normal2'].to(device)

        gt_rmap1 = minbatch['rmap1'].to(device)
        gt_rmap2 = minbatch['rmap2'].to(device)

        rot1 = minbatch['rot1'].to(device)
        rot2 = minbatch['rot2'].to(device)

        # data aug inspired by the Bas-Relief anbiguity
        def create_gbr_matrix():
            bs = img1.size(0)
            device = img1.device
            mu = 0.5 * torch.randn((bs,), device=device)
            nu = 0.5 * torch.randn((bs,), device=device)
            s = torch.exp(0.5 * torch.randn((bs,), device=device))
            invG = torch.eye(3, device=device)[None].repeat(bs,1,1)
            invG[:,2,2] = 1./s
            invG[:,2,0] = -mu/s
            invG[:,2,1] = -nu/s
            return invG.transpose(-1,-2)

        M1 = create_gbr_matrix()
        M2 = create_gbr_matrix()

        def transform_normal_map(normal, M):
            n_ = torch.sum(M[:,:,:,None,None] * normal[:,None,:,:,:], dim=2)
            return n_ / torch.sqrt(torch.sum(n_**2, dim=1, keepdim=True) + 1e-6) 
        normal1_ = transform_normal_map(normal1, M1)
        normal2_ = transform_normal_map(normal2, M2)


        rmaps_distorted = rm_net(
            torch.cat([img1, img2], dim=0), 
            torch.cat([normal1_, normal2_], dim=0), 
        )[-1]['rmap']
        rmap1_distorted = rmaps_distorted[:img1.size(0),:,64:192,64:192]
        rmap2_distorted = rmaps_distorted[img1.size(0):,:,64:192,64:192]

        # data aug
        scale = 0.25 + 1.5 * torch.rand((rmap1_distorted.size(0),), device=device)
        theta1 = 2.0 * np.pi * torch.rand((rmap1_distorted.size(0),), device=device)
        theta2 = 2.0 * np.pi * torch.rand((rmap1_distorted.size(0),), device=device)
        rmap1_distorted_rotated = rotate_img(rmap1_distorted, theta1, torch.ones_like(scale))
        rmap2_distorted_rotated = rotate_img(rmap2_distorted, theta2, torch.ones_like(scale))

        fea1_distorted_rotated = rmap_fea_ext(rmap1_distorted_rotated)
        fea2_distorted_rotated = rmap_fea_ext(rmap2_distorted_rotated)

        fea1_distorted = rotate_img(fea1_distorted_rotated, -theta1, torch.ones_like(scale))
        fea2_distorted = rotate_img(fea2_distorted_rotated, -theta2, torch.ones_like(scale))        

        def undistort_rmap(rmap, M):
            normal_grid = create_normal_grid(256,'probe')[64:192,64:192].to(device)
            n_ = (M[:,None,None,:,:] @ normal_grid[None,:,:,:,None])[...,0]
            n_ = n_ / torch.sqrt(torch.sum(n_**2, dim=-1, keepdim=True) + 1e-6)
            n_ = torch.stack(n_.unbind(-1), dim=1)

            rmap_pad = F.pad(rmap, (64,64,64,64), "constant", 0)
            rmap_ = sample_rmap(rmap_pad, n_, projection_mode='probe', interep_mode='bilinear')
            return rmap_ * (normal_grid[...,2] > 0).float()

        rmap1 = undistort_rmap(rmap1_distorted, M1)
        rmap2 = undistort_rmap(rmap2_distorted, M2)
        fea1 = undistort_rmap(fea1_distorted, M1)
        fea2 = undistort_rmap(fea2_distorted, M2)

        rmap2_aligned = rotate_rmap(rmap2, rot1, rot2)
        fea2_aligned = rotate_rmap(fea2, rot1, rot2)

        mask = torch.all(rmap1[0] > 0, dim=0).view(-1)
        loss = 0.
        for idx_scale in range(5):
            fea1_ = F.interpolate(fea1, scale_factor=0.5**(idx_scale), mode='bilinear', antialias=True)
            fea2_aligned_ = F.interpolate(fea2_aligned, scale_factor=0.5**(idx_scale), mode='bilinear', antialias=True)
            mask_ = (F.interpolate(torch.all(rmap1[0:1] > 0, dim=1, keepdim=True).float(), scale_factor=0.5**(idx_scale), mode='bilinear', antialias=True) > 0.99).view(-1)
            for idx_scene in range(rmap1.size(0)):
                queries = fea1_[idx_scene].view(fea1_.size(1),-1).transpose(0,1)[mask_]
                positive_keys = fea2_aligned_[idx_scene].view(fea2_aligned_.size(1),-1).transpose(0,1)[mask_]

                loss = loss + info_nce_loss(queries, positive_keys) / rmap1.size(0)


        if idx_minbatch % 100 == 0:
            nf = torch.quantile(rmap1[0][rmap1[0] > 0], 0.9)#torch.clamp(torch.max(gt_rmap[0]), None, 1)
            plt.subplot(6,4,1)
            plot_hdr(img1 / nf)
            plt.ylabel('In Image')
            plt.subplot(6,4,2)
            plot_hdr(img2 / nf)
            plt.subplot(6,4,5)
            plot_normal_map(normal1)
            plt.ylabel('In Normal')
            plt.subplot(6,4,6)
            plot_normal_map(normal2)
            plt.subplot(6,4,7)
            plt.ylabel('Aug Normal')
            plot_normal_map(normal1_)
            plt.subplot(6,4,8)
            plot_normal_map(normal2_)
            plt.subplot(6,4,9)
            plot_hdr(gt_rmap1 / nf)
            plt.ylabel('GT RM')
            plt.subplot(6,4,10)
            plot_hdr(gt_rmap2 / nf)
            plt.subplot(6,4,11)
            plot_hdr(rmap1_distorted / nf)
            plt.ylabel('Aug RM')
            plt.subplot(6,4,12)
            plot_hdr(rmap2_distorted / nf)
            plt.subplot(6,4,14)
            plot_hdr(rmap2_aligned / nf)
            plt.ylabel('Aligned InvAug')
            plt.subplot(6,4,15)
            plot_hdr(rmap1 / nf)
            plt.ylabel('InvAug RM')
            plt.subplot(6,4,16)
            plot_hdr(rmap2 / nf)
            plt.subplot(6,4,17)
            plt.imshow(fea1.detach().cpu().numpy()[0,0])
            plt.ylabel('InvAug Fea')
            plt.subplot(6,4,18)
            plt.imshow(fea2.detach().cpu().numpy()[0,0])
            plt.subplot(6,4,22)
            plt.imshow(fea2_aligned.detach().cpu().numpy()[0,0])
            plt.ylabel('InvAug Aligned')
            plt.show()


        # backward
        optimizer.zero_grad()        
        loss.backward()
        with torch.no_grad():
            for p in rmap_fea_ext.parameters():
                if p.grad is None:
                    continue
                p.grad[torch.isnan(p.grad)] = 0.0
                p.grad[torch.isinf(p.grad)] = 0.0
        optimizer.step()        

        add_to_dict('Train/Loss', loss.item(), global_steps)

        # update bar postfix
        bar.set_postfix(
            loss=np.mean(loss_dict['Train/Loss']),
        )

    # val
    bar = tqdm(valloader)
    bar.set_description('Epoch '+str(idx_itr+1).zfill(2)+' (val) ')
    rmap_fea_ext.eval()
    with torch.no_grad():
        for idx_minbatch, minbatch in enumerate(bar):
            global_steps = idx_itr * len(valloader) + idx_minbatch

            img1 = minbatch['img1'].to(device)
            img2 = minbatch['img2'].to(device)

            normal1 = minbatch['normal1'].to(device)
            normal2 = minbatch['normal2'].to(device)

            gt_rmap1 = minbatch['rmap1'].to(device)
            gt_rmap2 = minbatch['rmap2'].to(device)

            rot1 = minbatch['rot1'].to(device)
            rot2 = minbatch['rot2'].to(device)

            rmaps = rm_net(
                torch.cat([img1, img2], dim=0), 
                torch.cat([normal1, normal2], dim=0), 
            )[-1]['rmap']
            rmap1 = rmaps[:img1.size(0),:,64:192,64:192]
            rmap2 = rmaps[img1.size(0):,:,64:192,64:192]

            fea1 = rmap_fea_ext(rmap1)
            fea2 = rmap_fea_ext(rmap2)

            rmap2_aligned = rotate_rmap(rmap2, rot1, rot2)
            fea2_aligned = rotate_rmap(fea2, rot1, rot2)

            mask = torch.all(rmap1[0] > 0, dim=0).view(-1)
            loss = 0.
            for idx_scale in range(5):
                fea1_ = F.interpolate(fea1, scale_factor=0.5**(idx_scale), mode='bilinear', antialias=True)
                fea2_aligned_ = F.interpolate(fea2_aligned, scale_factor=0.5**(idx_scale), mode='bilinear', antialias=True)
                mask_ = (F.interpolate(torch.all(rmap1[0:1] > 0, dim=1, keepdim=True).float(), scale_factor=0.5**(idx_scale), mode='bilinear', antialias=True) > 0.99).view(-1)
                for idx_scene in range(rmap1.size(0)):
                    queries = fea1_[idx_scene].view(fea1_.size(1),-1).transpose(0,1)[mask_]
                    positive_keys = fea2_aligned_[idx_scene].view(fea2_aligned_.size(1),-1).transpose(0,1)[mask_]

                    loss = loss + info_nce_loss(queries, positive_keys) / rmap1.size(0)

            add_to_dict('Val/Loss', loss.item(), global_steps)

            # update bar postfix
            bar.set_postfix(
                loss=np.mean(loss_dict['Val/Loss']),
            )

    if np.mean(loss_dict['Val/Loss']) < min_val_loss:
        min_val_loss = np.mean(loss_dict['Val/Loss'])

        # save weights
        torch.save({
            'idx_itr': idx_itr,
            'train_loss': np.mean(loss_dict['Train/Loss']),
            'val_loss': np.mean(loss_dict['Val/Loss']),
            'min_val_loss': min_val_loss,
            'rmap_fea_ext_state_dict': rmap_fea_ext.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, weight_dir+'/best.ckpt')
        print('best.ckpt is updated')

    # save weights
    torch.save({
        'idx_itr': idx_itr,
        'train_loss': np.mean(loss_dict['Train/Loss']),
        'val_loss': np.mean(loss_dict['Val/Loss']),
        'min_val_loss': min_val_loss,
        'rmap_fea_ext_state_dict': rmap_fea_ext.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, weight_dir+'/latest.ckpt') 

    torch.save(loss_dict, weight_dir+'/'+str(idx_itr).zfill(3)+'_loss.ckpt') 
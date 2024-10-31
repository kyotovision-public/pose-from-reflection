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
from core.normal_fea_ext import NormalMapFeatureExtractor
from core.sfs_utils import *

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
parser.add_argument('--wo-data-aug', action='store_true')
args = parser.parse_args()

weight_dir = './weights/normal-fea-ext-da'
wo_data_aug = args.wo_data_aug
if wo_data_aug:
    weight_dir='./weights/normal-fea-ext'
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
normal_fea_ext = NormalMapFeatureExtractor()
normal_fea_ext = nn.DataParallel(normal_fea_ext)
normal_fea_ext.to(device)


sfsnet = SimpleSfSNet(wo_lambertian=True)
loss_files = glob.glob('./weights/simple-sfsnet-da/???.ckpt')
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

info_nce_loss = InfoNCE()

# optimizer
optimizer = torch.optim.Adam(normal_fea_ext.parameters())

# img: BS*C*H*W
# theta: BS
def rotate_img(img, theta, scale, mode='bilinear'):
    BS,C,H,W = img.size()
    v,u = torch.meshgrid(torch.arange(H), torch.arange(W))
    x = 2 * (u.to(img.device) + 0.5) / W - 1.0
    y = 2 * (v.to(img.device) + 0.5) / H - 1.0

    theta = theta[:,None,None]
    x_ = torch.cos(theta) * x - torch.sin(theta) * y
    y_ = torch.sin(theta) * x + torch.cos(theta) * y

    grid = scale[:,None,None,None] * torch.stack([x_, y_], dim=-1)
    img_ = F.grid_sample(img, grid, mode=mode, padding_mode='zeros', align_corners=False)

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

list_ckpt = sorted(glob.glob(weight_dir+'/latest.ckpt'))
idx_itr_ofs = 0
min_val_loss = 1e12
if len(list_ckpt) > 0:
    path = list_ckpt[-1]
    checkpoint = torch.load(path)
    print('existing checkpoint '+path+' loaded')
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    normal_fea_ext.module.load_state_dict(checkpoint['normal_fea_ext_state_dict'])
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
    normal_fea_ext.train()
    for idx_minbatch, minbatch in enumerate(bar):
        global_steps = idx_itr * len(trainloader) + idx_minbatch
        img1 = minbatch['img1'].to(device)
        img2 = minbatch['img2'].to(device)

        gt_normal1 = minbatch['normal1'].to(device)
        gt_normal2 = minbatch['normal2'].to(device)

        gt_rmap1 = minbatch['rmap1'].to(device)
        gt_rmap2 = minbatch['rmap2'].to(device)

        gt_corr_map = minbatch['corr_map'].to(device).float()

        rot1 = minbatch['rot1'].to(device)
        rot2 = minbatch['rot2'].to(device)

        gt_rmap1_pad = F.pad(gt_rmap1, (64,64,64,64))
        gt_rmap2_pad = F.pad(gt_rmap2, (64,64,64,64))


        normal1 = sfsnet(img1, img1, gt_rmap1_pad, gt_rmap1_pad)['est_normal']
        normal2 = sfsnet(img2, img2, gt_rmap2_pad, gt_rmap2_pad)['est_normal']
        

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

        if not wo_data_aug:
            M1 = create_gbr_matrix()
            M2 = create_gbr_matrix()

            def transform_normal_map(normal, M):
                n_ = torch.sum(M[:,:,:,None,None] * normal[:,None,:,:,:], dim=2)
                return n_ / torch.sqrt(torch.sum(n_**2, dim=1, keepdim=True) + 1e-6) 
            normal1_ = transform_normal_map(normal1, M1)
            normal2_ = transform_normal_map(normal2, M2)
        else:
            normal1_ = normal1
            normal2_ = normal2

        # data aug
        if not wo_data_aug:
            scale1 = 0.25 + 1.5 * torch.rand((normal1_.size(0),), device=device)
            theta1 = 2.0 * np.pi * torch.rand((normal1_.size(0),), device=device)

            normal1_ = rotate_normal_map(normal1_, theta1, scale1)
            gt_corr_map = rotate_img(gt_corr_map.float(), theta1, scale1, mode='nearest')

            scale2 = 0.25 + 1.5 * torch.rand((normal2_.size(0),), device=device)
            theta2 = 2.0 * np.pi * torch.rand((normal2_.size(0),), device=device)
            normal2_ = rotate_normal_map(normal2_, theta2, scale2)

            vis_mask_ = (gt_corr_map[:,0:1] > 0) * (gt_corr_map[:,1:2] > 0)
            corr_u, corr_v = gt_corr_map.float().unbind(1)
            h,w = gt_corr_map.size()[-2:]
            corr_u_ =  torch.cos(theta2)[:,None,None] * (corr_u - w // 2) + torch.sin(theta2)[:,None,None] * (corr_v - h // 2)
            corr_v_ = -torch.sin(theta2)[:,None,None] * (corr_u - w // 2) + torch.cos(theta2)[:,None,None] * (corr_v - h // 2)
            corr_u_ = corr_u_ / scale2[:,None,None] + w // 2
            corr_v_ = corr_v_ / scale2[:,None,None] + h // 2
            gt_corr_map = torch.stack([corr_u_, corr_v_], dim=1)
            vis_mask_ = vis_mask_ * (gt_corr_map[:,0:1] > 0) * (gt_corr_map[:,1:2] > 0)
            vis_mask_ = vis_mask_ * (gt_corr_map[:,0:1] < w) * (gt_corr_map[:,1:2] < h)
            gt_corr_map *= vis_mask_.float()

            assert not torch.any(torch.isnan(normal1_))
            assert not torch.any(torch.isinf(normal1_))
            assert not torch.any(torch.isnan(normal2_))
            assert not torch.any(torch.isinf(normal2_))
            assert not torch.any(torch.isnan(gt_corr_map))
            assert not torch.any(torch.isinf(gt_corr_map))



        fea1 = normal_fea_ext(normal1_)
        fea2 = normal_fea_ext(normal2_)
        assert not torch.any(torch.isnan(fea1))
        assert not torch.any(torch.isinf(fea1))
        assert not torch.any(torch.isnan(fea2))
        assert not torch.any(torch.isinf(fea2))

        if False:
            plot_normal_map(normal1_)
            plt.show()

            

            plt.subplot(2,2,1)
            plot_normal_map(normal1_)
            plt.subplot(2,2,2)
            plot_normal_map(normal2_)
            plt.subplot(2,2,3)
            plt.imshow(gt_corr_map[0,0].detach().cpu())
            plt.subplot(2,2,4)
            plt.imshow(gt_corr_map[0,1].detach().cpu())
            plt.show()

            print(gt_corr_map.size())

        vis_mask = (gt_corr_map[:,0].long() > 0) * (gt_corr_map[:,1].long() > 0) # BS*H*W
        loss_sum = 10 * 1e-20
        loss_num = 1e-20
        for idx_scene in range(fea1.size(0)):
            mask = vis_mask[idx_scene].view(-1)
            queries = fea1[idx_scene].view(fea1.size(1),-1).transpose(0,1)[mask] # N*C

            indices = (gt_corr_map[idx_scene,1].long() * gt_corr_map.size(-1) + gt_corr_map[idx_scene,0].long()).view(-1)
            positive_keys = fea2[idx_scene].view(fea2.size(1),-1).transpose(0,1)[indices][mask]

            assert not torch.any(torch.isnan(queries))
            assert not torch.any(torch.isinf(queries))
            assert not torch.any(torch.isnan(positive_keys))
            assert not torch.any(torch.isinf(positive_keys))


            if len(queries) < 100:
                continue

            loss_sum = loss_sum + info_nce_loss(queries, positive_keys)
            loss_num += 1.

            assert not torch.any(torch.isnan(loss_sum))
            assert not torch.any(torch.isinf(loss_sum))
        loss = loss_sum / loss_num


        if idx_minbatch % 100 == 0:
            nf = torch.quantile(gt_rmap1[0][gt_rmap1[0] > 0], 0.9)#torch.clamp(torch.max(gt_rmap[0]), None, 1)
            plt.subplot(3,2,1)
            plot_normal_map(normal1)
            plt.ylabel('In Normal')
            plt.subplot(3,2,2)
            plot_normal_map(normal2)
            plt.subplot(3,2,3)
            plt.ylabel('Aug Normal')
            plot_normal_map(normal1_)
            plt.subplot(3,2,4)
            plot_normal_map(normal2_)
            plt.subplot(3,2,5)
            plt.imshow(fea1.detach().cpu().numpy()[0,0])
            plt.ylabel('Fea')
            plt.subplot(3,2,6)
            plt.imshow(fea2.detach().cpu().numpy()[0,0])
            plt.show()


        # backward
        optimizer.zero_grad()        
        loss.backward()
        with torch.no_grad():
            for p in normal_fea_ext.parameters():
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
    normal_fea_ext.eval()
    with torch.no_grad():
        for idx_minbatch, minbatch in enumerate(bar):
            global_steps = idx_itr * len(valloader) + idx_minbatch
            img1 = minbatch['img1'].to(device)
            img2 = minbatch['img2'].to(device)

            gt_normal1 = minbatch['normal1'].to(device)
            gt_normal2 = minbatch['normal2'].to(device)

            gt_rmap1 = minbatch['rmap1'].to(device)
            gt_rmap2 = minbatch['rmap2'].to(device)

            gt_corr_map = minbatch['corr_map'].to(device).float()

            rot1 = minbatch['rot1'].to(device)
            rot2 = minbatch['rot2'].to(device)

            gt_rmap1_pad = F.pad(gt_rmap1, (64,64,64,64))
            gt_rmap2_pad = F.pad(gt_rmap2, (64,64,64,64))


            normal1 = sfsnet(img1, img1, gt_rmap1_pad, gt_rmap1_pad)['est_normal']
            normal2 = sfsnet(img2, img2, gt_rmap2_pad, gt_rmap2_pad)['est_normal']
            
            normal1_ = normal1
            normal2_ = normal2

            fea1 = normal_fea_ext(normal1_)
            fea2 = normal_fea_ext(normal2_)


            vis_mask = (gt_corr_map[:,0].long() > 0) * (gt_corr_map[:,1].long() > 0) # BS*H*W
            loss_sum = 10 * 1e-20
            loss_num = 1e-20
            for idx_scene in range(fea1.size(0)):
                mask = vis_mask[idx_scene].view(-1)
                queries = fea1[idx_scene].view(fea1.size(1),-1).transpose(0,1)[mask] # N*C

                indices = (gt_corr_map[idx_scene,1].long() * gt_corr_map.size(-1) + gt_corr_map[idx_scene,0].long()).view(-1)
                positive_keys = fea2[idx_scene].view(fea2.size(1),-1).transpose(0,1)[indices][mask]

                if len(queries) < 100:
                    continue

                loss_sum = loss_sum + info_nce_loss(queries, positive_keys)
                loss_num += 1.
            loss = loss_sum / loss_num



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
            'normal_fea_ext_state_dict': normal_fea_ext.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, weight_dir+'/best.ckpt')
        print('best.ckpt is updated')

    # save weights
    torch.save({
        'idx_itr': idx_itr,
        'train_loss': np.mean(loss_dict['Train/Loss']),
        'val_loss': np.mean(loss_dict['Val/Loss']),
        'min_val_loss': min_val_loss,
        'normal_fea_ext_state_dict': normal_fea_ext.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, weight_dir+'/latest.ckpt') 

    torch.save(loss_dict, weight_dir+'/'+str(idx_itr).zfill(3)+'_loss.ckpt') 
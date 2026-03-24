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
from core.rmap_utils import sample_rmap
from core.criterion import ImageLogL1Loss, ImageGradientLoss
from core.training_utils import eval_rm_results

import numpy as np

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import glob
import argparse

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BS = 16

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-dir', type=str, default=os.environ['HOME']+'/data/tmp/deepsharm')
parser.add_argument('--wo-mask', action='store_true')
parser.add_argument('--wo-img-filtering', action='store_true')
args = parser.parse_args()

weight_dir = './weights/rm-net'
if args.wo_mask:
    weight_dir += '-wo-mask'
if args.wo_img_filtering:
    weight_dir += '-wo-img-filtering'
weight_dir += '-da'
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
rm_net = ReflectanceMapNet(wo_mask=args.wo_mask,wo_img_filtering=args.wo_img_filtering)
rm_net = nn.DataParallel(rm_net)
rm_net.to(device)

# loss functions
image_loss = ImageLogL1Loss()
image_loss.to(device)
image_grad_loss = ImageGradientLoss()
image_grad_loss.to(device)


# optimizer
optimizer = torch.optim.Adam(rm_net.parameters())

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

list_ckpt = sorted(glob.glob(weight_dir+'/latest.ckpt'))
idx_itr_ofs = 0
min_val_loss = 1e12
if len(list_ckpt) > 0:
    path = list_ckpt[-1]
    checkpoint = torch.load(path)
    print('existing checkpoint '+path+' loaded')
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    rm_net.module.load_state_dict(checkpoint['rm_net_state_dict'])
    idx_itr_ofs = checkpoint['idx_itr'] + 1
    min_val_loss = checkpoint['min_val_loss']

for idx_itr in range(idx_itr_ofs, 200):
    loss_dict = {}

    # train
    bar = tqdm(trainloader)
    bar.set_description('Epoch '+str(idx_itr+1).zfill(2)+' (train)')
    rm_net.train()
    for idx_minbatch, minbatch in enumerate(bar):
        global_steps = idx_itr * len(trainloader) + idx_minbatch
        img = minbatch['imgs'][:,0].to(device)
        normal = minbatch['normals'][:,0].to(device)
        gt_rmap = minbatch['rmaps'][:,0].to(device)
        gt_diffuse_rmap = minbatch['diffuse_rmaps'][:,0].to(device)
        gt_diffuse_img = minbatch['diffuse_imgs'][:,0].to(device)

        mask = (torch.sum(normal**2, dim=1, keepdim=True) > 0.1).float()

        max_color = torch.max(gt_rmap.view(gt_rmap.size(0),-1), dim=1)[0]
        img /= max_color[:,None,None,None]
        gt_rmap /= max_color[:,None,None,None]
        gt_diffuse_rmap /= max_color[:,None,None,None]
        gt_diffuse_img /= max_color[:,None,None,None]

        # data aug
        if True:
            scale = 0.25 + 1.5 * torch.rand((img.size(0),), device=device)
            theta = 2.0 * np.pi * torch.rand((img.size(0),), device=device)
            img = rotate_img(img, theta, scale)
            gt_diffuse_img = rotate_img(gt_diffuse_img, theta, scale)
            gt_rmap = rotate_img(gt_rmap, theta, torch.ones_like(scale))
            gt_diffuse_rmap = rotate_img(gt_diffuse_rmap, theta, torch.ones_like(scale))
            normal = rotate_normal_map(normal, theta, scale)

        # creste gt mask
        sampled_img = sample_rmap(gt_rmap, normal, projection_mode='probe')
        error_img = torch.sum(torch.abs(torch.log1p(10 * torch.clamp(img, 0, None)) - torch.log1p(10 * torch.clamp(sampled_img, 0, None))), dim=1, keepdim=True)

        gt_mask = torch.all(img > 1e-3, dim=1, keepdim=True).float()
        gt_mask = gt_mask * (torch.sum(normal**2, dim=1, keepdim=True) > 0.1**2).float()
        gt_mask = gt_mask * torch.exp(-10 * error_img**2)
        gt_mask_ = gt_mask / torch.clamp(torch.mean(gt_mask, dim=(2,3), keepdim=True), 1e-2, None)

        assert torch.all(torch.isnan(img) == False)
        assert torch.all(torch.isinf(img) == False)
        assert torch.all(torch.isnan(normal) == False)
        assert torch.all(torch.isinf(normal) == False)

        rmap_results = rm_net(img, normal)

        rmap_targets = {
            'gt_rmap': gt_rmap,
            'sampled_img': sampled_img,
            'gt_mask_': gt_mask_,
        }
        loss, loss_dict, logger = eval_rm_results(
            rmap_results, rmap_targets, loss_dict, logger, 
            global_step=global_steps,
            log_header='Train'
        )

        if False:#idx_minbatch % 100 == 0:
            rmap_result = rmap_results[-1]
            est_rmap = rmap_result['rmap']
            est_mask = rmap_result['est_mask']
            est_img = rmap_result['est_img']

            import matplotlib.pyplot as plt
            plt.subplot(2,8,1)
            nf = torch.quantile(img[0][img[0] > 0], 0.9)#torch.clamp(torch.max(gt_rmap[0]), None, 1)
            plot_hdr(img / nf)
            plt.subplot(2,8,2)
            plot_normal_map(normal)

            plt.subplot(2,8,3)
            plt.imshow(est_mask[0,0].detach().cpu())
            plt.subplot(2,8,4)
            plot_hdr(est_rmap / nf)
            #plt.subplot(2,8,5)
            #plot_hdr(est_diffuse_rmap / nf)

            plt.subplot(2,8,6)
            plot_hdr(est_img / nf)
            #plt.subplot(2,8,7)
            #plot_hdr(est_diffuse_img_wo_shadow / nf)
            #plt.subplot(2,8,8)
            #plot_hdr(est_diffuse_img / nf)

            plt.subplot(2,8,11)
            plt.imshow(gt_mask[0,0].detach().cpu())
            plt.subplot(2,8,12)
            plot_hdr(gt_rmap / nf)
            plt.subplot(2,8,13)
            plot_hdr(gt_diffuse_rmap / nf)

            plt.subplot(2,8,16)
            plot_hdr(gt_diffuse_img / nf)
            plt.show()

        # backward
        optimizer.zero_grad()        
        loss.backward()
        optimizer.step()        

        logger.add_scalar('Parameters/Interp_Sharpness', rm_net.module.mapping_module.log_query_norm.exp().item(), global_steps)

        # update bar postfix
        bar.set_postfix(
            loss=np.mean(loss_dict['Train/Total_RM_Loss']),
        )

    # val
    bar = tqdm(valloader)
    bar.set_description('Epoch '+str(idx_itr+1).zfill(2)+' (val) ')
    rm_net.eval()
    with torch.no_grad():
        for idx_minbatch, minbatch in enumerate(bar):
            global_steps = idx_itr * len(valloader) + idx_minbatch
            img = minbatch['imgs'][:,0].to(device)
            normal = minbatch['normals'][:,0].to(device)
            gt_rmap = minbatch['rmaps'][:,0].to(device)
            gt_diffuse_rmap = minbatch['diffuse_rmaps'][:,0].to(device)
            gt_diffuse_img = minbatch['diffuse_imgs'][:,0].to(device)

            mask = (torch.sum(normal**2, dim=1, keepdim=True) > 0.1).float()

            max_color = torch.max(gt_rmap.view(gt_rmap.size(0),-1), dim=1)[0]
            img /= max_color[:,None,None,None]
            gt_rmap /= max_color[:,None,None,None]
            gt_diffuse_rmap /= max_color[:,None,None,None]
            gt_diffuse_img /= max_color[:,None,None,None]

            # creste gt mask
            sampled_img = sample_rmap(gt_rmap, normal, projection_mode='probe')
            error_img = torch.sum(torch.abs(torch.log1p(10 * torch.clamp(img, 0, None)) - torch.log1p(10 * torch.clamp(sampled_img, 0, None))), dim=1, keepdim=True)

            gt_mask = torch.all(img > 1e-3, dim=1, keepdim=True).float()
            gt_mask = gt_mask * (torch.sum(normal**2, dim=1, keepdim=True) > 0.1**2).float()
            gt_mask = gt_mask * torch.exp(-10 * error_img**2)
            gt_mask_ = gt_mask / torch.clamp(torch.mean(gt_mask, dim=(2,3), keepdim=True), 1e-2, None)

            rmap_results = rm_net(img, normal)

            rmap_targets = {
                'gt_rmap': gt_rmap,
                'sampled_img': sampled_img,
                'gt_mask_': gt_mask_,
            }
            loss, loss_dict, logger = eval_rm_results(
                rmap_results, rmap_targets, loss_dict, logger, 
                global_step=global_steps,
                log_header='Val'
            )

            # update bar postfix
            bar.set_postfix(
                loss=np.mean(loss_dict['Val/Total_RM_Loss']),
            )

    if np.mean(loss_dict['Val/Total_RM_Loss']) < min_val_loss:
        min_val_loss = np.mean(loss_dict['Val/Total_RM_Loss'])

        # save weights
        torch.save({
                'idx_itr': idx_itr,
                'val_loss': np.mean(loss_dict['Val/Total_RM_Loss']),
                'min_val_loss': min_val_loss,
                'rm_net_state_dict': rm_net.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
        }, weight_dir+'/best.ckpt')
        print('best.ckpt is updated')

    # save weights
    torch.save({
            'idx_itr': idx_itr,
            'val_loss': np.mean(loss_dict['Val/Total_RM_Loss']),
            'min_val_loss': min_val_loss,
            'rm_net_state_dict': rm_net.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
    }, weight_dir+'/latest.ckpt') 

    torch.save(loss_dict, weight_dir+'/'+str(idx_itr).zfill(3)+'_loss.ckpt') 
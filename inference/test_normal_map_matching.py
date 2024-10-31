import os
import sys
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
os.environ['MKL_NUM_THREADS']='1'
os.environ['NUMEXPR_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--result-dir', type=str, default='./run/test_on_synthetic_erroneous/080')
parser.add_argument('-v1', '--first-view', type=int, default=0)
parser.add_argument('-v2', '--second-view', type=int, default=1)
parser.add_argument('--wo-data-aug', action='store_true')
parser.add_argument('--wo-gbr', action='store_true')
parser.add_argument('--wo-rm', action='store_true')
parser.add_argument('--sym', action='store_true')
parser.add_argument('--naive', action='store_true')
parser.add_argument('--prefiltering', action='store_true')
parser.add_argument('--normal-reg', action='store_true')
parser.add_argument('--gpu', type=int, default=None)
args = parser.parse_args()

if not (args.gpu is None):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(args.gpu)

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

from core.sfs_utils import plot_normal_map
from core.normal_fea_ext import NormalMapFeatureExtractor


np.random.seed(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#result_dir='./run/test_on_synthetic_erroneous/080'
result_dir=args.result_dir
wo_data_aug = args.wo_data_aug
wo_gbr = args.wo_gbr
wo_rm = args.wo_rm
sym = args.sym
naive = args.naive
prefiltering = args.prefiltering
use_nreg = args.normal_reg

out_dir = result_dir+'/corrs'
if wo_data_aug:
    out_dir+='_wo_da'
if wo_gbr:
    out_dir+='_wo_gbr'
if wo_rm:
    out_dir+='_wo_rm'
if prefiltering:
    out_dir+='_prefiltering'
if use_nreg:
    out_dir+='_nreg'
if sym:
    out_dir+='_sym'
if naive:
    out_dir+='_naive'
os.makedirs(out_dir, exist_ok=True)

idx1 = args.first_view
idx2 = args.second_view
if idx1 > idx2:
    idx1 = args.second_view
    idx2 = args.first_view

src_normal_map = result_dir+'/est_normal_sfs_final.pt'
if os.path.exists(result_dir+'/est_normal_nlsfs_final.pt'):
    src_normal_map = result_dir+'/est_normal_nlsfs_final.pt'
#src_normal_map = result_dir+'/gt_normal.pt'

normals = torch.load(src_normal_map)
gt_extrinsics = torch.load(result_dir+'/gt_extrinsic_matrices.pt').numpy()

learning_based = True
multiscale = True
visualization=False

print(normals.size())

normal1 = normals[idx1:idx1+1].to(device)
normal2 = normals[idx2:idx2+1].to(device)

normal1_ldr = (0.5*(normal1[0] + 1)).detach().cpu().numpy().transpose((1,2,0))
normal2_ldr = (0.5*(normal2[0] + 1)).detach().cpu().numpy().transpose((1,2,0))

normal1_ldr = (255. * np.clip(normal1_ldr,0.,1.)).astype(np.uint8)
normal2_ldr = (255. * np.clip(normal2_ldr,0.,1.)).astype(np.uint8)

gt_rot_1 = gt_extrinsics[idx1,:3,:3]
gt_rot_2 = gt_extrinsics[idx2,:3,:3]

mask1 = torch.any(normal1 != 0., dim=1, keepdim=True).float()
mask2 = torch.any(normal2 != 0., dim=1, keepdim=True).float()

cosine_threshold = 0.95
for ofs_v in [-1,0,1]:
    for ofs_u in [-1,0,1]:
        normal1_sifted = torch.roll(normal1,(ofs_v,ofs_u),(-2,-1))
        mask1 *= (torch.sum(normal1_sifted * normal1, dim=1, keepdim=True) > cosine_threshold).float()

        normal2_sifted = torch.roll(normal2,(ofs_v,ofs_u),(-2,-1))
        mask2 *= (torch.sum(normal2_sifted * normal2, dim=1, keepdim=True) > cosine_threshold).float()

rvec = torch.tensor([0., 0., 0.], device=device, requires_grad=True)


normal_fea_ext = NormalMapFeatureExtractor()
if wo_data_aug:
    normal_fea_ext.load_state_dict(torch.load(project_dir+'/weights/normal-fea-ext/best.ckpt')['normal_fea_ext_state_dict'])
else:
    normal_fea_ext.load_state_dict(torch.load(project_dir+'/weights/normal-fea-ext-da/best.ckpt')['normal_fea_ext_state_dict'])
for p in normal_fea_ext.parameters():
    p.requires_grad = False
normal_fea_ext.eval()
normal_fea_ext.to(device)

if multiscale:
    fea1 = []
    fea2 = []
    min_scale = 128. / normal1.size(-1)
    scale = 2.
    while scale >= min_scale:
        fea1.append(
            torch.nn.functional.interpolate(
                normal_fea_ext(
                    torch.nn.functional.interpolate(
                        normal1,
                        scale_factor=scale,
                        mode='bilinear',
                        antialias=True
                    )
                ),
                size=(normal1.size(-2), normal1.size(-1)),
                mode='bilinear',
                antialias=True
            )
        )
        fea2.append(
            torch.nn.functional.interpolate(
                normal_fea_ext(
                    torch.nn.functional.interpolate(
                        normal2,
                        scale_factor=scale,
                        mode='bilinear',
                        antialias=True
                    )
                ),
                size=(normal2.size(-2), normal2.size(-1)),
                mode='bilinear',
                antialias=True
            )
        )
        scale *= 0.5
    #fea1 = torch.cat(fea1, dim=1)
    #fea2 = torch.cat(fea2, dim=1)
    fea1 = torch.mean(torch.stack(fea1, dim=1), dim=1)
    fea2 = torch.mean(torch.stack(fea2, dim=1), dim=1)

    fea1 = fea1 / torch.sqrt(torch.sum(fea1**2, dim=1, keepdim=True) + 1e-6)
    fea2 = fea2 / torch.sqrt(torch.sum(fea2**2, dim=1, keepdim=True) + 1e-6)

else:
    fea1 = normal_fea_ext(normal1)
    fea2 = normal_fea_ext(normal2)

if False:
    for i in range(10):
        plt.subplot(4,10,1)
        plot_normal_map(normal1)
        plt.subplot(4,10,2)
        plt.imshow(mask1[0,0].cpu())
        plt.subplot(4,10,11+i)
        plt.imshow(fea1[0,i].cpu())
        plt.subplot(4,10,21)
        plot_normal_map(normal2)
        plt.subplot(4,10,22)
        plt.imshow(mask2[0,0].cpu())
        plt.subplot(4,10,31+i)
        plt.imshow(fea2[0,i].cpu())
    plt.show()


    plt.figure(figsize=(7,14))
    np.random.seed(1)
    for _ in range(8):
        u_ref = np.random.randint(normal1.size(-1))
        v_ref = np.random.randint(normal1.size(-2))
        while mask1[0,0,v_ref,u_ref] == 0:
            u_ref = np.random.randint(normal1.size(-1))
            v_ref = np.random.randint(normal1.size(-2))

        #u_ref = 56
        #v_ref = 71
        sim_img = (torch.exp(torch.sum(fea1[0,:,v_ref:v_ref+1,u_ref:u_ref+1] * fea2[0],dim=0) - 1) * mask2[0,0]).cpu()
        v_sim, u_sim = [res[0] for res in np.where(sim_img == np.max(sim_img.numpy()))]


        plt.subplot(8,4,4*_+1)
        plot_normal_map(normal1)
        plt.scatter([u_ref], [v_ref])
        if _ == 7:
            plt.xlabel('query_pt')
        plt.subplot(8,4,4*_+2)
        plot_normal_map(normal2)
        plt.scatter([u_sim], [v_sim])
        if _ == 7:
            plt.xlabel('est_match')
        if False:
            plt.subplot(8,4,4*_+3)
            plot_normal_map(normal2)
            plt.scatter([u_gt], [v_gt])
            if _ == 7:
                plt.xlabel('gt_match')
        plt.subplot(8,4,4*_+4)
        plt.imshow(sim_img, vmin=0, vmax=1)
        if _ == 7:
            plt.xlabel('sim_map')
    plt.show()


#for v_ref in range(128):
#    for u_ref in range(128):
#        if ((u_ref - 64)**2 + (v_ref - 64)**2) >= 60**2:
#            continue
#        sim_img = torch.exp(torch.sum(fea1[0,:,v_ref:v_ref+1,u_ref:u_ref+1] * fea2[0],dim=0) - 1).cpu()
#        plt.imshow(sim_img > 0.75)
#        plt.show()



np.random.seed(1)
similarity_threshold = 0.7
ratio_threshold = 0.95
distance_threshold = 10
list_ref = []
list_nn = []
normal_dir_1 = []
normal_dir_2 = []
list_sim = []
#list_gt = []
dst_map = np.zeros((normal1.size(-2),normal1.size(-1),2))
dst_map[:] = -1
#gt_dst_map = np.zeros((normal1.size(-2),normal1.size(-1),2))
#gt_dst_map[:] = -1
bar = tqdm(range(normal1.size(-1) * normal1.size(-2)))
for idx_pixel in bar:
    u_ref = idx_pixel % normal1.size(-1)
    v_ref = idx_pixel // normal1.size(-1)
    if mask1[0,0,v_ref,u_ref] == 0:
        continue

    #gt_dst_map[v_ref, u_ref] = compute_gt_correspondence(u_ref, v_ref)


    sim_img = (torch.exp(torch.sum(fea1[0,:,v_ref:v_ref+1,u_ref:u_ref+1] * fea2[0],dim=0) - 1) * mask2[0,0]).cpu()
    max_sim = np.max(sim_img.numpy())
    #if max_sim < similarity_threshold:
    #    continue
    v_sim, u_sim = [res[0] for res in np.where(sim_img == max_sim)]

    # ratio test
    dists = [(v_sim - v_)**2 + (u_sim - u_)**2 for v_, u_ in zip(*np.where(sim_img > ratio_threshold * max_sim))]
    if np.max(dists) > distance_threshold**2:
        continue

    # back projection test
    sim_img_bak = (torch.exp(torch.sum(fea1[0] * fea2[0,:,v_sim:v_sim+1,u_sim:u_sim+1],dim=0) - 1) * mask1[0,0]).cpu()
    max_sim_bak = np.max(sim_img_bak.numpy())
    v_sim_bak, u_sim_bak = [res[0] for res in np.where(sim_img_bak == max_sim_bak)]
    dist_bak = (u_ref - u_sim_bak)**2 + (v_ref - v_sim_bak)**2
    if dist_bak > distance_threshold**2:
        continue

    list_ref.append([u_ref, v_ref])
    list_nn.append([u_sim, v_sim])
    normal_dir_1.append(normal1[0,:,v_ref,u_ref].cpu().numpy())
    normal_dir_2.append(normal2[0,:,v_sim,u_sim].cpu().numpy())
    list_sim.append(max_sim)

    dst_map[v_ref, u_ref] = np.array([u_sim, v_sim])

    bar.set_postfix(num_matches=len(list_ref))

list_ref = np.array(list_ref)
list_nn = np.array(list_nn)
list_sim = np.array(list_sim)

idx_good_corrs = list_sim > (np.max(list_sim) * similarity_threshold)
list_ref = list_ref[idx_good_corrs]
list_nn = list_nn[idx_good_corrs]
list_sim = list_sim[idx_good_corrs]
normal_dir_1 = [normal_dir_1[i] for i in range(len(normal_dir_1)) if idx_good_corrs[i]]
normal_dir_2 = [normal_dir_2[i] for i in range(len(normal_dir_2)) if idx_good_corrs[i]]

print('Number of good corrs:', len(normal_dir_1))


normal_dir_1 = np.stack(normal_dir_1, axis=0) * np.array([1.,-1.,-1.])
normal_dir_1 /= np.linalg.norm(normal_dir_1, axis=-1, keepdims=True)
normal_dir_2 = np.stack(normal_dir_2, axis=0) * np.array([1.,-1.,-1.])
normal_dir_2 /= np.linalg.norm(normal_dir_2, axis=-1, keepdims=True)

np.savez(
    out_dir+'/corrs_'+str(idx1)+'_'+str(idx2)+'_nm.npz',
    normal1=normal_dir_1,
    normal2=normal_dir_2,
    coord1=list_ref,
    coord2=list_nn,
)

skip = max(1, len(list_ref) // 32)
if False:
    plt.figure(figsize=(19.2,4.8))
    plt.subplot(1,3,1)
    plot_normal_map(normal1)
    
    for u_ref, v_ref in list_ref[::skip]:
        plt.scatter([u_ref], [v_ref])

    plt.xlabel('Query Points')

    plt.subplot(1,3,2)
    plot_normal_map(normal2)
    for u_nn, v_nn in list_nn[::skip]:
        plt.scatter([u_nn], [v_nn])
    plt.xlabel('Estimated Matches')

    #plt.subplot(1,3,3)
    #plot_normal_map(normal2)
    #for u_gt, v_gt in list_gt:
    #    plt.scatter([u_gt], [v_gt])
    #plt.xlabel('GT Matches')

    plt.show()

kp1 = [cv2.KeyPoint(u,v, 1.) for u,v in list_ref.astype(np.float32)]
kp2 = [cv2.KeyPoint(u,v, 1.) for u,v in list_nn.astype(np.float32)]
#kp2_gt = [cv2.KeyPoint(u,v, 1.) for u,v in list_gt.astype(np.float32)]
matches = [cv2.DMatch(i, i, 0) for i in range(len(kp1))]

skip = max(1, len(matches) // 50)
cv2.setRNGSeed(6)
result = cv2.drawMatches(
    normal1_ldr, 
    kp1, 
    normal2_ldr, 
    kp2, 
    matches[::skip], 
    None, 
    flags=2
)
cv2.imwrite(out_dir+'/matches_nm_'+str(idx1)+'_'+str(idx2)+'.png', result[:,:,::-1])

if visualization:
    #cv2.setRNGSeed(6)
    #gt_result = cv2.drawMatches(
    #    normal1_ldr, 
    #    kp1, 
    #    normal2_ldr, 
    #    kp2_gt, 
    #    matches[::skip], 
    #    None, 
    #    flags=2
    #)
    #plt.subplot(2,1,1)
    plt.imshow(result)
    plt.ylabel('Est Matches')
    #plt.subplot(2,1,2)
    #plt.imshow(gt_result)
    #plt.ylabel('GT Matches')
    plt.show()

    plt.subplot(2,2,1)
    plot_normal_map(normal1)
    plt.ylabel('query rmap')
    plt.subplot(2,2,2)
    plot_normal_map(normal2)
    plt.ylabel('dst rmap')
    plt.subplot(2,2,3)
    plt.imshow(dst_map[:,:,0], alpha=(dst_map[:,:,0]>=0).astype(np.float))
    plt.ylabel('est dst u')
    plt.subplot(2,2,4)
    plt.imshow(dst_map[:,:,1], alpha=(dst_map[:,:,1]>=0).astype(np.float))
    plt.ylabel('est dst v')
    plt.show()
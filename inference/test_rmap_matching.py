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
parser.add_argument('--w-ms', action='store_true')
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
from core.rmap_utils import create_normal_grid
from tqdm import tqdm

from core.sfs_utils import plot_hdr
from core.rmap_fea_ext import ReflectanceMapFeatureExtractor


np.random.seed(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#result_dir='./run/test_on_synthetic_erroneous/080'
result_dir=args.result_dir

wo_data_aug = args.wo_data_aug
wo_gbr = args.wo_gbr
wo_rm = args.wo_rm
w_ms = args.w_ms
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
if w_ms:
    out_dir+='_w_ms'
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

#img = cv2.imread(result_dir+'/080/gt_rmap.exr',-1)
img = cv2.imread(result_dir+'/est_rmap_final.exr',-1)
if os.path.exists(result_dir+'/est_rmap_nlsfs_final.exr'):
    img = cv2.imread(result_dir+'/est_rmap_nlsfs_final.exr',-1)
elif os.path.exists(result_dir+'/est_rmap_sfs_final.exr'):
    img = cv2.imread(result_dir+'/est_rmap_sfs_final.exr',-1)
#img = cv2.imread(result_dir+'/gt_rmap.exr',-1)
gt_extrinsics = torch.load(result_dir+'/gt_extrinsic_matrices.pt').numpy()

multiangle=True
learning_based = True
use_ransac = False
visualization=False

reso = img.shape[0]
img1 = img[:,reso*idx1:reso*(idx1+1),::-1].copy()
img2 = img[:,reso*idx2:reso*(idx2+1),::-1].copy()

gt_rot_1 = gt_extrinsics[idx1,:3,:3]
gt_rot_2 = gt_extrinsics[idx2,:3,:3]
gt_rot_12 = torch.from_numpy(gt_rot_2 @ np.linalg.inv(gt_rot_1)).float().to(device)

img1 = torch.stack(torch.from_numpy(img1).unbind(-1),0)[None].float().to(device)
img2 = torch.stack(torch.from_numpy(img2).unbind(-1),0)[None].float().to(device)

nf = torch.quantile(torch.cat([img1,img2],dim=-1)[torch.cat([img1,img2],dim=-1) > 0], 0.9)
#print(nf)
#img1 = img1 / nf
#img2 = img2 / nf

#img1 = gaussian_blur_rmap(img1)
#img2 = gaussian_blur_rmap(img2)

img1_ldr = (255 * np.clip(torch.stack(torch.unbind(img1[0], 0), dim=-1).cpu().numpy(),0,1)**(1/2.2)).astype(np.uint8)
img2_ldr = (255 * np.clip(torch.stack(torch.unbind(img2[0], 0), dim=-1).cpu().numpy(),0,1)**(1/2.2)).astype(np.uint8)




rmap_fea_ext = ReflectanceMapFeatureExtractor()
if wo_data_aug:
    rmap_fea_ext.load_state_dict(torch.load(project_dir+'/weights/rmap-fea-ext/best.ckpt')['rmap_fea_ext_state_dict'])
elif w_ms:
    rmap_fea_ext.load_state_dict(torch.load(project_dir+'/weights/rmap-fea-ext-da-ms/best.ckpt')['rmap_fea_ext_state_dict'])
else:
    rmap_fea_ext.load_state_dict(torch.load(project_dir+'/weights/rmap-fea-ext-da-ms/best.ckpt')['rmap_fea_ext_state_dict'])
for p in rmap_fea_ext.parameters():
    p.requires_grad = False
rmap_fea_ext.eval()
rmap_fea_ext.to(device)

# img: BS*C*H*W
# theta: BS
def rotate_img(img, theta, scale):
    BS,C,H,W = img.size()
    v,u = torch.meshgrid(torch.arange(H), torch.arange(W))
    x = 2 * (u.to(img.device) + 0.5) / W - 1.0
    y = 2 * (v.to(img.device) + 0.5) / H - 1.0

    x_ = np.cos(theta) * x - np.sin(theta) * y
    y_ = np.sin(theta) * x + np.cos(theta) * y

    grid = scale * torch.stack([x_, y_], dim=-1)[None]
    img_ = torch.nn.functional.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    return img_

if multiangle:
    fea1 = []
    fea2 = []
    for theta in range(0,360,15):
        fea1.append(rotate_img(rmap_fea_ext(rotate_img(img1, np.radians(theta), 1.)), np.radians(-theta), 1.))
        fea2.append(rotate_img(rmap_fea_ext(rotate_img(img2, np.radians(theta), 1.)), np.radians(-theta), 1.))

    fea1 = torch.mean(torch.stack(fea1, dim=1), dim=1)
    fea2 = torch.mean(torch.stack(fea2, dim=1), dim=1)

    fea1 = fea1 / torch.sqrt(torch.sum(fea1**2, dim=1, keepdim=True) + 1e-6)
    fea2 = fea2 / torch.sqrt(torch.sum(fea2**2, dim=1, keepdim=True) + 1e-6)
else:
    fea1 = rmap_fea_ext(img1)
    fea2 = rmap_fea_ext(img2)

if visualization:
    for i in range(10):
        plt.subplot(4,10,1)
        plot_hdr(img1)
        plt.subplot(4,10,11+i)
        plt.imshow(fea1[0,i].cpu())
        plt.subplot(4,10,21)
        plot_hdr(img2)
        plt.subplot(4,10,31+i)
        plt.imshow(fea2[0,i].cpu())
    plt.show()


def compute_gt_correspondence(u_ref, v_ref):
    normal_grid = create_normal_grid(img1.size(-1),'probe').numpy()
    n_ref = normal_grid[v_ref,u_ref]
    n_ref_ = n_ref * np.array([1.,-1.,-1.])
    n_dst_ = (gt_rot_2 @ np.linalg.inv(gt_rot_1) @ n_ref_[:,None])[:,0]
    n_dst = n_dst_ * np.array([1.,-1.,-1.])

    phi = np.arccos(np.clip(n_dst[2],-0.999999,0.999999))
    rn = np.sqrt(n_dst[0]**2 + n_dst[1]**2 + 1e-6)
    r = phi / np.pi / rn
    u = n_dst[0] * r
    v = -n_dst[1] * r
    u = 0.5 * (u + 1) * img1.size(-1)
    v = 0.5 * (v + 1) * img1.size(-2)
    #print(n_dst, normal_grid[int(v),int(u)])
    return u,v

if False:
    plt.figure(figsize=(7,14))
    np.random.seed(1)
    for _ in range(8):
        u_ref = np.random.randint(img1.size(-1))
        v_ref = np.random.randint(img1.size(-2))
        while ((u_ref - 64)**2 + (v_ref - 64)**2) >= 60**2:
            u_ref = np.random.randint(img1.size(-1))
            v_ref = np.random.randint(img1.size(-2))

        #u_ref = 56
        #v_ref = 71
        sim_img = torch.exp(torch.sum(fea1[0,:,v_ref:v_ref+1,u_ref:u_ref+1] * fea2[0],dim=0) - 1).cpu()
        v_sim, u_sim = [res[0] for res in np.where(sim_img == np.max(sim_img.numpy()))]

        u_gt, v_gt = compute_gt_correspondence(u_ref, v_ref)

        plt.subplot(8,4,4*_+1)
        plot_hdr(img1)
        plt.scatter([u_ref], [v_ref])
        if _ == 7:
            plt.xlabel('query_pt')
        plt.subplot(8,4,4*_+2)
        plot_hdr(img2)
        plt.scatter([u_sim], [v_sim])
        if _ == 7:
            plt.xlabel('est_match')
        if True:
            plt.subplot(8,4,4*_+3)
            plot_hdr(img2)
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
ratio_threshold = 0.9
cosine_threshold = 0.9
zenith_threshold = np.radians(170)
normal_grid = create_normal_grid(img1.size(-1),'probe').numpy()
normal_grid *= np.array([1.,-1.,-1.])
rmap_mask = (torch.from_numpy(np.sum(normal_grid**2, axis=-1)).to(device) > 0.25).float()

list_ref = []
list_nn = []
list_sim = []
list_gt = []
light_dir_1 = []
light_dir_2 = []
gt_light_dir_2 = []
dst_map = np.zeros((img1.size(-2),img1.size(-1),2))
dst_map[:] = -1
gt_dst_map = np.zeros((img1.size(-2),img1.size(-1),2))
gt_dst_map[:] = -1
bar = tqdm(range(img1.size(-1) * img1.size(-2)))
for idx_pixel in bar:
    u_ref = idx_pixel % img1.size(-1)
    v_ref = idx_pixel // img1.size(-1)
    zenith_ref = np.sqrt((u_ref - 64)**2 + (v_ref - 64)**2) / 64. * np.pi
    if zenith_ref > zenith_threshold:
        continue

    gt_dst_map[v_ref, u_ref] = compute_gt_correspondence(u_ref, v_ref)

    sim_img = (torch.exp(-torch.sum((fea1[0,:,v_ref:v_ref+1,u_ref:u_ref+1] - fea2[0])**2,dim=0)) * rmap_mask).cpu()
    max_sim = np.max(sim_img.numpy())
    #if max_sim < similarity_threshold:
    #    continue
    v_sim, u_sim = [res[0] for res in np.where(sim_img == max_sim)]
    n_sim = normal_grid[v_sim, u_sim]

    zenith_sim = np.sqrt((u_sim - 64)**2 + (v_sim - 64)**2) / 64. * np.pi
    if zenith_sim > zenith_threshold:
        continue

    # ratio test
    cosines = [np.sum(n_sim * normal_grid[v_, u_]) for v_, u_ in zip(*np.where(sim_img > ratio_threshold * max_sim))]
    if np.min(cosines) < cosine_threshold:
        continue

    # back projection test
    sim_img_bak = (torch.exp(-torch.sum((fea1[0] - fea2[0,:,v_sim:v_sim+1,u_sim:u_sim+1])**2,dim=0)) * rmap_mask).cpu()
    max_sim_bak = np.max(sim_img_bak.numpy())
    v_sim_bak, u_sim_bak = [res[0] for res in np.where(sim_img_bak == max_sim_bak)]
    n_sim_bak = normal_grid[v_sim_bak, u_sim_bak]
    n_ref = normal_grid[v_ref,u_ref]
    cosine_bak = np.sum(n_ref * n_sim_bak)
    if cosine_bak < cosine_threshold:
        continue

    u_gt, v_gt = compute_gt_correspondence(u_ref, v_ref)

    list_ref.append([u_ref, v_ref])
    list_nn.append([u_sim, v_sim])
    list_gt.append(compute_gt_correspondence(u_ref, v_ref))
    list_sim.append(max_sim)

    light_dir_1.append(normal_grid[v_ref, u_ref])
    light_dir_2.append(normal_grid[v_sim, u_sim])
    gt_light_dir_2.append(normal_grid[int(v_gt), int(u_gt)])

    dst_map[v_ref, u_ref] = np.array([u_sim, v_sim])

    bar.set_postfix(num_matches=len(list_ref))

list_sim = np.array(list_sim)
idx_good_corrs = list_sim > (np.max(list_sim) * similarity_threshold)

list_ref = np.array(list_ref)[idx_good_corrs]
list_nn = np.array(list_nn)[idx_good_corrs]
list_gt = np.array(list_gt)[idx_good_corrs]

light_dir_1 = np.array(light_dir_1)[idx_good_corrs]
light_dir_2 = np.array(light_dir_2)[idx_good_corrs]
gt_light_dir_2 = np.array(gt_light_dir_2)[idx_good_corrs]

print('Number of good corrs:', len(light_dir_1))


light1 = torch.from_numpy(light_dir_1).float().to(device)
light2 = torch.from_numpy(light_dir_2).float().to(device)
light2_gt = torch.from_numpy(gt_light_dir_2).float().to(device)

normal_dir_1 = light_dir_1 + np.array([0., 0., -1.,])
normal_dir_1 /= np.linalg.norm(normal_dir_1, axis=-1, keepdims=True)
normal_dir_2 = light_dir_2 + np.array([0., 0., -1.,])
normal_dir_2 /= np.linalg.norm(normal_dir_2, axis=-1, keepdims=True)

#gt_normal_dir_1 = light_dir_1 + np.array([0., 0., -1.,])
#gt_normal_dir_1 /= np.linalg.norm(gt_normal_dir_1, axis=-1, keepdims=True)
#gt_normal_dir_2 = gt_light_dir_2 + np.array([0., 0., -1.,])
#gt_normal_dir_2 /= np.linalg.norm(gt_normal_dir_2, axis=-1, keepdims=True)

np.savez(
    out_dir+'/corrs_'+str(idx1)+'_'+str(idx2)+'_rm.npz',
    normal1=normal_dir_1,
    normal2=normal_dir_2,
    rm_coord1=list_ref,
    rm_coord2=list_nn,
    fea1=fea1.detach().cpu().numpy(),
    fea2=fea2.detach().cpu().numpy(),
)

assert all(normal_dir_1[:,2] <= 0)

# compute R that satisfies R@v1.T = v2.T
def compute_rot(v1, v2):
    X = v1.detach().cpu().numpy().T
    Y = v2.detach().cpu().numpy().T
    U,s,Vt = np.linalg.svd(X@(Y.T))
    H = np.diag([1.0, 1.0, np.linalg.det(Vt.T@U.T)])
    return torch.from_numpy(Vt.T@H@U.T).float().to(device)

est_rot_12 = compute_rot(light1, light2)
view_error = torch.acos(torch.clamp(torch.sum(est_rot_12[2] * gt_rot_12[2]),-1,1)).item()
print('estimated relative rotation:')
print(est_rot_12.cpu().numpy())
print('GT relative rotation:')
print(gt_rot_12.cpu().numpy())
print('Angular Error of estimated viewing direction:', np.degrees(view_error), 'deg')

# RANSAC
if use_ransac:
    print('RANSAC')

    best_light1_in = []
    best_light2_in = []
    best_score = 0.0
    bar = tqdm(range(10000))
    for _ in bar:
        # random sampling of n matches from all detected matches
        idx_ = np.random.choice(len(light1), 10)
        light1_ = light1[idx_]
        light2_ = light2[idx_]

        # compute the relative pose (a rotation matrix) using the sampled matches
        maybe_rot_21 = torch.inverse(compute_rot(light1_, light2_))

        # gather inliers
        light2a = (maybe_rot_21 @ light2.T).T
        dp = torch.sum(light2a * light1, dim=-1)
        maybe_idx_in = dp > 0.95
        maybe_light1_in = light1[maybe_idx_in]
        maybe_light2_in = light2[maybe_idx_in]

        sa1 = (2 * np.pi)**2 / (128 * 128) * torch.sinc(torch.acos(torch.clamp(-light1[:,2], -0.99999, 0.99999)) / np.pi)
        sa2 = (2 * np.pi)**2 / (128 * 128) * torch.sinc(torch.acos(torch.clamp(-light2[:,2], -0.99999, 0.99999)) / np.pi)

        score = torch.sum(maybe_idx_in.float() * sa1 * sa2).item()

        if score > 0*np.pi: # some weighting according to geodesic distance between viewing directions and normals in the detected matches may be needed
            # simply evaluate goodness by the number of inliers
            # (an alternative is to evaluate residual errors of inliers)
            if score > best_score:
                best_score = score
                best_light1_in = maybe_light1_in
                best_light2_in = maybe_light2_in
                best_idx_in = maybe_idx_in

        bar.set_postfix(num_inliers = len(best_light1_in), score=best_score)

    light1 = best_light1_in
    light2 = best_light2_in
    list_ref = list_ref[best_idx_in.cpu().numpy()]
    list_nn = list_nn[best_idx_in.cpu().numpy()]
    list_gt = list_gt[best_idx_in.cpu().numpy()]

est_rot_12 = compute_rot(light1, light2)
view_error = torch.acos(torch.clamp(torch.sum(est_rot_12[2] * gt_rot_12[2]),-1,1)).item()
print('estimated relative rotation:')
print(est_rot_12.cpu().numpy())
print('GT relative rotation:')
print(gt_rot_12.cpu().numpy())
print('Angular Error of estimated viewing direction:', np.degrees(view_error), 'deg')

kp1 = [cv2.KeyPoint(u,v, 1.) for u,v in list_ref.astype(np.float32)]
kp2 = [cv2.KeyPoint(u,v, 1.) for u,v in list_nn.astype(np.float32)]
kp2_gt = [cv2.KeyPoint(u,v, 1.) for u,v in list_gt.astype(np.float32)]
matches = [cv2.DMatch(i, i, 0) for i in range(len(kp1))]
skip = max(1, len(matches) // 50)

cv2.setRNGSeed(6)
result = cv2.drawMatches(
    img1_ldr, 
    kp1, 
    img2_ldr, 
    kp2, 
    matches[::skip], 
    None, 
    flags=2
)
cv2.imwrite(out_dir+'/matches_rm_'+str(idx1)+'_'+str(idx2)+'.png', result[:,:,::-1])

if visualization:
    skip = max(1, len(list_ref) // 32)

    if False:
        plt.figure(figsize=(19.2,4.8))
        plt.subplot(1,3,1)
        plot_hdr(img1)
        for u_ref, v_ref in list_ref[::skip]:
            plt.scatter([u_ref], [v_ref])
        plt.xlabel('Query Points')

        plt.subplot(1,3,2)
        plot_hdr(img2)
        for u_nn, v_nn in list_nn[::skip]:
            plt.scatter([u_nn], [v_nn])
        plt.xlabel('Estimated Matches')

        plt.subplot(1,3,3)
        plot_hdr(img2)
        for u_gt, v_gt in list_gt[::skip]:
            plt.scatter([u_gt], [v_gt])
        plt.xlabel('GT Matches')

        plt.show()

    cv2.setRNGSeed(6)
    gt_result = cv2.drawMatches(
        img1_ldr, 
        kp1, 
        img2_ldr, 
        kp2_gt, 
        matches[::skip], 
        None, 
        flags=2
    )
    plt.subplot(2,1,1)
    plt.imshow(result)
    plt.ylabel('Est Matches')
    plt.subplot(2,1,2)
    plt.imshow(gt_result)
    plt.ylabel('GT Matches')
    plt.show()

    plt.subplot(3,2,1)
    plot_hdr(img1)
    plt.ylabel('query rmap')
    plt.subplot(3,2,2)
    plot_hdr(img2)
    plt.ylabel('dst rmap')
    plt.subplot(3,2,3)
    plt.imshow(dst_map[:,:,0], alpha=(dst_map[:,:,0]>=0).astype(np.float))
    plt.ylabel('est dst u')
    plt.subplot(3,2,4)
    plt.imshow(dst_map[:,:,1], alpha=(dst_map[:,:,1]>=0).astype(np.float))
    plt.ylabel('est dst v')
    plt.subplot(3,2,5)
    plt.imshow(gt_dst_map[:,:,0], alpha=(gt_dst_map[:,:,0]>=0).astype(np.float))
    plt.ylabel('gt dst u')
    plt.subplot(3,2,6)
    plt.imshow(gt_dst_map[:,:,1], alpha=(gt_dst_map[:,:,0]>=0).astype(np.float))
    plt.ylabel('gt dst v')
    plt.show()
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
parser.add_argument('object_id', type=int, default=0)
parser.add_argument('-v1', '--first-view', type=int, default=0)
parser.add_argument('-v2', '--second-view', type=int, default=1)
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
import json

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset

from core.sfs_utils import plot_hdr
from core.sfm_utils import solve_surf_point_locations_perspective, matrix_to_euler
from core.sfm_utils import res_func_pixel, euler_to_matrix, solve_translation_perspective
from core.dataset import TwoViewRealImageDataset

from lightglue import LightGlue, SuperPoint#, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd

np.random.seed(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

object_id=args.object_id
dataset_dir='/home/kyamashita/data/scenes_with_reflective_objects/out'
img_dir=dataset_dir+'/'+str(object_id).zfill(5)
#result_dir='./run/test_on_synthetic_erroneous/080'

out_dir = './run/image_matching_lightglue_ortho/'+str(object_id).zfill(2)
os.makedirs(out_dir, exist_ok=True)

idx1 = args.first_view
idx2 = args.second_view
if idx1 > idx2:
    idx1 = args.second_view
    idx2 = args.first_view

img1 = cv2.imread(img_dir+'/view-'+str(1+idx1).zfill(2)+'.exr',-1)[:,:,::-1]
img2 = cv2.imread(img_dir+'/view-'+str(1+idx2).zfill(2)+'.exr',-1)[:,:,::-1]

mask1_nl = cv2.imread(img_dir+'/view-'+str(1+idx1).zfill(2)+'_m.png',0)
mask2_nl = cv2.imread(img_dir+'/view-'+str(1+idx2).zfill(2)+'_m.png',0)


mask1 = cv2.imread(img_dir+'/view-'+str(1+idx1).zfill(2)+'_m2.png',0)# * (mask1_nl == 0)
mask2 = cv2.imread(img_dir+'/view-'+str(1+idx2).zfill(2)+'_m2.png',0)# * (mask2_nl == 0)

if True:
    img1 = img1 * (mask1[...,None] > 0).astype(float)
    img2 = img2 * (mask2[...,None] > 0).astype(float)

gt_rot_1 = np.load(img_dir+'/view-'+str(1+idx1).zfill(2)+'_extrinsics.npy')[:3,:3]
gt_rot_2 = np.load(img_dir+'/view-'+str(1+idx2).zfill(2)+'_extrinsics.npy')[:3,:3]

K1 = np.load(img_dir+'/view-'+str(1+idx1).zfill(2)+'_intrinsics.npy')[:3,:3]
K2 = np.load(img_dir+'/view-'+str(1+idx2).zfill(2)+'_intrinsics.npy')[:3,:3]

multiangle=True
learning_based = True
use_ransac = False
visualization=False

gt_rot_12 = torch.from_numpy(gt_rot_2 @ np.linalg.inv(gt_rot_1)).float().to(device)

img1 = torch.stack(torch.from_numpy(img1).unbind(-1),0)[None].float().to(device)
img2 = torch.stack(torch.from_numpy(img2).unbind(-1),0)[None].float().to(device)

img1_small = torch.nn.functional.interpolate(img1,scale_factor=0.25)
img2_small = torch.nn.functional.interpolate(img2,scale_factor=0.25)

#img1 = img1_small
#img2 = img2_small

nf = torch.quantile(torch.cat([
    img1_small,
    img2_small,
],dim=-1)[torch.cat([img1_small,img2_small],dim=-1) > 0], 0.9)
#print(nf)
#img1 = img1 / nf
#img2 = img2 / nf

#img1 = gaussian_blur_rmap(img1)
#img2 = gaussian_blur_rmap(img2)

img1_ldr = (255 * np.clip(torch.stack(torch.unbind(img1[0], 0), dim=-1).cpu().numpy(),0,1)**(1/2.2)).astype(np.uint8)
img2_ldr = (255 * np.clip(torch.stack(torch.unbind(img2[0], 0), dim=-1).cpu().numpy(),0,1)**(1/2.2)).astype(np.uint8)

img_diagonal = np.sqrt(img1.size(-1)**2+img1.size(-2)**2)


cv2.imwrite(out_dir+'/img_'+str(idx1)+'.png', cv2.cvtColor(img1_ldr, cv2.COLOR_RGB2BGR))
cv2.imwrite(out_dir+'/img_'+str(idx2)+'.png', cv2.cvtColor(img2_ldr, cv2.COLOR_RGB2BGR))

# SuperPoint+LightGlue
extractor = SuperPoint(max_num_keypoints=2048*16).eval().cuda()  # load the extractor
matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher

# or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
#extractor = DISK(max_num_keypoints=2048*16).eval().cuda()  # load the extractor
#matcher = LightGlue(features='disk').eval().cuda()  # load the matcher

# load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
image0 = load_image(out_dir+'/img_'+str(idx1)+'.png').cuda()
image1 = load_image(out_dir+'/img_'+str(idx2)+'.png').cuda()

# extract local features
feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
feats1 = extractor.extract(image1)

# match the features
matches01 = matcher({'image0': feats0, 'image1': feats1})
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
matches = matches01['matches']  # indices with shape (K,2)
points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

pixel1 = points0.cpu().numpy()
pixel2 = points1.cpu().numpy()


# estimation of phi & theta
th_pix=5
def eval_param_ortho(param):
    phi, theta = param
    err_pix = res_func_pixel((phi, theta), pixel1 , pixel2)

    score_nm = np.sum(
        np.exp(-(err_pix / th_pix)**2)
    ) / len(pixel1)

    idx_in_nm = (np.abs(err_pix) < th_pix)

    return score_nm, idx_in_nm

list_theta_deg = np.arange(0,180)
list_phi_deg = np.arange(0,180)
list_score = []
best_score = 0
for theta_deg in list_theta_deg:
    theta = np.radians(theta_deg)
    list_score_ = []
    for phi_deg in list_phi_deg:
        phi = np.radians(phi_deg)
        score, idx_in = eval_param_ortho((phi, theta))
        list_score_.append(score)
        if score > best_score:
            best_score = score
            best_param = (phi,theta)
            best_idx_in = idx_in
    list_score.append(list_score_)
score_map = np.array(list_score)

if True:
    import scipy
    def objective(param):
        return np.exp(-eval_param_ortho(param)[0])
    result = scipy.optimize.least_squares(objective, best_param)
    best_param = result.x
    best_scorem, best_idx_in = eval_param_ortho(best_param)

piexl1 = pixel1[best_idx_in]
piexl2 = pixel2[best_idx_in]

# draw matches
kp1 = [cv2.KeyPoint(u,v, 1.) for u,v in pixel1.astype(np.float32)]
kp2 = [cv2.KeyPoint(u,v, 1.) for u,v in pixel2.astype(np.float32)]
matches = [cv2.DMatch(i, i, 0) for i in range(len(kp1))]

skip = max(1, len(matches) // 1000)
cv2.setRNGSeed(6)
img_inliers = cv2.drawMatches(
    img1_ldr, 
    kp1, 
    img2_ldr, 
    kp2, 
    matches[::skip], 
    None, 
    flags=2
)
cv2.imwrite(out_dir+'/matches_pix_in_ortho_'+str(idx1)+'_'+str(idx2)+'.png', img_inliers[:,:,::-1])

print('score (ortho):',best_score)
print('(phi,theta):',np.degrees(best_param))
print('(phi_gt,theta_gt):',np.degrees(matrix_to_euler(gt_rot_12.T.detach().cpu().numpy())))
if False:
    plt.imshow(score_map)
    plt.show()

# estimation based on perspective model
phi, theta = best_param
bar = tqdm(np.arange(0,360,1))
list_score = []
th_pix_perspective=5
best_score = 0
for eta_deg in bar:
    eta = np.radians(eta_deg)
    R21 = euler_to_matrix(phi, eta, theta)
    if True:
        t21 = solve_translation_perspective(pixel1, pixel2,K1,K2,R21,loss='soft_l1')
    else:
        x2 = (R21 @ np.linalg.inv(K2) @ np.concatenate([pixel2,np.ones_like(pixel2[:,:1])],axis=1).T).T # N*3
        x1 = (np.linalg.inv(K2) @ np.concatenate([pixel1,np.ones_like(pixel1[:,:1])],axis=1).T).T # N*3
        X = x1[:,:,None] @ x2[:,None,:] # N*3*3
        A = np.stack([
            -X[:,1,2] + X[:,2,1],
            -X[:,2,0] + X[:,0,2],
            -X[:,0,1] + X[:,1,0]
        ], axis=1) # N*3
        U,s,Vt = np.linalg.svd(A)
        t21 = Vt[np.argmin(s**2)]

        residual_epipolar = np.min(s**2)
    surf_points = solve_surf_point_locations_perspective(pixel1, pixel2, K1, K2, R21, t21)
    if np.sum(surf_points[:,2] < 0) > (0.5 * len(surf_points)):
        t21 *= -1
        surf_points *= -1

    pixel1_reproj_ = (K1 @ surf_points.T).T
    pixel1_reproj_depth = pixel1_reproj_[:,2]
    pixel1_reproj = pixel1_reproj_[:,:2] / pixel1_reproj_[:,2:3]

    pixel2_reproj_ = (K2 @ (R21.T @ (surf_points.T - t21[:,None]))).T
    pixel2_reproj_depth = pixel2_reproj_[:,2]
    pixel2_reproj = pixel2_reproj_[:,:2] / pixel2_reproj_[:,2:3]

    reproj_error = 0.5 * (np.linalg.norm(pixel1 - pixel1_reproj, axis=1) + np.linalg.norm(pixel2 - pixel2_reproj, axis=1))

    score = np.sum(
        np.exp(-(reproj_error / th_pix_perspective)**2)
          * (pixel1_reproj_depth > 0) * (pixel2_reproj_depth > 0)
    ) / len(pixel1)

    if np.cos(eta) > np.cos(np.radians(10)):
        score = 0

    idx_in = (np.abs(reproj_error) < th_pix_perspective)

    if score > best_score:
        best_score = score
        best_idx_in = idx_in
        best_eta = eta
        best_t21 = t21

    if False:
        print(reproj_error)
        print(np.median(reproj_error))
        plt.subplot(1,2,1)
        plt.scatter(pixel1[:,0],pixel1[:,1])
        plt.scatter(pixel1_reproj[:,0],pixel1_reproj[:,1])
        plt.subplot(1,2,2)
        plt.scatter(pixel2[:,0],pixel2[:,1])
        plt.scatter(pixel2_reproj[:,0],pixel2_reproj[:,1])
        plt.show()

    list_score.append(score)

eta = best_eta
R21 = euler_to_matrix(phi, eta, theta)
t21 = best_t21

piexl1 = pixel1[best_idx_in]
piexl2 = pixel2[best_idx_in]

surf_points = solve_surf_point_locations_perspective(pixel1, pixel2, K1, K2, R21, t21)

if True:
    import open3d as o3d
    pcd_est = o3d.geometry.PointCloud()
    pcd_est.points = o3d.utility.Vector3dVector(surf_points)

    o3d.io.write_point_cloud(out_dir+'/surf_points_0.ply', pcd_est)

print('score (perspective):',best_score)
print('est_euler_angles:',np.degrees(matrix_to_euler(R21)))
print('gt_euler_angles:',np.degrees(matrix_to_euler(gt_rot_12.T.detach().cpu().numpy())))

if True:
    plt.plot(list_score)
    plt.savefig(out_dir+'/eta_vs_score.png')
    plt.close()

# draw inlier matches
kp1 = [cv2.KeyPoint(u,v, 1.) for u,v in pixel1.astype(np.float32)]
kp2 = [cv2.KeyPoint(u,v, 1.) for u,v in pixel2.astype(np.float32)]
matches = [cv2.DMatch(i, i, 0) for i in range(len(kp1))]

skip = max(1, len(matches) // 1000)
cv2.setRNGSeed(6)
img_inliers = cv2.drawMatches(
    img1_ldr, 
    kp1, 
    img2_ldr, 
    kp2, 
    matches[::skip], 
    None, 
    flags=2
)
cv2.imwrite(out_dir+'/matches_pix_in_perspective_'+str(idx1)+'_'+str(idx2)+'.png', img_inliers[:,:,::-1])

# adjust world offset & scale
v1,u1 = np.where(mask1_nl > 0)
pixel_cen_1 = np.array([np.mean(u1), np.mean(v1)])
v2,u2 = np.where(mask2_nl > 0)
pixel_cen_2 = np.array([np.mean(u2), np.mean(v2)])
object_center = solve_surf_point_locations_perspective(pixel_cen_1[None], pixel_cen_2[None], K1, K2, R21, t21)[0]
object_center_2 = (object_center - t21) @ R21
print('object center:', object_center)

# adjust world scale
surf_points_proj_1_ = (K1 @ surf_points.T).T
surf_points_proj_1 = surf_points_proj_1_[:,:2] / surf_points_proj_1_[:,2:3]
surf_points_proj_2_ = (K1 @ R21.T @ (surf_points.T - t21[:,None])).T
surf_points_proj_2 = surf_points_proj_2_[:,:2] / surf_points_proj_2_[:,2:3]
surf_points_mask_1 = [mask1_nl[int(v),int(u)] > 0 for u,v in surf_points_proj_1]
surf_points_mask_2 = [mask2_nl[int(v),int(u)] > 0 for u,v in surf_points_proj_2]

surf_points_nl = surf_points[surf_points_mask_1]

m1 = np.stack([
    u1 * object_center[2],
    v1 * object_center[2],
    np.ones_like(u1) * object_center[2],
], axis=0)
pts_nl_1 = (np.linalg.inv(K1) @ m1).T
m2 = np.stack([
    u2 * object_center_2[2],
    v2 * object_center_2[2],
    np.ones_like(u2) * object_center_2[2],
], axis=0)
pts_nl_2 = (R21 @ np.linalg.inv(K2) @ m2 + t21[:,None]).T
pts_nl = np.concatenate([surf_points_nl, pts_nl_1,pts_nl_2],axis=0)
if True:
    import open3d as o3d
    pcd_est = o3d.geometry.PointCloud()
    pcd_est.points = o3d.utility.Vector3dVector(pts_nl)

    o3d.io.write_point_cloud(out_dir+'/coarse_points_nl_0.ply', pcd_est)
object_radius = np.max(np.linalg.norm(pts_nl - object_center,axis=1))


print('object radius:', object_radius)

# adjust scale
world_scale_factor = 1. / (1.4 * object_radius)
t21 *= world_scale_factor
object_center *= world_scale_factor
object_radius *= world_scale_factor
surf_points *= world_scale_factor
pts_nl *= world_scale_factor

# adjust offset
world_center = object_center.copy()
t1 = world_center
t2 = (R21.T @ (t1 - t21)[:,None])[:,0]
surf_points -= world_center
pts_nl -= world_center
object_center -= world_center

R1 = np.eye(3)
R2 = R21.T

T1 = np.concatenate([np.concatenate([R1, t1[:,None]], axis=1), np.array([[0.,0.,0.,1.]])], axis=0)
T2 = np.concatenate([np.concatenate([R2, t2[:,None]], axis=1), np.array([[0.,0.,0.,1.]])], axis=0)

# y-axis = up
M = np.diag([1.,-1.,-1.,1.])
T1 = T1 @ M
T2 = T2 @ M
T21 = T1 @ np.linalg.inv(T2)
R1 = T1[:3,:3]
t1 = T1[:3,3]
R2 = T2[:3,:3]
t2 = T2[:3,3]
R21 = T21[:3,:3]
t21 = T21[:3,3]
surf_points *= np.array([1.,-1.,-1.])
pts_nl *= np.array([1.,-1.,-1.])
object_center *= np.array([1.,-1.,-1.])

P1 = np.concatenate([K1 @ np.concatenate([R1, t1[:,None]], axis=1), np.array([[0.,0.,0.,1.]])], axis=0)
P2 = np.concatenate([K2 @ np.concatenate([R2, t2[:,None]], axis=1), np.array([[0.,0.,0.,1.]])], axis=0)

phi, eta, theta = matrix_to_euler(R21)

surf_points_ = cv2.triangulatePoints(P1[:3,:4],P2[:3,:4],pixel1.T,pixel2.T)
surf_points = (surf_points_[:3] / surf_points_[3:4]).T

if True:
    import open3d as o3d
    pcd_est = o3d.geometry.PointCloud()
    pcd_est.points = o3d.utility.Vector3dVector(surf_points)

    o3d.io.write_point_cloud(out_dir+'/surf_points.ply', pcd_est)
if True:
    import open3d as o3d
    pcd_est = o3d.geometry.PointCloud()
    pcd_est.points = o3d.utility.Vector3dVector(pts_nl)

    o3d.io.write_point_cloud(out_dir+'/coarse_points_nl.ply', pcd_est)

# load intrinsic params for cropped images
with open('./confs/test_joint_opt_real_scene.json','r') as f:
    confs = json.load(f)
dataset_path = confs['dataset_path'] #os.environ['HOME']+'/data/mvs_eval/rendered'
dataset = TwoViewRealImageDataset(dataset_path, object_id, **confs['dataset_confs'])
subset_ofs = 0
subset_len = 2

list_split = np.arange(len(dataset))
test_subset_indices =  list_split[subset_ofs:subset_ofs+subset_len][int(confs['view_start']):int(confs['view_end']):int(confs['view_skip'])]
test_dataset = Subset(dataset, test_subset_indices)

K1_cropped = test_dataset[0]['intrinsics'][0].numpy()
K2_cropped = test_dataset[1]['intrinsics'][0].numpy()

P1_cropped = np.concatenate([K1_cropped @ np.concatenate([R1, t1[:,None]], axis=1), np.array([[0.,0.,0.,1.]])], axis=0)
P2_cropped = np.concatenate([K2_cropped @ np.concatenate([R2, t2[:,None]], axis=1), np.array([[0.,0.,0.,1.]])], axis=0)

pixel1_cropped_ = (P1_cropped[:3,:3] @ pts_nl.T + P1_cropped[:3,3:4]).T
pixel1_cropped = pixel1_cropped_[:,:2] / pixel1_cropped_[:,2:3]
pixel2_cropped_ = (P2_cropped[:3,:3] @ pts_nl.T + P2_cropped[:3,3:4]).T
pixel2_cropped = pixel2_cropped_[:,:2] / pixel2_cropped_[:,2:3]
#plt.scatter(pixel1_cropped[:,0],pixel1_cropped[:,1])
#plt.show()


print(P1_cropped)
print(P2_cropped)

np.savez(
    out_dir+'/est_pose_'+str(idx1)+'_'+str(idx2)+'.npz',
    coord1=pixel1_cropped,
    coord2=pixel2_cropped,
    R21=R21,
    phi=phi,
    eta=eta,
    theta=theta,
    t21=t21,
    R1=R1,
    R2=R2,
    P1=P1_cropped,
    P2=P2_cropped,
    K1=K1_cropped,
    K2=K2_cropped,
    t1=t1,
    t2=t2,
    l1=1.,
    m1=0.,
    n1=0.,
    l2=1.,
    m2=0.,
    n2=0.,
)

def compute_pose_error(R_est, R_gt):
    return np.arccos(np.clip(0.5 * (np.trace(R_est @ R_gt.T) - 1.), -1., 1.))

result_text = ''
result_text += 'R21_error_deg: '+str(np.degrees(compute_pose_error(R21, gt_rot_12.cpu().numpy().T)))+'\n'
result_text += 'euler_angles_deg_est: '+' '.join([str(np.degrees(v)) for v in matrix_to_euler(R21)])+'\n'
result_text += 'euler_angles_deg_gt: '+' '.join([str(np.degrees(v)) for v in matrix_to_euler(gt_rot_12.cpu().numpy().T)])+'\n'
with open(out_dir+'/accuracy_0_1.txt', 'w') as f:
    f.write(result_text)

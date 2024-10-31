import os
import sys
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
os.environ['MKL_NUM_THREADS']='1'
os.environ['NUMEXPR_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

import scipy

from core.sfs_utils import plot_hdr, save_normal_map, save_hdr, save_hdr_as_ldr
from core.sfm_utils import res_func_pixel, matrix_to_euler, compute_relative_rot, compute_pose_error, solve_translation_perspective, solve_surf_point_locations_perspective

import argparse

np.random.seed(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--result-dir', type=str, default='./run/test_on_synthetic_erroneous/080')
parser.add_argument('-v1', '--first-view', type=int, default=2)
parser.add_argument('-v2', '--second-view', type=int, default=3)
parser.add_argument('--wo-rm', action='store_true')
args = parser.parse_args()

#result_dir='./run/test_on_synthetic_erroneous/080'
result_dir=args.result_dir

wo_rm = args.wo_rm

out_dir = result_dir+'/corrs_wo_gbr'
if wo_rm:
    out_dir+='_wo_rm'

idx1 = args.first_view
idx2 = args.second_view
if idx1 > idx2:
    idx1 = args.second_view
    idx2 = args.first_view

th_rm_deg = 10
th_nm_deg = 10
th_pix = 5
th_rm_num = 1000
th_nm_num = 100
optimize_gbr = True
rm_corrs = np.load(out_dir+'/corrs_'+str(idx1)+'_'+str(idx2)+'_rm.npz')
nm_corrs = np.load(out_dir+'/corrs_'+str(idx1)+'_'+str(idx2)+'_nm.npz')
normals = torch.load(result_dir+'/est_normal_sfs_final.pt')

normal1 = normals[idx1:idx1+1].to(device)
normal2 = normals[idx2:idx2+1].to(device)

normal1_ldr = (0.5*(normal1[0] + 1)).detach().cpu().numpy().transpose((1,2,0))
normal2_ldr = (0.5*(normal2[0] + 1)).detach().cpu().numpy().transpose((1,2,0))

normal1_ldr = (255. * np.clip(normal1_ldr,0.,1.)).astype(np.uint8)
normal2_ldr = (255. * np.clip(normal2_ldr,0.,1.)).astype(np.uint8)

rmap = cv2.imread(result_dir+'/est_rmap_final.exr',-1)

reso = rmap.shape[0]
rmap1 = rmap[:,reso*idx1:reso*(idx1+1),::-1].copy()
rmap2 = rmap[:,reso*idx2:reso*(idx2+1),::-1].copy()

rmap1 = torch.stack(torch.from_numpy(rmap1).unbind(-1),0)[None].float().to(device)
rmap2 = torch.stack(torch.from_numpy(rmap2).unbind(-1),0)[None].float().to(device)

nf = torch.quantile(torch.cat([rmap1,rmap2],dim=-1)[torch.cat([rmap1,rmap2],dim=-1) > 0], 0.9)
rmap1_ldr = (255 * np.clip(torch.stack(torch.unbind(rmap1[0], 0), dim=-1).cpu().numpy(),0,1)**(1/2.2)).astype(np.uint8)
rmap2_ldr = (255 * np.clip(torch.stack(torch.unbind(rmap2[0], 0), dim=-1).cpu().numpy(),0,1)**(1/2.2)).astype(np.uint8)

gt_extrinsics = torch.load(result_dir+'/gt_extrinsic_matrices.pt').numpy()
gt_extrinsics_1 = gt_extrinsics[idx1]
gt_extrinsics_2 = gt_extrinsics[idx2]
gt_extrinsics_21 = gt_extrinsics_1 @ np.linalg.inv(gt_extrinsics_2)
gt_rot_1 = gt_extrinsics[idx1,:3,:3]
gt_rot_2 = gt_extrinsics[idx2,:3,:3]
gt_rot_21 = gt_rot_1 @ np.linalg.inv(gt_rot_2)
gt_tran_21 = gt_extrinsics_21[:3,3]
print('GT Realitive Pose:')
print(gt_rot_21)
print('GT Euler Angles:')
print(matrix_to_euler(gt_rot_21))


normal1_rm = rm_corrs['normal1']
normal2_rm = rm_corrs['normal2']
rm_coord1 = rm_corrs['rm_coord1']
rm_coord2 = rm_corrs['rm_coord2']
if wo_rm:
    normal1_rm = normal1_rm[:0]
    normal2_rm = normal2_rm[:0]
    rm_coord1 = rm_coord1[:0]
    rm_coord2 = rm_coord2[:0]
normal1_nm = nm_corrs['normal1']
normal2_nm = nm_corrs['normal2']
pixel1 = nm_corrs['coord1']
pixel2 = nm_corrs['coord2']

def normal_to_light(normal):
    n2 = np.sum(normal**2, axis=-1) + 1e-9
    return np.stack([
        -2. * normal[:,0] * normal[:,2] / n2,
        -2. * normal[:,1] * normal[:,2] / n2,
        -2. * normal[:,2]**2 / n2 + 1.,
    ], axis=-1)

# compute R that satisfies v1.T = R@v2.T
def compute_relative_rot(v1, v2):
    X = v1.T
    Y = v2.T
    U,s,Vt = np.linalg.svd(Y@(X.T))
    H = np.diag([1.0, 1.0, np.linalg.det(Vt.T@U.T)])
    R21 = Vt.T@H@U.T
    return R21

normal1 = np.concatenate([normal1_nm, normal_to_light(normal1_rm)], axis=0)
normal2 = np.concatenate([normal2_nm, normal_to_light(normal2_rm)], axis=0)


R21_est_naive = compute_relative_rot(
    normal1,
    normal2
)


def print_pose_error(R_est, R_gt):
    print('Pose Error:', np.degrees(compute_pose_error(R_est, R_gt)))


print('Pose estimated by naive Kabsch algorithm')
print(R21_est_naive)
print_pose_error(R21_est_naive, gt_rot_21)

# RANSAC
best_score = 0

bar = tqdm(range(10000))
for _ in bar:
    # random sampling of n matches from all detected matches
    sampled_idx_normal = np.random.choice(len(normal1_nm), 2)
    normal1_sampled = normal1[sampled_idx_normal]
    normal2_sampled = normal2[sampled_idx_normal]

    R21_est = compute_relative_rot(
        normal1_sampled,
        normal2_sampled
    )
    phi_est, eta_est, theta_est = matrix_to_euler(R21_est)

    normal2_nm_warped = normal2_nm @ R21_est.T
    normal2_rm_warped = normal_to_light(normal2_rm) @ R21_est.T

    err_nm = np.arccos(np.clip(np.sum(normal1_nm * normal2_nm_warped, axis=-1),-1,1))
    err_rm = np.arccos(np.clip(np.sum(normal_to_light(normal1_rm) * normal2_rm_warped, axis=-1),-1,1))
    err_pix = res_func_pixel((phi_est, theta_est), pixel1, pixel2)

    idx_in_nm = (np.degrees(err_nm) < th_nm_deg)
    idx_in_rm = (np.degrees(err_rm) < th_rm_deg)

    if len(normal1_nm) == len(pixel1):
        idx_in_nm *= (np.abs(err_pix) < th_pix)

    if (np.sum(idx_in_nm) < th_nm_num) or ((not wo_rm) and ((np.sum(idx_in_rm) < th_rm_num))):
        continue

    score_nm = np.sum(idx_in_nm) / (len(idx_in_nm) + 1e-7)
    score_rm = np.sum(idx_in_rm) / (len(idx_in_rm) + 1e-7)

    score = score_nm + score_rm
    if score > best_score:
        best_score = score
        best_idx_in_nm = idx_in_nm
        best_idx_in_rm = idx_in_rm

        print('Updated')
        print_pose_error(R21_est, gt_rot_21)
        print('')

    bar.set_postfix(
        best_score=best_score,
        num_in_nm=np.sum(best_idx_in_nm),
    )

    #print(np.sum(idx_in_rm), len(idx_in_rm))
    #print(np.sum(idx_in_nm), len(idx_in_nm))

# refinement with inliers
R21_est = compute_relative_rot(
    np.concatenate([normal1_nm[best_idx_in_nm], normal_to_light(normal1_rm[best_idx_in_rm])]),
    np.concatenate([normal2_nm[best_idx_in_nm], normal_to_light(normal2_rm[best_idx_in_rm])])
)

G1_est = np.eye(3)
G2_est = np.eye(3)
G21_tilde_est = R21_est
l1_est = 1.
m1_est = 0.
n1_est = 0.
l2_est = 1.
m2_est = 0.
n2_est = 0.
phi_est, eta_est, theta_est = matrix_to_euler(R21_est)

# translation estimation
if os.path.exists(result_dir+'/gt_intrinsic_matrices.pt'):
    intrinsics = torch.load(result_dir+'/gt_intrinsic_matrices.pt').numpy()
    K1 = intrinsics[idx1]
    K2 = intrinsics[idx2]
else:
    K = np.array([
        [4000, 0, 128],
        [0, 4000, 128],
        [0, 0, 1]
    ])
    K1 = K
    K2 = K



t21_est = solve_translation_perspective(pixel1[best_idx_in_nm], pixel2[best_idx_in_nm], K1, K2, R21_est)
surf_points_est = solve_surf_point_locations_perspective(pixel1[best_idx_in_nm], pixel2[best_idx_in_nm], K1, K2, R21_est, t21_est)
if np.sum(surf_points_est[:,2] < 0) > (0.5 * len(surf_points_est)):
    t21_est *= -1
    surf_points_est *= -1

print('GT Realitive Translation:')
print(gt_tran_21 / np.linalg.norm(gt_tran_21))
print('Estimated Relative Translation')
print(t21_est)

bbox_min = np.amin(surf_points_est, axis=0)
bbox_max = np.amax(surf_points_est, axis=0)
bbox_center = 0.5 * (bbox_min + bbox_max)
bbox_diagonal = np.linalg.norm(bbox_max - bbox_min)

print('bbox center:', bbox_center)
print('bbox diagonal:', bbox_diagonal)    

# recover and adjust absolute translations according to recovered surface points
s = 1 / bbox_diagonal
t1_est = s * bbox_center
t21_est = s * t21_est
t2_est = (R21_est.T @ (t1_est - t21_est)[:,None])[:,0]
print('t1:', t1_est)
print('t2:', t2_est)
print('t21:', t21_est)

print('t1_gt:', gt_extrinsics_1[:3,3])
print('t2_gt:', gt_extrinsics_2[:3,3])

R1_est = np.diag([1., -1., -1.,])
R2_est = R21_est.T @ R1_est
P1_est = np.concatenate([K1 @ np.concatenate([R1_est, t1_est[:,None]], axis=1), np.array([[0.,0.,0.,1.]])], axis=0)
P2_est = np.concatenate([K2 @ np.concatenate([R2_est, t2_est[:,None]], axis=1), np.array([[0.,0.,0.,1.]])], axis=0)


# 
print('Pose estimated by RANSAC')
print(R21_est)
print('Euler Angles')
print(matrix_to_euler(R21_est))
print_pose_error(R21_est, gt_rot_21)

print('Number of Normal Corrs:', len(normal1_nm))
print('Number of Estimated Inlier Normal Corrs:', len(normal1_nm[best_idx_in_nm]))
print('Number of RM Corrs:', len(normal1_rm))
print('Number of Estimated Inlier RM Corrs:', len(normal1_rm[best_idx_in_rm]))

np.savez(
    out_dir+'/corrs_in_'+str(idx1)+'_'+str(idx2)+'_nm.npz',
    normal1=normal1_nm[best_idx_in_nm],
    normal2=normal2_nm[best_idx_in_nm],
    coord1=nm_corrs['coord1'][best_idx_in_nm],
    coord2=nm_corrs['coord2'][best_idx_in_nm],
    G21_tilde=G21_tilde_est,
    phi=phi_est,
    eta=eta_est,
    theta=theta_est,
)

np.savez(
    out_dir+'/corrs_in_'+str(idx1)+'_'+str(idx2)+'_rm.npz',
    normal1=normal1_rm[best_idx_in_rm],
    normal2=normal2_rm[best_idx_in_rm],
    rm_coord1=rm_coord1[best_idx_in_rm],
    rm_coord2=rm_coord2[best_idx_in_rm],
)

np.savez(
    out_dir+'/est_pose_'+str(idx1)+'_'+str(idx2)+'.npz',
    R21=R21_est,
    t21=t21_est,
    R1=R1_est,
    R2=R2_est,
    P1=P1_est,
    P2=P2_est,
    K1=K1,
    K2=K2,
    t1=t1_est,
    t2=t2_est,
    l1=l1_est,
    m1=m1_est,
    n1=n1_est,
    l2=l2_est,
    m2=m2_est,
    n2=n2_est,
)

normals = torch.load(result_dir+'/est_normal_sfs_final.pt')
if os.path.exists(result_dir+'/est_normal_nlsfs_final.pt'):
    normals = torch.load(result_dir+'/est_normal_nlsfs_final.pt')
torch.save(normals, out_dir+'/est_normal_sfs_final_undistorted.pt')
save_normal_map(out_dir+'/est_normal_sfs_final_undistorted.png', torch.cat(normals.unbind(0), dim=-1))

rmap_undistorted = torch.cat([rmap1, rmap2], dim=0).detach().cpu()

torch.save(rmap_undistorted, out_dir+'/est_rmap_final_undistorted.pt')
save_hdr_as_ldr(out_dir+'/est_rmap_final_undistorted.png', torch.cat(rmap_undistorted.unbind(0), dim=-1)[None])
save_hdr(out_dir+'/est_rmap_final_undistorted.exr', torch.cat(rmap_undistorted.unbind(0), dim=-1)[None])



if True:
    import open3d as o3d
    surf_points_est = solve_surf_point_locations_perspective(pixel1, pixel2, K1, K2, R21_est, t21_est)
    surf_points_global_est = (np.linalg.inv(R1_est) @ (surf_points_est - t1_est).T).T
    pcd_est = o3d.geometry.PointCloud()
    pcd_est.points = o3d.utility.Vector3dVector(surf_points_global_est)

    o3d.io.write_point_cloud(out_dir+'/surf_points.ply', pcd_est)

# draw inlier matches
kp1 = [cv2.KeyPoint(u,v, 1.) for u,v in nm_corrs['coord1'][best_idx_in_nm].astype(np.float32)]
kp2 = [cv2.KeyPoint(u,v, 1.) for u,v in nm_corrs['coord2'][best_idx_in_nm].astype(np.float32)]
matches = [cv2.DMatch(i, i, 0) for i in range(len(kp1))]

skip = max(1, len(matches) // 50)
cv2.setRNGSeed(6)
img_inliers = cv2.drawMatches(
    normal1_ldr, 
    kp1, 
    normal2_ldr, 
    kp2, 
    matches[::skip], 
    None, 
    flags=2
)
cv2.imwrite(out_dir+'/matches_nm_in_'+str(idx1)+'_'+str(idx2)+'.png', img_inliers[:,:,::-1])
#plt.imshow(img_inliers)
#plt.ylabel('Est Matches')
#plt.show()

result_text = ''
result_text += 'R21_error_deg: '+str(np.degrees(compute_pose_error(R21_est, gt_rot_21)))
with open(out_dir+'/accuracy_'+str(idx1)+'_'+str(idx2)+'.txt', 'w') as f:
    f.write(result_text)

kp1 = [cv2.KeyPoint(u,v, 1.) for u,v in rm_coord1[best_idx_in_rm].astype(np.float32)]
kp2 = [cv2.KeyPoint(u,v, 1.) for u,v in rm_coord1[best_idx_in_rm].astype(np.float32)]
matches = [cv2.DMatch(i, i, 0) for i in range(len(kp1))]
skip = max(1, len(matches) // 50)

cv2.setRNGSeed(6)
result = cv2.drawMatches(
    rmap1_ldr, 
    kp1, 
    rmap2_ldr, 
    kp2, 
    matches[::skip], 
    None, 
    flags=2
)
cv2.imwrite(out_dir+'/matches_rm_in_'+str(idx1)+'_'+str(idx2)+'.png', result[:,:,::-1])
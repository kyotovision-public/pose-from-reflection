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
from core.rmap_utils import create_normal_grid
from tqdm import tqdm

import scipy

from core.sfs_utils import plot_hdr, save_normal_map
from core.sfm_utils import solve_pose, res_func_pixel, matrix_to_euler, compute_relative_rot, decode_param, compute_pose_error, get_gbr_matrix,solve_phi_theta, decompose_invalid_relative_gbr, euler_to_matrix, encode_param
from core.sfm_utils import res_func, decompose_relative_gbr
import argparse

np.random.seed(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--result-dir', type=str, default='./run/test_on_synthetic_erroneous/080')
parser.add_argument('-v1', '--first-view', type=int, default=0)
parser.add_argument('-v2', '--second-view', type=int, default=1)
parser.add_argument('--wo-data-aug', action='store_true')
parser.add_argument('--wo-gbr', action='store_true')
parser.add_argument('--sym', action='store_true')
parser.add_argument('--prefiltering', action='store_true')
parser.add_argument('--normal-reg', action='store_true')
args = parser.parse_args()

#result_dir='./run/test_on_synthetic_erroneous/080'
result_dir=args.result_dir
wo_data_aug = args.wo_data_aug
wo_gbr = args.wo_gbr
prefiltering = args.prefiltering
use_nreg = args.normal_reg

out_dir = result_dir+'/corrs'
if wo_data_aug:
    out_dir+='_wo_da'
if wo_gbr:
    out_dir+='_wo_gbr'
if prefiltering:
    out_dir+='_prefiltering'
if use_nreg:
    out_dir+='_nreg'
if True:
    out_dir+='_sym'
os.makedirs(out_dir, exist_ok=True)

idx1 = args.first_view
idx2 = args.second_view
if idx1 > idx2:
    idx1 = args.second_view
    idx2 = args.first_view

th_rm_deg = 20
th_nm_deg = 20
th_reg_deg = 20
th_pix = 5
th_rm_num = 1000
th_nm_num = 100
score_exp_rm = 0.0
score_exp_reg = 0.1
optimize_gbr = True
nm_corrs = np.load(out_dir+'/corrs_'+str(idx1)+'_'+str(idx2)+'_nm.npz')
rm_corrs = np.load(out_dir+'/corrs_'+str(idx1)+'_'+str(idx2)+'_rm.npz')
normals = torch.load(result_dir+'/est_normal_sfs_final.pt')
if os.path.exists(result_dir+'/est_normal_nlsfs_final.pt'):
    normals = torch.load(result_dir+'/est_normal_nlsfs_final.pt')

normal1 = normals[idx1:idx1+1].to(device)
normal2 = normals[idx2:idx2+1].to(device)

normal1_ldr = (0.5*(normal1[0] + 1)).detach().cpu().numpy().transpose((1,2,0))
normal2_ldr = (0.5*(normal2[0] + 1)).detach().cpu().numpy().transpose((1,2,0))

normal1_ldr = (255. * np.clip(normal1_ldr,0.,1.)).astype(np.uint8)
normal2_ldr = (255. * np.clip(normal2_ldr,0.,1.)).astype(np.uint8)

gt_extrinsics = torch.load(result_dir+'/gt_extrinsic_matrices.pt').numpy()
gt_rot_1 = gt_extrinsics[idx1,:3,:3]
gt_rot_2 = gt_extrinsics[idx2,:3,:3]
gt_rot_21 = gt_rot_1 @ np.linalg.inv(gt_rot_2)
gt_rot_12 = gt_rot_2 @ np.linalg.inv(gt_rot_1)
print('GT Realitive Pose:')
print(gt_rot_21)
print('GT Euler Angles:')
print(matrix_to_euler(gt_rot_21))

if os.path.exists(result_dir+'/gt_gbr_params.pt'):
    gt_gbr_params = torch.load(result_dir+'/gt_gbr_params.pt').numpy()
    l1_gt, m1_gt, n1_gt = gt_gbr_params[idx1] * np.array([1., -1., 1.])
    l2_gt, m2_gt, n2_gt = gt_gbr_params[idx2] * np.array([1., -1., 1.])
    G1_gt = get_gbr_matrix(l1_gt, m1_gt, n1_gt)
    G2_gt = get_gbr_matrix(l2_gt, m2_gt, n2_gt)
    G21_tilde_gt =  np.linalg.inv(G1_gt).T @ gt_rot_21 @ G2_gt.T
    print('GT Relative GBR:')
    print(G21_tilde_gt)


#normal1_rm = rm_corrs['normal1']
#normal2_rm = rm_corrs['normal2']
normal1_nm = nm_corrs['normal1']
normal2_nm = nm_corrs['normal2']
pixel1 = nm_corrs['coord1']
pixel2 = nm_corrs['coord2']

if prefiltering:
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

    pixel1 = pixel1[best_idx_in]
    pixel2 = pixel2[best_idx_in]
    normal1_nm = normal1_nm[best_idx_in]
    normal2_nm = normal2_nm[best_idx_in]

    if True:
        # draw inlier matches
        kp1 = [cv2.KeyPoint(u,v, 1.) for u,v in pixel1.astype(np.float32)]
        kp2 = [cv2.KeyPoint(u,v, 1.) for u,v in pixel2.astype(np.float32)]
        matches = [cv2.DMatch(i, i, 0) for i in range(len(kp1))]

        skip = max(1, len(matches) // 100)
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
        cv2.imwrite(out_dir+'/matches_pix_in_ortho_'+str(idx1)+'_'+str(idx2)+'.png', img_inliers[:,:,::-1])


    print('score (ortho):',best_score)
    print('(phi,theta):',np.degrees(best_param))
    print('(phi_gt,theta_gt):',np.degrees(matrix_to_euler(gt_rot_12.T)))
    if True:
        plt.imshow(score_map)
        plt.xlabel('phi')
        plt.ylabel('theta')
        plt.savefig(out_dir+'/pix_score_map.png')
        plt.close()

if True:
    normal_grid = create_normal_grid(128, 'probe').numpy() * np.array([1., -1., -1.])
    density_grid_1 = np.zeros_like(normal_grid[:,:,0])
    density_grid_2 = np.zeros_like(normal_grid[:,:,0])
    normal1_nm_density = np.zeros_like(normal1_nm[:,0])
    normal2_nm_density = np.zeros_like(normal2_nm[:,0])
    bar = tqdm(range(len(normal1_nm[:])))
    bar.set_description('Corrs Density Estimation (NM)')
    for idx_normal in bar:
        density_grid_1 += np.exp(-100 * (1. - np.sum(normal_grid * normal1_nm[idx_normal], axis=-1)))
        normal1_nm_density += np.exp(-100 * (1. - np.sum(normal1_nm * normal1_nm[idx_normal], axis=-1)))
        density_grid_2 += np.exp(-100 * (1. - np.sum(normal_grid * normal2_nm[idx_normal], axis=-1)))
        normal2_nm_density += np.exp(-100 * (1. - np.sum(normal2_nm * normal2_nm[idx_normal], axis=-1)))

    sample_importance_nm = 1. / ((normal1_nm_density * normal2_nm_density)**(0.5))
    sample_importance_nm /= np.mean(sample_importance_nm)

    if True:
        #plt.semilogy(sorted(sample_importance_nm))
        #plt.show()

        plt.subplot(2,3,1)
        plt.scatter(normal1_nm[:,0], normal1_nm[:,1])
        plt.subplot(2,3,2)
        plt.imshow(0.5 * (normal_grid * np.array([1., -1., -1.]) + 1.0))
        plt.subplot(2,3,3)
        plt.imshow(density_grid_1)
        plt.subplot(2,3,4)
        plt.scatter(normal2_nm[:,0], normal2_nm[:,1])
        plt.subplot(2,3,5)
        plt.imshow(0.5 * (normal_grid * np.array([1., -1., -1.]) + 1.0))
        plt.subplot(2,3,6)
        plt.imshow(density_grid_2)
        plt.savefig(out_dir+'/density_nm.png')
        plt.close()

#normal2_nm = distort_normal(undistort_normal(normal1_nm, 0.6,0.01,-0.1) @ gt_rot_21, 0.5, 0.01, 0.1)
#normal2_nm = normal1_nm @ gt_rot_21

normal1_rm = rm_corrs['normal1']
normal2_rm = rm_corrs['normal2']
rmap1_fea = torch.from_numpy(rm_corrs['fea1'])
rmap2_fea = torch.from_numpy(rm_corrs['fea2'])

R21_est_naive = compute_relative_rot(
    normal1_nm,
    normal2_nm
)


def print_pose_error(R_est, R_gt):
    print('Pose Error:', np.degrees(compute_pose_error(R_est, R_gt)))


print('Pose estimated by naive Kabsch algorithm')
print(R21_est_naive)
print_pose_error(R21_est_naive, gt_rot_21)


result_wo_ransac = solve_pose({
    'pixel1': pixel1,
    'pixel2': pixel2,
    'normal1_nm': normal1_nm,
    'normal2_nm': normal2_nm,
    'normal1_rm': normal1_nm[:0], # normal1_rm,
    'normal2_rm': normal2_nm[:0], # normal2_rm,
}, opencv_coord=True)


R21_est_wo_ransac = R21_est = result_wo_ransac['R21_est']#decode_param(result_wo_ransac.x)[-1]
print('Pose estimated without RANSAC')
print(R21_est_wo_ransac)
print_pose_error(R21_est_wo_ransac, gt_rot_21)
#print('GBR parames estimated without RANSAC')
#print(decode_param(result_wo_ransac.x)[:-1])

if False:
    normal_grid = create_normal_grid(1024, 'probe').numpy()
    normal_grid_mask = (normal_grid[...,2] > 1e-6).astype(np.float32)
    #normal_grid_mask = (np.sum(normal_grid**2,axis=-1) > 1e-1).astype(np.float32)
    normal_grid *= normal_grid_mask[...,None]
    normal_grid_sa = np.sinc(np.arccos(np.clip(normal_grid[...,2], -1, 1)) / np.pi) * normal_grid_mask
    normal_grid_sa *= np.pi**2 / (normal_grid.shape[0] // 2)**2

    if False:
        light_grid = create_normal_grid(512, 'probe').numpy()
        light_grid_mask = (np.sum(light_grid**2,axis=-1) > 1e-1).astype(np.float32)
        light_grid_sa = np.sinc(np.arccos(np.clip(light_grid[...,2], -1, 1)) / np.pi) * light_grid_mask
        light_grid_sa *= np.pi**2 / (light_grid.shape[0] // 2)**2

        print(np.sum(normal_grid_sa) / np.pi, np.sum(light_grid_sa) / np.pi)

        plt.subplot(1,3,1)
        plt.imshow(normal_grid_sa[256:768,256:768])
        plt.subplot(1,3,2)
        plt.imshow(light_grid_sa)
        plt.subplot(1,3,3)
        plt.imshow(light_grid_sa / np.clip(normal_grid_sa[256:768,256:768], 1e-20, None))
        plt.show()

    G = get_gbr_matrix(0.1, 0.2, 0.2)

    normal_grid_e1 = np.cross(normal_grid, np.stack([-normal_grid[...,1], normal_grid[...,0], 0 * normal_grid[...,0]], axis=-1))
    normal_grid_e1 /= np.clip(np.linalg.norm(normal_grid_e1, axis=-1, keepdims=True),1e-20,None)
    normal_grid_e2 = np.cross(normal_grid_e1, normal_grid)
    normal_grid_e2 /= np.clip(np.linalg.norm(normal_grid_e2, axis=-1, keepdims=True),1e-20,None)

    normal_grid_undistorted = np.sum(normal_grid[...,:,None] * G, axis=-2)
    normal_grid_undistorted_norm = np.linalg.norm(normal_grid_undistorted, axis=-1)
    normal_grid_undistorted /= np.clip(normal_grid_undistorted_norm[...,None], 1e-20, None)
    normal_grid_e1_undistorted = np.sum(normal_grid_e1[...,:,None] * G, axis=-2) / np.clip(normal_grid_undistorted_norm[...,None], 1e-20, None)
    normal_grid_e1_undistorted -= np.sum(normal_grid_e1_undistorted * normal_grid_undistorted, axis=-1, keepdims=True) * normal_grid_undistorted
    normal_grid_e2_undistorted = np.sum(normal_grid_e2[...,:,None] * G, axis=-2) / np.clip(normal_grid_undistorted_norm[...,None], 1e-20, None)
    normal_grid_e2_undistorted -= np.sum(normal_grid_e2_undistorted * normal_grid_undistorted, axis=-1, keepdims=True) * normal_grid_undistorted

    normal_grid_sa_ratio = np.linalg.norm(np.cross(normal_grid_e1_undistorted, normal_grid_e2_undistorted), axis=-1)
    #normal_grid_sa_ratio /= np.clip(normal_grid_undistorted_norm, 1e-20, None)**2
    normal_grid_sa_ratio *= 4 * np.clip(normal_grid_undistorted[...,2], 0, 1)

    print(np.sum(normal_grid_sa) / (4 * np.pi))
    print(np.sum(normal_grid_sa * normal_grid_sa_ratio) / (4 * np.pi))
    from core.sfs_utils import plot_normal_map
    for i in range(3):
        plt.subplot(4,5,1+i)
        plt.imshow(normal_grid[...,i])

        plt.subplot(4,5,6+i)
        plt.imshow(normal_grid_e1[...,i])
        plt.subplot(4,5,11+i)
        plt.imshow(normal_grid_e2[...,i])
    plt.subplot(4,5,4)
    plt.imshow(normal_grid_mask)
    plt.subplot(4,5,5)
    plt.imshow(normal_grid_sa)

    plt.subplot(4,5,9)
    plt.imshow(np.sum(normal_grid_e1 * normal_grid, axis=-1), vmin=-1,vmax=1)
    plt.subplot(4,5,14)
    plt.imshow(np.sum(normal_grid_e2 * normal_grid, axis=-1), vmin=-1,vmax=1)
    plt.subplot(4,5,15)
    plt.imshow(np.sum(normal_grid_e2 * normal_grid_e1, axis=-1), vmin=-1,vmax=1)

    plt.subplot(4,5,16)
    plt.imshow(normal_grid_sa_ratio)
    plt.subplot(4,5,17)
    plt.imshow(normal_grid_sa * normal_grid_sa_ratio)
    plt.subplot(4,5,18)
    plt.imshow(0.5 * (normal_grid + 1))
    plt.subplot(4,5,19)
    plt.imshow(0.5 * (normal_grid_undistorted + 1))
    plt.subplot(4,5,20)
    plt.imshow(normal_grid_undistorted_norm)

    plt.show()


def eval_param_nm(param, idx_in_nm=None):
    if idx_in_nm is None:
        idx_in_nm = np.arange(len(normal1_nm))
    l1_est, m1_est, n1_est, l2_est, m2_est, n2_est,R21_est = decode_param(param)
    phi_est, eta_est, theta_est = matrix_to_euler(R21_est)

    G1_est = get_gbr_matrix(l1_est, m1_est, n1_est)
    G2_est = get_gbr_matrix(l2_est, m2_est, n2_est)
    G21_tilde_est = np.linalg.inv(G1_est).T @ R21_est @ G2_est.T

    normal1_nm_warped = normal1_nm[idx_in_nm] @ np.linalg.inv(G21_tilde_est).T
    normal2_nm_warped = normal2_nm[idx_in_nm] @ G21_tilde_est.T

    normal1_nm_warped /= (np.linalg.norm(normal1_nm_warped, axis=1, keepdims=True) + 1e-2)
    normal2_nm_warped /= (np.linalg.norm(normal2_nm_warped, axis=1, keepdims=True) + 1e-2)
    #normal1_nm_warped, normal2_nm_warped = warp_normal(
    #    result.x, normal1_nm, normal2_nm, normal1_nm, normal2_nm
    #)[2:]
    err_nm_1 = np.arccos(np.clip(np.sum(normal1_nm[idx_in_nm]  * normal2_nm_warped, axis=-1),-1,1))
    err_nm_2 = np.arccos(np.clip(np.sum(normal2_nm[idx_in_nm]  * normal1_nm_warped, axis=-1),-1,1))
    err_pix = res_func_pixel((phi_est, theta_est), pixel1[idx_in_nm] , pixel2[idx_in_nm] )

    score_nm = np.sum(
        np.exp(-(err_pix / th_pix)**2) * 
        np.exp(-(np.degrees(err_nm_1) / th_nm_deg)**2) * 
        np.exp(-(np.degrees(err_nm_2) / th_nm_deg)**2) * 
        sample_importance_nm[idx_in_nm]
    ) / len(normal1_nm[idx_in_nm])
    
    idx_in_nm = (np.degrees(err_nm_1) < th_nm_deg) * (np.degrees(err_nm_2) < th_nm_deg) 
    idx_in_nm = idx_in_nm * (normal1_nm_warped[:,2] < -0.1) * (normal2_nm_warped[:,2] < -0.1)
    if len(normal1_nm) == len(pixel1):
        idx_in_nm *= (np.abs(err_pix) < th_pix)

    return score_nm, idx_in_nm

def eval_param_rm(param, idx_in_rm=None):
    #return 0.0, []
    if idx_in_rm is None:
        idx_in_rm = np.arange(len(normal1_rm))
    l1_est, m1_est, n1_est, l2_est, m2_est, n2_est,R21_est = decode_param(param)

    G1_est = get_gbr_matrix(l1_est, m1_est, n1_est)
    G2_est = get_gbr_matrix(l2_est, m2_est, n2_est)

    # eval goodness using correspondences in RMs
    def light_to_normal(light):
        n = light + np.array([0.,0.,-1.])
        return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-20)
    def normal_to_light(normal):
        n2 = np.sum(normal**2, axis=-1) + 1e-20
        return np.stack([
            -2. * normal[:,0] * normal[:,2] / n2,
            -2. * normal[:,1] * normal[:,2] / n2,
            -2. * normal[:,2]**2 / n2 + 1.,
        ], axis=-1)

    normal1_rm_e1 = np.cross(normal1_rm, np.stack([-normal1_rm[:,1], normal1_rm[:,0], 0 * normal1_rm[:,0]], axis=-1))
    normal1_rm_e1 /= np.clip(np.linalg.norm(normal1_rm_e1, axis=1, keepdims=True), 1e-20, None)
    normal1_rm_e2 = np.cross(normal1_rm_e1, normal1_rm)
    normal1_rm_e2 /= np.clip(np.linalg.norm(normal1_rm_e2, axis=1, keepdims=True), 1e-20, None)

    normal1_rm_undistorted = normal1_rm @ G1_est
    normal1_rm_undistorted_norm = np.linalg.norm(normal1_rm_undistorted,axis=-1,keepdims=True)
    normal1_rm_undistorted /= (normal1_rm_undistorted_norm + 1e-20)
    normal1_rm_e1_undistorted = normal1_rm_e1 @ G1_est / (normal1_rm_undistorted_norm + 1e-20)
    normal1_rm_e1_undistorted -= np.sum(normal1_rm_undistorted * normal1_rm_e1_undistorted, axis=-1, keepdims=True) * normal1_rm_undistorted
    normal1_rm_e2_undistorted = normal1_rm_e2 @ G1_est / (normal1_rm_undistorted_norm + 1e-20)
    normal1_rm_e2_undistorted -= np.sum(normal1_rm_undistorted * normal1_rm_e2_undistorted, axis=-1, keepdims=True) * normal1_rm_undistorted

    normal1_rm_warped = light_to_normal(normal_to_light(normal1_rm @ G1_est) @ R21_est)  @ np.linalg.inv(G2_est)
    normal1_rm_warped /= (np.linalg.norm(normal1_rm_warped, axis=1, keepdims=True) + 1e-2)
    normal2_rm_warped = light_to_normal(normal_to_light(normal2_rm @ G2_est) @ R21_est.T)  @ np.linalg.inv(G1_est)
    normal2_rm_warped /= (np.linalg.norm(normal2_rm_warped, axis=1, keepdims=True) + 1e-2)


    err_rm_1 = np.arccos(np.clip(np.sum(normal1_rm * normal2_rm_warped, axis=-1),-1,1))
    err_rm_2 = np.arccos(np.clip(np.sum(normal2_rm * normal1_rm_warped, axis=-1),-1,1))

    idx_in_rm = (np.degrees(err_rm_1) < th_rm_deg) * (np.degrees(err_rm_2) < th_rm_deg) 

    sa_rm_1 = np.sinc(np.arccos(np.clip(-normal1_rm[:,2], -1, 1)) / np.pi) # solid angle
    #ratio_area = np.linalg.norm(np.cross(normal1_rm_e1_undistorted, normal1_rm_e2_undistorted), axis=-1)
    #ratio_area /= (normal1_rm_undistorted_norm[...,0] + 1e-2)**2
    #ratio_area *= 4 * np.clip(-normal1_rm_undistorted[:,2], 0, 1) # normal to light dir
    #sa_rm_1 *= np.clip(ratio_area,None,100)
    #plt.plot(sorted(sa_rm_1))
    #plt.show()
    #score_rm = np.sum(idx_in_rm * sa_rm_1)
    score_rm = np.sum(
        np.exp(-(np.degrees(err_rm_1) / th_rm_deg)**2) * 
        np.exp(-(np.degrees(err_rm_2) / th_rm_deg)**2) * 
        sa_rm_1
    ) / len(normal1_rm)

    return score_rm, idx_in_rm

def eval_param_reg(param, idx_in_nm=None):
    if idx_in_nm is None:
        idx_in_nm = np.arange(len(normal1_nm))

    l1_est, m1_est, n1_est, l2_est, m2_est, n2_est,R21_est = decode_param(param)

    G1_est = get_gbr_matrix(l1_est, m1_est, n1_est)
    G2_est = get_gbr_matrix(l2_est, m2_est, n2_est)

    if not use_nreg:
        score_reg = np.exp(
            -1. * (
                np.linalg.norm(G1_est - np.eye(3),ord='fro')**2 + 
                np.linalg.norm(G2_est - np.eye(3),ord='fro')**2
            )
        )
    else:
        normal1_nm_undistorted = normal1_nm[idx_in_nm] @ G1_est
        normal1_nm_undistorted_norm = np.linalg.norm(normal1_nm_undistorted,axis=-1,keepdims=True)
        normal1_nm_undistorted /= (normal1_nm_undistorted_norm + 1e-20)

        normal2_nm_undistorted = normal2_nm[idx_in_nm] @ G2_est
        normal2_nm_undistorted_norm = np.linalg.norm(normal2_nm_undistorted,axis=-1,keepdims=True)
        normal2_nm_undistorted /= (normal2_nm_undistorted_norm + 1e-20)

        err_reg_1 = np.arccos(np.clip(np.sum(normal1_nm * normal1_nm_undistorted, axis=-1),-1,1))
        err_reg_2 = np.arccos(np.clip(np.sum(normal2_nm * normal2_nm_undistorted, axis=-1),-1,1))

        score_reg = np.sum(
            np.exp(-(np.degrees(err_reg_1) / th_reg_deg)**2) * 
            np.exp(-(np.degrees(err_reg_2) / th_reg_deg)**2) * 
            sample_importance_nm[idx_in_nm]
        ) / len(normal1_nm[idx_in_nm])

    return score_reg

def encode_param_sym(R21_est, eta_est_):
    rvec = cv2.Rodrigues(R21_est)[0][:,0]
    return eta_est_, rvec[0], rvec[1], rvec[2]

def decode_param_sym(param):
    eta_ = param[0]
    R21 = cv2.Rodrigues(param[1:4])[0]
    phi, eta, theta = matrix_to_euler(R21)
    l2 = np.sin(eta_) / np.sin(eta)
    m2 = (np.cos(eta_) / np.cos(eta) - 1.) / np.tan(eta) * np.sin(theta)
    n2 = -(np.cos(eta_) / np.cos(eta) - 1.) / np.tan(eta) * np.cos(theta)

    l1 = l2
    m1 = -(np.cos(eta_) / np.cos(eta) - 1.) / np.tan(eta) * np.sin(phi)
    n1 = (np.cos(eta_) / np.cos(eta) - 1.) / np.tan(eta) * np.cos(phi)

    return l1, m1, n1, l2, m2, n2, R21

# RANSAC
best_score = -np.inf

bar = tqdm(range(200))
for _ in bar:
    # random sampling of n matches from all detected matches
    sampled_idx_nm = np.random.choice(len(normal1_nm), 5)
    pixel1_sampled = pixel1[sampled_idx_nm]
    pixel2_sampled = pixel2[sampled_idx_nm]
    normal1_nm_sampled = normal1_nm[sampled_idx_nm]
    normal2_nm_sampled = normal2_nm[sampled_idx_nm]

    possible_phi_theta = solve_phi_theta(pixel1_sampled, pixel2_sampled)

    for phi_est, theta_est in possible_phi_theta:
        for eta_est in [-0.25 * np.pi, 0.25 * np.pi]:
            R21_est = euler_to_matrix(phi_est, eta_est, theta_est)
            eta_est_ = matrix_to_euler(R21_est)[1]

            param_init_ = encode_param_sym(R21_est, eta_est_)

            def res_func_sym(param):
                l1, m1, n1, l2, m2, n2, R21 = decode_param_sym(param)

                return res_func(
                    encode_param(l1, m1, n1, l2, m2, n2, R21),
                    pixel1_sampled, pixel2_sampled,
                    normal1_nm_sampled[:0], normal2_nm_sampled[:0],
                    normal1_nm_sampled, normal2_nm_sampled,
                )
            
            #print(decode_param_sym(param_init_))
            
            result = scipy.optimize.least_squares(
                res_func_sym, 
                param_init_,
                #xtol=1e-15,
                #gtol=1e-15,
                #ftol=1e-15,
                max_nfev=100,
                bounds=(
                    [np.sign(eta_est_) * np.radians(5), -np.inf, -np.inf, -np.inf],
                    [np.sign(eta_est_) * np.radians(85), np.inf, np.inf, np.inf]
                )
            )
            l1_est, m1_est, n1_est, l2_est, m2_est, n2_est,R21_est = decode_param_sym(result.x)

            G1_est = get_gbr_matrix(l1_est, m1_est, n1_est)
            G2_est = get_gbr_matrix(l2_est, m2_est, n2_est)
            G21_tilde_est = np.linalg.inv(G1_est).T @ R21_est @ G2_est.T
            phi_est, eta_est, theta_est = matrix_to_euler(R21_est)

            if (G21_tilde_est[2,0]**2 + G21_tilde_est[2,1]**2) < (0.1**2):
                continue

            if True:
                R21_est = euler_to_matrix(phi_est, np.sign(eta_est) * 0.25 * np.pi, theta_est)
                l1_est, m1_est, n1_est, l2_est, m2_est, n2_est = decompose_relative_gbr(
                    G21_tilde_est,
                    R21_est
                    
                )

                G1_est = get_gbr_matrix(l1_est, m1_est, n1_est)
                G2_est = get_gbr_matrix(l2_est, m2_est, n2_est)
                G21_tilde_est = np.linalg.inv(G1_est).T @ R21_est @ G2_est.T
                phi_est, eta_est, theta_est = matrix_to_euler(R21_est)

            # eval goodness of samples
            if np.linalg.det(G21_tilde_est) < 1e-4:
                continue

            score_nm, idx_in_nm = eval_param_nm(encode_param(l1_est, m1_est, n1_est, l2_est, m2_est, n2_est,R21_est))


            if (np.sum(idx_in_nm) < th_nm_num):
                continue


            if True:
                best_param_ = []
                list_eta = []
                list_score = []
                list_score_rm = []
                list_score_reg = []
                phi, _, theta = matrix_to_euler(R21_est)
                G21_tilde = np.linalg.inv(G1_est).T @ R21_est @ G2_est.T
                if (G21_tilde[2,0]**2 + G21_tilde[2,1]**2) < 1e-3:
                    continue
                # check rmap consistency
                for eta in np.arange(8.5 * np.pi / 180, 352. * np.pi / 180, 1 * np.pi / 180):
                    if np.cos(eta) < np.cos(np.radians(75)):
                        continue
                    R21_est_a = euler_to_matrix(phi, eta, theta)
                    if wo_gbr:
                        R21_est_a = nm_corrs['R21']
                    l1_est_a, m1_est_a, n1_est_a, l2_est_a, m2_est_a, n2_est_a = decompose_relative_gbr(G21_tilde, R21_est_a)

                    G1_est_a = get_gbr_matrix(l1_est_a, m1_est_a, n1_est_a)
                    G2_est_a = get_gbr_matrix(l2_est_a, m2_est_a, n2_est_a)
                    G21_tilde_init =  np.linalg.inv(G1_est_a).T @ R21_est_a @ G2_est_a.T
                    if np.sum((G21_tilde_init - G21_tilde)**2) > 1e-3:        
                        continue

                    G1_est_a = get_gbr_matrix(l1_est_a, m1_est_a, n1_est_a)
                    G2_est_a = get_gbr_matrix(l2_est_a, m2_est_a, n2_est_a)

                    #score_nm, idx_in_nm = eval_param_nm(encode_param(l1_est_a, m1_est_a, n1_est_a, l2_est_a, m2_est_a, n2_est_a,R21_est_a))
                    score_rm, idx_in_rm = eval_param_rm(encode_param(l1_est_a, m1_est_a, n1_est_a, l2_est_a, m2_est_a, n2_est_a,R21_est_a))

                    if idx_in_rm is None:
                        idx_in_rm = np.arange(len(normal1_rm))

                    score_reg = eval_param_reg(encode_param(l1_est_a, m1_est_a, n1_est_a, l2_est_a, m2_est_a, n2_est_a,R21_est_a))

                    list_score.append(score_rm**score_exp_rm * score_reg**score_exp_reg)
                    list_score_rm.append(score_rm)
                    list_score_reg.append(score_reg)
                    list_eta.append(eta)

                    if list_score[-1] == max(list_score):
                        best_score_rm_ = score_rm
                        best_score_reg_ = score_reg
                        best_param_ = encode_param(l1_est_a, m1_est_a, n1_est_a, l2_est_a, m2_est_a, n2_est_a, R21_est_a)
                        best_idx_in_rm_ = idx_in_rm
                if len(list_score) == 0:
                    continue
                if best_param_ is None:
                    continue
                l1_est, m1_est, n1_est, l2_est, m2_est, n2_est, R21_est = decode_param(best_param_)
                phi_est, eta_est, theta_est = matrix_to_euler(R21_est)
                score_rm = best_score_rm_
                score_reg = best_score_reg_


            score = score_nm * score_rm**score_exp_rm * score_reg**score_exp_reg
            if score > best_score:

                if False:
                    plt.subplot(3,1,1)
                    plt.plot(list_eta, list_score)
                    plt.subplot(3,1,2)
                    plt.plot(list_eta, list_score_rm)
                    plt.subplot(3,1,3)
                    plt.plot(list_eta, list_score_reg)
                    plt.show()

                #best_result_dict = result_dict
                best_param = encode_param(l1_est, m1_est, n1_est, l2_est, m2_est, n2_est,R21_est)#result.x
                best_score = score
                best_idx_in_nm = idx_in_nm
                best_idx_in_rm = best_idx_in_rm_

                print('Updated', score_nm, score_rm, score_reg)
                print_pose_error(R21_est, gt_rot_21)
                print(np.linalg.det(G21_tilde_est))
                print(phi_est, eta_est, theta_est)
                print(l1_est, m1_est, n1_est, l2_est, m2_est, n2_est)
                print('')

                if False:
                    plt.figure(figsize=(7.2,7.2))

                    plt.subplot(2,2,1)
                    plt.scatter(
                        normal1_nm[idx_in_nm][:,0],
                        normal1_nm[idx_in_nm][:,1],
                    )
                    plt.xlim([-1,1])
                    plt.ylim([-1,1])
                    plt.gca().set_aspect('equal')

                    plt.subplot(2,2,2)
                    plt.scatter(
                        normal2_nm[idx_in_nm][:,0],
                        normal2_nm[idx_in_nm][:,1],
                    )
                    plt.xlim([-1,1])
                    plt.ylim([-1,1])
                    plt.gca().set_aspect('equal')

                    plt.subplot(2,2,3)
                    plt.scatter(
                        normal2_nm_warped[idx_in_nm][:,0],
                        normal2_nm_warped[idx_in_nm][:,1],
                    )
                    plt.xlim([-1,1])
                    plt.ylim([-1,1])
                    plt.gca().set_aspect('equal')

                    plt.subplot(2,2,4)
                    plt.scatter(
                        normal1_nm_warped[idx_in_nm][:,0],
                        normal1_nm_warped[idx_in_nm][:,1],
                    )
                    plt.xlim([-1,1])
                    plt.ylim([-1,1])
                    plt.gca().set_aspect('equal')

                    plt.show()

            bar.set_postfix(
                best_score=best_score,
                num_in_nm=np.sum(best_idx_in_nm),
                num_in_rm=np.sum(best_idx_in_rm),
            )

        #print(np.sum(idx_in_rm), len(idx_in_rm))
        #print(np.sum(idx_in_nm), len(idx_in_nm))

if True:
    normal_grid = create_normal_grid(128, 'probe').numpy() * np.array([1., -1., -1.])
    density_grid_1 = np.zeros_like(normal_grid[:,:,0])
    density_grid_2 = np.zeros_like(normal_grid[:,:,0])
    bar = tqdm(range(len(normal1_nm[best_idx_in_nm][:])))
    bar.set_description('Corrs Density Estimation (NM)')
    for idx_normal in bar:
        density_grid_1 += np.exp(-100 * (1. - np.sum(normal_grid * normal1_nm[best_idx_in_nm][idx_normal], axis=-1)))
        density_grid_2 += np.exp(-100 * (1. - np.sum(normal_grid * normal2_nm[best_idx_in_nm][idx_normal], axis=-1)))

    if True:
        #plt.semilogy(sorted(sample_importance_nm))
        #plt.show()

        plt.subplot(2,3,1)
        plt.scatter(normal1_nm[best_idx_in_nm][:,0], normal1_nm[best_idx_in_nm][:,1])
        plt.subplot(2,3,2)
        plt.imshow(0.5 * (normal_grid * np.array([1., -1., -1.]) + 1.0))
        plt.subplot(2,3,3)
        plt.imshow(density_grid_1)
        plt.subplot(2,3,4)
        plt.scatter(normal2_nm[best_idx_in_nm][:,0], normal2_nm[best_idx_in_nm][:,1])
        plt.subplot(2,3,5)
        plt.imshow(0.5 * (normal_grid * np.array([1., -1., -1.]) + 1.0))
        plt.subplot(2,3,6)
        plt.imshow(density_grid_2)
        plt.savefig(out_dir+'/density_nm_in.png')
        plt.close()


def res_func_(param):
    return -eval_param_nm(param)[0] * eval_param_rm(param)[0]**score_exp_rm * eval_param_reg(param)**score_exp_reg

# refinement
if use_nreg:
    l1_est, m1_est, n1_est, l2_est, m2_est, n2_est,R21_est = decode_param(best_param)

    G1_est = get_gbr_matrix(l1_est, m1_est, n1_est)
    G2_est = get_gbr_matrix(l2_est, m2_est, n2_est)
    G21_tilde_est = np.linalg.inv(G1_est).T @ R21_est @ G2_est.T
    phi_est, eta_est, theta_est = matrix_to_euler(R21_est)


    eta_est_ = matrix_to_euler(R21_est)[1]
    param_init_ = encode_param_sym(R21_est, eta_est_)

    def res_func_sym(param):
        l1, m1, n1, l2, m2, n2, R21 = decode_param_sym(param)

        return res_func(
            encode_param(l1, m1, n1, l2, m2, n2, R21),
            pixel1[best_idx_in_nm], pixel2[best_idx_in_nm],
            normal1_nm[:0], normal2_nm[:0],
            normal1_nm[best_idx_in_nm], normal2_nm[best_idx_in_nm],
        )
    
    #print(decode_param_sym(param_init_))
    
    result = scipy.optimize.least_squares(
        res_func_sym, 
        param_init_,
        #xtol=1e-15,
        #gtol=1e-15,
        #ftol=1e-15,
        max_nfev=100,
        bounds=(
            [np.sign(eta_est_) * np.radians(5), -np.inf, -np.inf, -np.inf],
            [np.sign(eta_est_) * np.radians(85), np.inf, np.inf, np.inf]
        )
    )
    l1_est, m1_est, n1_est, l2_est, m2_est, n2_est,R21_est = decode_param_sym(result.x)
        


    G1_est = get_gbr_matrix(l1_est, m1_est, n1_est)
    G2_est = get_gbr_matrix(l2_est, m2_est, n2_est)
    G21_tilde_est = np.linalg.inv(G1_est).T @ R21_est @ G2_est.T
    phi_est, eta_est, theta_est = matrix_to_euler(R21_est)

    print(phi_est, eta_est, theta_est)
    print(l1_est, m1_est, n1_est, l2_est, m2_est, n2_est)

    if True:
        list_eta = []
        list_l1 = []
        list_l2 = []
        list_m1 = []
        list_m2 = []
        list_n1 = []
        list_n2 = []
        list_score = []
        phi, _, theta = matrix_to_euler(R21_est)
        G21_tilde = np.linalg.inv(G1_est).T @ R21_est @ G2_est.T
        # check rmap consistency
        for eta in np.arange(8.5 * np.pi / 180, 352. * np.pi / 180, 1 * np.pi / 180):
            if np.cos(eta) < np.cos(np.radians(75)):
                continue
            R21_est_a = euler_to_matrix(phi, eta, theta)
            if wo_gbr:
                R21_est_a = nm_corrs['R21']
            l1_est_a, m1_est_a, n1_est_a, l2_est_a, m2_est_a, n2_est_a = decompose_relative_gbr(G21_tilde, R21_est_a)

            G1_est_a = get_gbr_matrix(l1_est_a, m1_est_a, n1_est_a)
            G2_est_a = get_gbr_matrix(l2_est_a, m2_est_a, n2_est_a)
            G21_tilde_init =  np.linalg.inv(G1_est_a).T @ R21_est_a @ G2_est_a.T
            if np.sum((G21_tilde_init - G21_tilde)**2) > 1e-3:        
                continue

            G1_est_a = get_gbr_matrix(l1_est_a, m1_est_a, n1_est_a)
            G2_est_a = get_gbr_matrix(l2_est_a, m2_est_a, n2_est_a)

            #score_nm, idx_in_nm = eval_param_nm(encode_param(l1_est_a, m1_est_a, n1_est_a, l2_est_a, m2_est_a, n2_est_a,R21_est_a))
            #score_rm, idx_in_rm = eval_param_rm(encode_param(l1_est_a, m1_est_a, n1_est_a, l2_est_a, m2_est_a, n2_est_a,R21_est_a))
            score_reg = eval_param_reg(encode_param(l1_est_a, m1_est_a, n1_est_a, l2_est_a, m2_est_a, n2_est_a,R21_est_a))

            list_score.append(score_reg)
            list_eta.append(eta)

            list_l1.append(l1_est_a)
            list_l2.append(l2_est_a)
            list_m1.append(m1_est_a)
            list_m2.append(m2_est_a)
            list_n1.append(n1_est_a)
            list_n2.append(n2_est_a)

            if list_score[-1] == max(list_score):
                best_param_ = encode_param(l1_est_a, m1_est_a, n1_est_a, l2_est_a, m2_est_a, n2_est_a, R21_est_a)
                best_idx_in_rm_ = idx_in_rm
        #plt.plot(list_eta, list_score)
        #plt.show()
        best_param = best_param_

        l1_est, m1_est, n1_est, l2_est, m2_est, n2_est,R21_est = decode_param(best_param)

        G1_est = get_gbr_matrix(l1_est, m1_est, n1_est)
        G2_est = get_gbr_matrix(l2_est, m2_est, n2_est)
        G21_tilde_est = np.linalg.inv(G1_est).T @ R21_est @ G2_est.T
        phi_est, eta_est, theta_est = matrix_to_euler(R21_est)

        if False:
            plt.subplot(1,3,1)
            plt.plot(list_l1)
            plt.plot(list_l2)
            plt.subplot(1,3,2)
            plt.plot(list_m1)
            plt.plot(list_m2)
            plt.subplot(1,3,3)
            plt.plot(list_n1)
            plt.plot(list_n2)
            plt.show()

elif True:
    #result_dict = solve_pose({
    #    'pixel1': pixel1[best_idx_in_nm],
    #    'pixel2': pixel2[best_idx_in_nm],
    #    'normal1_nm': normal1_nm[best_idx_in_nm],
    #    'normal2_nm': normal2_nm[best_idx_in_nm],
    #    'normal1_rm': normal1_nm[:0], # normal1_rm,
    #    'normal2_rm': normal2_nm[:0], # normal2_rm,
    #}, opencv_coord=True, wo_gbr=wo_gbr, use_multi_initial_params=True, use_two_steps=True)

    l1_est, m1_est, n1_est, l2_est, m2_est, n2_est,R21_est = decode_param(best_param)
    phi_est, eta_est, theta_est = matrix_to_euler(R21_est)

    result = scipy.optimize.least_squares(
        res_func, 
        best_param,
        args=(
            pixel1[best_idx_in_nm], pixel2[best_idx_in_nm],
            normal1_nm_sampled[:0], normal2_nm_sampled[:0],
            normal1_nm[best_idx_in_nm], normal2_nm[best_idx_in_nm],
        ),
        xtol=1e-15,
        gtol=1e-15,
        ftol=1e-15,
        #max_nfev=50,
        loss='soft_l1',
    )
    print(result)
    best_param = result.x
    l1_est, m1_est, n1_est, l2_est, m2_est, n2_est,R21_est = decode_param(best_param)


    G1_est = get_gbr_matrix(l1_est, m1_est, n1_est)
    G2_est = get_gbr_matrix(l2_est, m2_est, n2_est)
    G21_tilde_est = np.linalg.inv(G1_est).T @ R21_est @ G2_est.T
    phi_est, eta_est, theta_est = matrix_to_euler(R21_est)

    if True:
        list_eta = []
        list_score = []
        phi, _, theta = matrix_to_euler(R21_est)
        G21_tilde = np.linalg.inv(G1_est).T @ R21_est @ G2_est.T
        # check rmap consistency
        for eta in np.arange(8.5 * np.pi / 180, 352. * np.pi / 180, 1 * np.pi / 180):
            if np.cos(eta) < np.cos(np.radians(75)):
                continue
            R21_est_a = euler_to_matrix(phi, eta, theta)
            if wo_gbr:
                R21_est_a = nm_corrs['R21']
            l1_est_a, m1_est_a, n1_est_a, l2_est_a, m2_est_a, n2_est_a = decompose_relative_gbr(G21_tilde, R21_est_a)

            G1_est_a = get_gbr_matrix(l1_est_a, m1_est_a, n1_est_a)
            G2_est_a = get_gbr_matrix(l2_est_a, m2_est_a, n2_est_a)
            G21_tilde_init =  np.linalg.inv(G1_est_a).T @ R21_est_a @ G2_est_a.T
            if np.sum((G21_tilde_init - G21_tilde)**2) > 1e-3:        
                continue

            G1_est_a = get_gbr_matrix(l1_est_a, m1_est_a, n1_est_a)
            G2_est_a = get_gbr_matrix(l2_est_a, m2_est_a, n2_est_a)

            #score_nm, idx_in_nm = eval_param_nm(encode_param(l1_est_a, m1_est_a, n1_est_a, l2_est_a, m2_est_a, n2_est_a,R21_est_a))
            #score_rm, idx_in_rm = eval_param_rm(encode_param(l1_est_a, m1_est_a, n1_est_a, l2_est_a, m2_est_a, n2_est_a,R21_est_a))
            score_reg = eval_param_reg(encode_param(l1_est_a, m1_est_a, n1_est_a, l2_est_a, m2_est_a, n2_est_a,R21_est_a))

            list_score.append(score_reg)
            list_eta.append(eta)

            if list_score[-1] == max(list_score):
                best_param_ = encode_param(l1_est_a, m1_est_a, n1_est_a, l2_est_a, m2_est_a, n2_est_a, R21_est_a)
                best_idx_in_rm_ = idx_in_rm
        #plt.plot(list_eta, list_score)
        #plt.show()
        best_param = best_param_

        l1_est, m1_est, n1_est, l2_est, m2_est, n2_est,R21_est = decode_param(best_param)

        G1_est = get_gbr_matrix(l1_est, m1_est, n1_est)
        G2_est = get_gbr_matrix(l2_est, m2_est, n2_est)
        G21_tilde_est = np.linalg.inv(G1_est).T @ R21_est @ G2_est.T
        phi_est, eta_est, theta_est = matrix_to_euler(R21_est)

elif False:
    pixel1_cv = pixel1 * np.array([1.,-1.])
    pixel2_cv = pixel2 * np.array([1.,-1.])
    normal1_nm_cv = normal1_nm * np.array([1., -1., -1.])
    normal2_nm_cv = normal2_nm * np.array([1., -1., -1.])
    normal1_rm_cv = normal1_rm * np.array([1., -1., -1.])
    normal2_rm_cv = normal2_rm * np.array([1., -1., -1.])
    l1_est, m1_est, n1_est, l2_est, m2_est, n2_est,R21_est = decode_param(best_param)
    best_param_cv = encode_param(
        l1_est, -m1_est, n1_est, 
        l2_est, -m2_est, n2_est,
        R21_est * np.array([
                [1., -1., -1.],
                [-1., 1., 1.],
                [-1., 1., 1.]
        ])
    )
    eta_est_cv = matrix_to_euler(
        R21_est * np.array([
                [1., -1., -1.],
                [-1., 1., 1.],
                [-1., 1., 1.]
        ])
    )[1]

    result = scipy.optimize.least_squares(
        res_func, 
        best_param_cv,
        args=(
            pixel1_cv[best_idx_in_nm], pixel2_cv[best_idx_in_nm],
            normal1_rm_cv[:0], normal2_rm_cv[:0], 
            #normal1_rm_cv[best_idx_in_rm], normal2_rm_cv[best_idx_in_rm],
            normal1_nm_cv[best_idx_in_nm], normal2_nm_cv[best_idx_in_nm],
            eta_est_cv
        ),
        xtol=1e-15,
        gtol=1e-15,
        ftol=1e-15,
        #max_nfev=50,
        #loss='soft_l1',
    )
    print(result)
    l1_est, m1_est, n1_est, l2_est, m2_est, n2_est,R21_est = decode_param(result.x)

    R21_est = R21_est * np.array([
        [1., -1., -1.],
        [-1., 1., 1.],
        [-1., 1., 1.]
    ])
    m1_est *= -1.
    m2_est *= -1.

    G1_est = get_gbr_matrix(l1_est, m1_est, n1_est)
    G2_est = get_gbr_matrix(l2_est, m2_est, n2_est)
    G21_tilde_est = np.linalg.inv(G1_est).T @ R21_est @ G2_est.T
    phi_est, eta_est, theta_est = matrix_to_euler(R21_est)
else:
    l1_est, m1_est, n1_est, l2_est, m2_est, n2_est,R21_est = decode_param(best_param)
    G1_est = get_gbr_matrix(l1_est, m1_est, n1_est)
    G2_est = get_gbr_matrix(l2_est, m2_est, n2_est)
    G21_tilde_est = np.linalg.inv(G1_est).T @ R21_est @ G2_est.T
    phi_est, eta_est, theta_est = matrix_to_euler(R21_est)

if False:
    R21_est = result_dict['R21_est']
    G1_est = result_dict['G1_est']
    G2_est = result_dict['G2_est']
    G21_tilde_est = result_dict['G21_est']
    l1_est = result_dict['l1_est']
    m1_est = result_dict['m1_est']
    n1_est = result_dict['n1_est']
    l2_est = result_dict['l2_est']
    m2_est = result_dict['m2_est']
    n2_est = result_dict['n2_est']
    phi_est, eta_est, theta_est = matrix_to_euler(R21_est)

    result = result_dict['optimization_result']
    print('optimization result:')
    print(result.fun)

# 
print('Pose estimated by RANSAC')
print(R21_est)
print('Euler Angles')
print(matrix_to_euler(R21_est))
print_pose_error(R21_est, gt_rot_21)
print('GBR parames estimated by RANSAC')
print(l1_est, m1_est, n1_est, l2_est, m2_est, n2_est)
print('Estimated compound transformation G21_tilde:')
print(G21_tilde_est)
print('det:', np.linalg.det(G21_tilde_est))

print('Number of Corrs:', len(normal1_nm))
print('Number of Estimated Inlier Corrs:', len(normal1_nm[best_idx_in_nm]))

normal1_undistorted = (normal1.view(3,-1).contiguous().T @ torch.from_numpy(G1_est).float().to(normal1.device)).T.view(1,3,256,256)
normal1_undistorted /= torch.sqrt(torch.sum(normal1_undistorted**2, dim=1,keepdim=True) + 1e-20)
normal2_undistorted = (normal2.view(3,-1).contiguous().T @ torch.from_numpy(G2_est).float().to(normal2.device)).T.view(1,3,256,256)
normal2_undistorted /= torch.sqrt(torch.sum(normal2_undistorted**2, dim=1,keepdim=True) + 1e-20)

normals_undistorted = torch.cat([normal1_undistorted, normal2_undistorted], dim=0)
print(normals_undistorted.size())
save_normal_map(out_dir+'/est_normal_sfs_final_undistorted_nm.png', torch.cat(normals_undistorted.unbind(0), dim=-1))


np.savez(
    out_dir+'/corrs_in_'+str(idx1)+'_'+str(idx2)+'_nm.npz',
    normal1=normal1_nm[best_idx_in_nm],
    normal2=normal2_nm[best_idx_in_nm],
    coord1=pixel1[best_idx_in_nm],
    coord2=pixel2[best_idx_in_nm],
    G21_tilde=G21_tilde_est,
    phi=phi_est,
    eta=eta_est,
    theta=theta_est,
)

result_text = ''
result_text += 'R21_error_deg: '+str(np.degrees(compute_pose_error(R21_est, gt_rot_21)))+'\n'
result_text += 'euler_angles_deg_est: '+' '.join([str(np.degrees(v)) for v in matrix_to_euler(R21_est)])+'\n'
result_text += 'euler_angles_deg_gt: '+' '.join([str(np.degrees(v)) for v in matrix_to_euler(gt_rot_21)])+'\n'
result_text += 'gbr_params_1_est: '+' '.join([str(v) for v in decode_param(best_param)[:-1][:3]])+'\n'
result_text += 'gbr_params_2_est: '+' '.join([str(v) for v in decode_param(best_param)[:-1][3:]])
with open(out_dir+'/accuracy_nm_'+str(idx1)+'_'+str(idx2)+'.txt', 'w') as f:
    f.write(result_text)

# draw inlier matches
kp1 = [cv2.KeyPoint(u,v, 1.) for u,v in pixel1[best_idx_in_nm].astype(np.float32)]
kp2 = [cv2.KeyPoint(u,v, 1.) for u,v in pixel2[best_idx_in_nm].astype(np.float32)]
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
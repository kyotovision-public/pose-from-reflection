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
from core.rmap_utils import create_normal_grid, sample_rmap, rotate_rmap
from tqdm import tqdm

import scipy

from core.sfs_utils import plot_hdr,save_normal_map,save_hdr,save_hdr_as_ldr
from core.sfm_utils import decompose_relative_gbr, euler_to_matrix, solve_translation_perspective, solve_surf_point_locations_perspective, matrix_to_euler

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
parser.add_argument('--normal-reg', action='store_true')
parser.add_argument('--prefiltering', action='store_true')
parser.add_argument('--static-camera', action='store_true')
args = parser.parse_args()

debug = False

#result_dir='./run/test_on_synthetic_erroneous/080'
result_dir=args.result_dir
wo_data_aug = args.wo_data_aug
wo_gbr = args.wo_gbr
sym = args.sym
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
if sym:
    out_dir+='_sym'
os.makedirs(out_dir, exist_ok=True)

idx1 = args.first_view
idx2 = args.second_view
if idx1 > idx2:
    idx1 = args.second_view
    idx2 = args.first_view

th_rm_deg = 10
th_nm_deg = 10
th_reg_deg = 20
th_rm_score = 500
th_nm_num = 1000
optimize_gbr = True
rm_corrs = np.load(out_dir+'/corrs_'+str(idx1)+'_'+str(idx2)+'_rm.npz')
nm_corrs = np.load(out_dir+'/corrs_in_'+str(idx1)+'_'+str(idx2)+'_nm.npz')
rmap = cv2.imread(result_dir+'/est_rmap_final.exr',-1)
if os.path.exists(result_dir+'/est_rmap_nlsfs_final.pt'):
    rmap = cv2.imread(result_dir+'/est_rmap_nlsfs_final.exr',-1)
elif os.path.exists(result_dir+'/est_rmap_sfs_final.pt'):
    rmap = cv2.imread(result_dir+'/est_rmap_sfs_final.exr',-1)

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

if os.path.exists(result_dir+'/gt_gbr_params.pt'):
    gt_gbr_params = torch.load(result_dir+'/gt_gbr_params.pt').numpy()
    l1_gt, m1_gt, n1_gt = gt_gbr_params[idx1] * np.array([1., -1., 1.])
    l2_gt, m2_gt, n2_gt = gt_gbr_params[idx2] * np.array([1., -1., 1.])
    print('GT GBR params:')
    print(l1_gt, m1_gt, n1_gt, l2_gt, m2_gt, n2_gt)

normal1_rm = rm_corrs['normal1']
normal2_rm = rm_corrs['normal2']
rm_coord1 = rm_corrs['rm_coord1']
rm_coord2 = rm_corrs['rm_coord2']
rmap1_fea = torch.from_numpy(rm_corrs['fea1'])
rmap2_fea = torch.from_numpy(rm_corrs['fea2'])

pixel1 = nm_corrs['coord1']
pixel2 = nm_corrs['coord2']
normal1_nm = nm_corrs['normal1']
normal2_nm = nm_corrs['normal2']
G21_tilde = nm_corrs['G21_tilde']
phi = nm_corrs['phi']
theta = nm_corrs['theta']

if False:
    normal_grid = create_normal_grid(128, 'probe').numpy() * np.array([1., -1., -1.])
    density_grid_1 = np.zeros_like(normal_grid[:,:,0])
    density_grid_2 = np.zeros_like(normal_grid[:,:,0])
    normal1_rm_density = np.zeros_like(normal1_rm[:,0])
    normal2_rm_density = np.zeros_like(normal2_rm[:,0])
    bar = tqdm(range(len(normal1_rm[:])))
    bar.set_description('Corrs Density Estimation (RM)')
    for idx_normal in bar:
        density_grid_1 += np.exp(-100 * (1. - np.sum(normal_grid * normal1_rm[idx_normal], axis=-1)))
        normal1_rm_density += np.exp(-100 * (1. - np.sum(normal1_rm * normal1_rm[idx_normal], axis=-1)))
        density_grid_2 += np.exp(-100 * (1. - np.sum(normal_grid * normal2_rm[idx_normal], axis=-1)))
        normal2_rm_density += np.exp(-100 * (1. - np.sum(normal2_rm * normal2_rm[idx_normal], axis=-1)))

    sample_importance_rm = 1. / ((normal1_rm_density * normal2_rm_density)**(0.5))

    if True:
        #plt.semilogy(sorted(sample_importance_nm))
        #plt.show()

        plt.subplot(2,3,1)
        plt.scatter(normal1_rm[:,0], normal1_rm[:,1])
        plt.subplot(2,3,2)
        plt.imshow(0.5 * (normal_grid * np.array([1., -1., -1.]) + 1.0))
        plt.subplot(2,3,3)
        plt.imshow(density_grid_1)
        plt.subplot(2,3,4)
        plt.scatter(normal2_rm[:,0], normal2_rm[:,1])
        plt.subplot(2,3,5)
        plt.imshow(0.5 * (normal_grid * np.array([1., -1., -1.]) + 1.0))
        plt.subplot(2,3,6)
        plt.imshow(density_grid_2)
        plt.savefig(out_dir+'/density_rm.png')
        plt.close()

def normal_to_light(normal):
    n2 = np.sum(normal**2, axis=-1) + 1e-20
    return np.stack([
        -2. * normal[:,0] * normal[:,2] / n2,
        -2. * normal[:,1] * normal[:,2] / n2,
        -2. * normal[:,2]**2 / n2 + 1.,
    ], axis=-1)

def light_to_normal(light):
    n = light + np.array([0.,0.,-1.])
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-20)

def undistort_normal(normal_gt, l, m, n):
    normal =  np.stack([
        normal_gt[:,0] - m * normal_gt[:,2],
        normal_gt[:,1] + n * normal_gt[:,2],
        l * normal_gt[:,2],
    ], axis=-1)
    return normal / np.linalg.norm(normal, axis=-1, keepdims=True)

def distort_normal(normal, l, m, n):
    return undistort_normal(normal, 1./l, -m/l, -n/l)

# compute R that satisfies v1.T = R@v2.T
def compute_relative_rot(v1, v2):
    X = v1.T
    Y = v2.T
    U,s,Vt = np.linalg.svd(Y@(X.T))
    H = np.diag([1.0, 1.0, np.linalg.det(Vt.T@U.T)])
    R21 = Vt.T@H@U.T
    return R21

def encode_param(l1, m1, n1, l2, m2, n2, R21):
    logl1 = np.log(l1)
    logl2 = np.log(l2)
    rvec = cv2.Rodrigues(R21)[0][:,0]
    return logl1, m1, n1, logl2, m2, n2, rvec[0], rvec[1], rvec[2]

def decode_param(param):
    logl1, m1, n1, logl2, m2, n2 = param[:6]
    l1 = np.exp(logl1)
    l2 = np.exp(logl2)
    R21 = cv2.Rodrigues(param[6:9])[0]
    if not optimize_gbr:
        return 1., 0., 0., 1., 0., 0., R21
    return l1, m1, n1, l2, m2, n2, R21

def get_gbr_matrix(l,m,n):
    return np.array([
        [1., 0., 0.],
        [0., 1., 0.],
        [-m, n, l]
    ])

def warp_normal(param, normal1_rm, normal2_rm, normal1_nm, normal2_nm):
    l1, m1, n1, l2, m2, n2, R21 = decode_param(param)

    normal1_rm_warped = distort_normal(
        light_to_normal(
            normal_to_light(
                undistort_normal(normal1_rm, l1, m1, n1)
            ) @ R21
        ),
        l2,m2,n2
    )
    normal2_rm_warped = distort_normal(
        light_to_normal(
            normal_to_light(
                undistort_normal(normal2_rm, l2, m2, n2)
            ) @ R21.T
        ),
        l1,m1,n1
    )

    normal1_nm_warped = distort_normal(
        undistort_normal(normal1_nm, l1, m1, n1) @ R21,
        l2,m2,n2
    )
    normal2_nm_warped = distort_normal(
        undistort_normal(normal2_nm, l2, m2, n2) @ R21.T,
        l1,m1,n1
    )
    return normal1_rm_warped, normal2_rm_warped, normal1_nm_warped, normal2_nm_warped

def res_func(param, normal1_rm, normal2_rm, normal1_nm, normal2_nm):
    l1, m1, n1, l2, m2, n2, R21 = decode_param(param)

    normal1_rm_warped, normal2_rm_warped, normal1_nm_warped, normal2_nm_warped = warp_normal(
        param, normal1_rm, normal2_rm, normal1_nm, normal2_nm
    )

    #sa_rm_1 = np.sinc(2 * np.arccos(np.clip(-normal1_rm[:,2:3], -1, 1)) / np.pi)
    #sa_rm_2 = np.sinc(2 * np.arccos(np.clip(-normal2_rm[:,2:3], -1, 1)) / np.pi)
    #weight = sa_rm_1 * sa_rm_2
    #weight = weight / np.sum(weight)
    res_rm = np.concatenate([
        ((normal1_rm - normal2_rm_warped)).reshape(-1),
        ((normal2_rm - normal1_rm_warped)).reshape(-1)
    ])
    #res_rm = (normal_to_light(normal1_rm) @ R21 - normal_to_light(normal2_rm)).reshape(-1)

    res_nm = np.concatenate([
        (normal1_nm - normal2_nm_warped).reshape(-1),
        (normal2_nm - normal1_nm_warped).reshape(-1),
    ])
    #res_nm = (normal1_nm @ R21 - normal2_nm).reshape(-1)

    return np.concatenate([res_rm, res_nm])




def compute_pose_error(R_est, R_gt):
    return np.arccos(np.clip(0.5 * (np.trace(R_est @ R_gt.T) - 1.), -1., 1.))

def print_pose_error(R_est, R_gt):
    print('Pose Error:', np.degrees(compute_pose_error(R_est, R_gt)))

R21_est_naive = compute_relative_rot(
    normal_to_light(normal1_rm),#np.concatenate([normal_to_light(normal1_rm), normal1_nm], axis=0), 
    normal_to_light(normal2_rm)#np.concatenate([normal_to_light(normal2_rm), normal2_nm], axis=0), 
)
print('Pose estimated by naive Kabsch algorithm(rm)')
print(R21_est_naive)
print_pose_error(R21_est_naive, gt_rot_21)

R21_est_naive = compute_relative_rot(
    normal1_nm,#np.concatenate([normal_to_light(normal1_rm), normal1_nm], axis=0), 
    normal2_nm#np.concatenate([normal_to_light(normal2_rm), normal2_nm], axis=0), 
)
print('Pose estimated by naive Kabsch algorithm(inlier nm)')
print(R21_est_naive)
print_pose_error(R21_est_naive, gt_rot_21)


result_wo_ransac = scipy.optimize.least_squares(
    res_func, 
    encode_param(1.,0.,0.,1.,0.,0., R21_est_naive), 
    args=(normal1_rm, normal2_rm, normal1_nm, normal2_nm), 
    xtol=1e-15,
    gtol=1e-15,
    ftol=1e-15,
    max_nfev=1000,
)


R21_est_wo_ransac = decode_param(result_wo_ransac.x)[-1]
print('Pose estimated without RANSAC')
print(R21_est_wo_ransac)
print_pose_error(R21_est_wo_ransac, gt_rot_21)
print('GBR parames estimated without RANSAC')
print(decode_param(result_wo_ransac.x)[:-1])

# Dense search w.r.t eta
best_score = 0

normal1_rm_e1 = np.cross(normal1_rm, np.stack([-normal1_rm[:,1], normal1_rm[:,0], 0 * normal1_rm[:,0]], axis=-1))
normal1_rm_e1 /= (np.linalg.norm(normal1_rm_e1, axis=1, keepdims=True))
normal1_rm_e2 = np.cross(normal1_rm_e1, normal1_rm)
normal1_rm_e2 /= (np.linalg.norm(normal1_rm_e2, axis=1, keepdims=True))

def eval_param_reg(param):

    l1_est, m1_est, n1_est, l2_est, m2_est, n2_est,R21_est = decode_param(param)

    G1_est = get_gbr_matrix(l1_est, m1_est, n1_est)
    G2_est = get_gbr_matrix(l2_est, m2_est, n2_est)

    score_reg = np.exp(
        -1. * (
            np.linalg.norm(G1_est - np.eye(3),ord='fro')**2 + 
            np.linalg.norm(G2_est - np.eye(3),ord='fro')**2
        )
    )

    return score_reg

if debug:
    corrected_rms = []
    corrected_nms = []
    list_eta = []
    list_rmap_fea_residual = []
    normal_grid = create_normal_grid(rmap1.size(-1),'probe').numpy()
    solid_angle = np.sinc(np.arccos(np.clip(normal_grid[...,2], -1, 1)) / np.pi) # solid angle
    solid_angle = solid_angle * (np.sum(normal_grid**2,axis=-1) > 0.9**2)

    for eta in np.arange(0.5 * np.pi / 180, 352. * np.pi / 180, 2.5 * np.pi / 180):
        R21_est_a = euler_to_matrix(phi, eta, theta)
        if wo_gbr:
            R21_est_a = nm_corrs['R21']
        l1_est_a, m1_est_a, n1_est_a, l2_est_a, m2_est_a, n2_est_a = decompose_relative_gbr(G21_tilde, R21_est_a)

        G1_est_a = get_gbr_matrix(l1_est_a, -m1_est_a, n1_est_a)
        G2_est_a = get_gbr_matrix(l2_est_a, -m2_est_a, n2_est_a)
        G21_tilde_init =  np.linalg.inv(G1_est_a).T @ R21_est_a @ G2_est_a.T
        if np.sum((G21_tilde_init - G21_tilde)**2) > 1e-3:            
            continue

        G1_est_a = torch.from_numpy(get_gbr_matrix(l1_est_a, m1_est_a, n1_est_a)).float()
        G2_est_a = torch.from_numpy(get_gbr_matrix(l2_est_a, m2_est_a, n2_est_a)).float()

        normals = torch.load(result_dir+'/est_normal_sfs_final.pt')
        if os.path.exists(result_dir+'/est_normal_nlsfs_final.pt'):
            normals = torch.load(result_dir+'/est_normal_nlsfs_final.pt')
        normal1 = normals[idx1:idx1+1] # [1,3,256,256]
        normal2 = normals[idx2:idx2+1] # [1,3,256,256]

        normal1_undistorted = (normal1.view(3,-1).contiguous().T @ G1_est_a).T.view(1,3,256,256)
        normal1_undistorted /= torch.sqrt(torch.sum(normal1_undistorted**2, dim=1,keepdim=True) + 1e-20)
        normal2_undistorted = (normal2.view(3,-1).contiguous().T @ G2_est_a).T.view(1,3,256,256)
        normal2_undistorted /= torch.sqrt(torch.sum(normal2_undistorted**2, dim=1,keepdim=True) + 1e-20)

        normals_undistorted = torch.cat([normal1_undistorted, normal2_undistorted], dim=0)

        R21_est_a_ = R21_est_a * np.array([
            [1., -1., -1.],
            [-1., 1., 1.],
            [-1., 1., 1.]
        ])
        normal2_undistorted_aligned = (normal2_undistorted.view(3,-1).contiguous().T @ R21_est_a_.T).T.view(1,3,256,256)
        normals_undistorted_aligned = torch.cat([normal1_undistorted, normal2_undistorted_aligned], dim=0)

        print(normals_undistorted.size())
        print('eta:', eta)

        def distort_rmap(rmap, G):
            normal_grid = create_normal_grid(256,'probe')[64:192,64:192]
            n_ = (G.T[None,None,:,:] @ normal_grid[:,:,:,None])[...,0]
            n_ = n_ / torch.sqrt(torch.sum(n_**2, dim=-1, keepdim=True) + 1e-6)
            n_ = torch.stack(n_.unbind(-1), dim=0)[None].repeat(len(rmap),1,1,1)

            rmap_pad = torch.nn.functional.pad(rmap, (64,64,64,64), "constant", 0)
            print(rmap_pad.size())
            print(n_.size())
            rmap_ = sample_rmap(rmap_pad, n_, projection_mode='probe', interep_mode='bilinear')
            return rmap_ * (normal_grid[...,2] > 0).float()

        rmap1_undistorted = distort_rmap(rmap1.cpu(), torch.inverse(G1_est_a))
        rmap2_undistorted = distort_rmap(rmap2.cpu(), torch.inverse(G2_est_a))
        rmap_undistorted = torch.cat([rmap1_undistorted, rmap2_undistorted], dim=0)

        rmap1_fea_undistorted = distort_rmap(rmap1_fea.cpu(), torch.inverse(G1_est_a))
        rmap2_fea_undistorted = distort_rmap(rmap2_fea.cpu(), torch.inverse(G2_est_a))

        rmap2_undistorted_aligned = rotate_rmap(
            rmap2_undistorted, 
            torch.eye(3)[None], 
            torch.from_numpy(R21_est_a.T).float()[None], 
            projection_mode='probe'
        )
        rmap2_fea_undistorted_aligned = rotate_rmap(
            rmap2_fea_undistorted, 
            torch.eye(3)[None], 
            torch.from_numpy(R21_est_a.T).float()[None], 
            projection_mode='probe'
        )
        rmap_undistorted_aligned = torch.cat([rmap1_undistorted, rmap2_undistorted_aligned], dim=0)

        rmap_fea_residual_map = torch.sum((rmap1_fea_undistorted - rmap2_fea_undistorted_aligned)**2, dim=1)[0]
        rmap_residual_map = torch.sum((rmap1_undistorted - rmap2_undistorted_aligned)**2, dim=1)[0]
        rmap_residual = torch.mean(rmap_residual_map * solid_angle)
        rmap_fea_residual = torch.mean(rmap_fea_residual_map * solid_angle)

        list_eta.append(eta)
        list_rmap_fea_residual.append(rmap_fea_residual)


        def no_tick():
            plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False) 
            plt.tick_params(bottom=False,left=False,right=False,top=False)
            plt.box(False)
        from core.sfs_utils import plot_normal_map
        plt.suptitle('eta={:.2f}'.format(eta))
        plt.subplot(4,2,1)
        plot_normal_map(torch.cat(normals.unbind(0), dim=-1)[None])
        no_tick()
        plt.ylabel('Input')
        plt.subplot(4,2,2)
        plot_hdr(torch.cat([rmap1, rmap2], dim=-1))
        no_tick()

        plt.subplot(4,2,3)
        plot_normal_map(torch.cat(normals_undistorted.unbind(0), dim=-1)[None])
        no_tick()
        plt.ylabel('Undistorted')
        plt.subplot(4,2,4)
        plot_hdr(torch.cat(rmap_undistorted.unbind(0), dim=-1)[None])
        no_tick()
        plt.subplot(4,2,5)
        plot_normal_map(torch.cat(normals_undistorted_aligned.unbind(0), dim=-1)[None])
        plt.ylabel('Aligned')
        no_tick()
        plt.subplot(4,2,6)
        plot_hdr(torch.cat(rmap_undistorted_aligned.unbind(0), dim=-1)[None])
        no_tick()
        plt.subplot(4,2,7)
        plt.imshow(rmap_residual_map, vmin=0, vmax=1.)
        no_tick()
        plt.subplot(4,2,8)
        plt.imshow(rmap_fea_residual_map, vmin=0, vmax=1.)
        no_tick()
        os.makedirs(out_dir+'/debug',exist_ok=True)
        plt.savefig(out_dir+'/debug/{:.2f}.png'.format(eta))
        #plt.show()
        plt.close()
    plt.plot(list_eta, list_rmap_fea_residual)
    plt.show()

while True:
    list_eta = []
    list_score = []
    list_pose_error = []

    for eta in np.arange(0.5 * np.pi / 180, 352. * np.pi / 180, 1 * np.pi / 180):
        R21_est_a = euler_to_matrix(phi, eta, theta)
        if wo_gbr:
            R21_est_a = nm_corrs['R21']
        l1_est_a, m1_est_a, n1_est_a, l2_est_a, m2_est_a, n2_est_a = decompose_relative_gbr(G21_tilde, R21_est_a)

        G1_est_a = get_gbr_matrix(l1_est_a, -m1_est_a, n1_est_a)
        G2_est_a = get_gbr_matrix(l2_est_a, -m2_est_a, n2_est_a)
        G21_tilde_init =  np.linalg.inv(G1_est_a).T @ R21_est_a @ G2_est_a.T
        if np.sum((G21_tilde_init - G21_tilde)**2) > 1e-3:            
            continue

        G1_est_a = get_gbr_matrix(l1_est_a, -m1_est_a, n1_est_a)
        G2_est_a = get_gbr_matrix(l2_est_a, -m2_est_a, n2_est_a)

        score_reg = eval_param_reg(encode_param(l1_est_a, m1_est_a, n1_est_a, l2_est_a, m2_est_a, n2_est_a,R21_est_a))

        # eval goodness using correspondences in RMs
        if args.static_camera:
            #print('static')
            normal1_rm_undistorted = normal1_rm @ G1_est_a
            normal1_rm_undistorted /= (np.linalg.norm(normal1_rm_undistorted,axis=1,keepdims=True) + 1e-20)

            normal1_rm_warped = light_to_normal(normal_to_light(normal1_rm @ G1_est_a))  @ np.linalg.inv(G2_est_a)
            normal1_rm_warped /= (np.linalg.norm(normal1_rm_warped, axis=1, keepdims=True) + 1e-20)
            normal2_rm_warped = light_to_normal(normal_to_light(normal2_rm @ G2_est_a))  @ np.linalg.inv(G1_est_a)
            normal2_rm_warped /= (np.linalg.norm(normal2_rm_warped, axis=1, keepdims=True) + 1e-20)


            err_rm_1 = np.arccos(np.clip(np.sum(normal1_rm * normal2_rm_warped, axis=-1),-1,1))
            err_rm_2 = np.arccos(np.clip(np.sum(normal2_rm * normal1_rm_warped, axis=-1),-1,1))

            idx_in_rm = (np.degrees(err_rm_1) < th_rm_deg) * (np.degrees(err_rm_2) < th_rm_deg) 

            sa_rm_1 = np.sinc(np.arccos(np.clip(-normal1_rm[:,2], -1, 1)) / np.pi) # solid angle
            ratio_area = np.linalg.norm(np.cross(normal1_rm_e1 @ G1_est_a, normal1_rm_e2 @ G1_est_a), axis=-1)
            ratio_area /= np.linalg.norm(normal1_rm_e1 @ G1_est_a, axis=-1)
            sa_rm_1 *= ratio_area # gbr transformation
            sa_rm_1 *= np.clip(-normal1_rm_undistorted[:,2], 0, 1) # normal to light dir
            #score_rm = np.sum(idx_in_rm * sa_rm_1)
            score_rm = np.sum(np.exp(-(np.degrees(err_rm_1) / th_rm_deg)**2) * np.exp(-(np.degrees(err_rm_2) / th_rm_deg)**2) * sa_rm_1)
        else:
            normal1_rm_undistorted = normal1_rm @ G1_est_a
            normal1_rm_undistorted /= (np.linalg.norm(normal1_rm_undistorted,axis=1,keepdims=True) + 1e-20)

            normal1_rm_warped = light_to_normal(normal_to_light(normal1_rm @ G1_est_a) @ R21_est_a)  @ np.linalg.inv(G2_est_a)
            normal1_rm_warped /= (np.linalg.norm(normal1_rm_warped, axis=1, keepdims=True) + 1e-20)
            normal2_rm_warped = light_to_normal(normal_to_light(normal2_rm @ G2_est_a) @ R21_est_a.T)  @ np.linalg.inv(G1_est_a)
            normal2_rm_warped /= (np.linalg.norm(normal2_rm_warped, axis=1, keepdims=True) + 1e-20)


            err_rm_1 = np.arccos(np.clip(np.sum(normal1_rm * normal2_rm_warped, axis=-1),-1,1))
            err_rm_2 = np.arccos(np.clip(np.sum(normal2_rm * normal1_rm_warped, axis=-1),-1,1))

            idx_in_rm = (np.degrees(err_rm_1) < th_rm_deg) * (np.degrees(err_rm_2) < th_rm_deg) 

            sa_rm_1 = np.sinc(np.arccos(np.clip(-normal1_rm[:,2], -1, 1)) / np.pi) # solid angle
            ratio_area = np.linalg.norm(np.cross(normal1_rm_e1 @ G1_est_a, normal1_rm_e2 @ G1_est_a), axis=-1)
            ratio_area /= np.linalg.norm(normal1_rm_e1 @ G1_est_a, axis=-1)
            #sa_rm_1 *= ratio_area # gbr transformation
            #sa_rm_1 *= np.clip(-normal1_rm_undistorted[:,2], 0, 1) # normal to light dir
            #score_rm = np.sum(idx_in_rm * sa_rm_1)
            score_rm = np.sum(np.exp(-(np.degrees(err_rm_1) / th_rm_deg)**2) * np.exp(-(np.degrees(err_rm_2) / th_rm_deg)**2) * sa_rm_1)

        list_score.append(score_rm * score_reg**0.1)
        list_eta.append(eta)

        if list_score[-1] == max(list_score):
            best_param = encode_param(l1_est_a, m1_est_a, n1_est_a, l2_est_a, m2_est_a, n2_est_a, R21_est_a)
            best_idx_in_rm = idx_in_rm
            best_eta = eta

    if max(list_score) > th_rm_score:
        break
    if th_rm_deg > 8:
        break

    th_rm_deg *= 1.4

plt.plot(list_eta, list_score, label='th_deg='+str(th_rm_deg))
plt.grid()
plt.savefig(out_dir+'/eta_vs_score.png')
plt.close()
#plt.show()

if True:
    one_deg = np.pi / 180
    list_eta = []
    list_score = []
    list_pose_error = []
    for eta in np.arange(best_eta-2*one_deg, best_eta+2*one_deg, 0.01 * np.pi / 180):
        R21_est_a = euler_to_matrix(phi, eta, theta)
        if wo_gbr:
            R21_est_a = nm_corrs['R21']
        l1_est_a, m1_est_a, n1_est_a, l2_est_a, m2_est_a, n2_est_a = decompose_relative_gbr(G21_tilde, R21_est_a)

        G1_est_a = get_gbr_matrix(l1_est_a, -m1_est_a, n1_est_a)
        G2_est_a = get_gbr_matrix(l2_est_a, -m2_est_a, n2_est_a)
        G21_tilde_init =  np.linalg.inv(G1_est_a).T @ R21_est_a @ G2_est_a.T
        if np.sum((G21_tilde_init - G21_tilde)**2) > 1e-3:
            continue

        G1_est_a = get_gbr_matrix(l1_est_a, -m1_est_a, n1_est_a)
        G2_est_a = get_gbr_matrix(l2_est_a, -m2_est_a, n2_est_a)

        score_reg = eval_param_reg(encode_param(l1_est_a, m1_est_a, n1_est_a, l2_est_a, m2_est_a, n2_est_a,R21_est_a))

        # eval goodness using correspondences in RMs
        if args.static_camera:
            normal1_rm_undistorted = normal1_rm @ G1_est_a
            normal1_rm_undistorted /= (np.linalg.norm(normal1_rm_undistorted,axis=1,keepdims=True) + 1e-20)

            normal1_rm_warped = light_to_normal(normal_to_light(normal1_rm @ G1_est_a))  @ np.linalg.inv(G2_est_a)
            normal1_rm_warped /= (np.linalg.norm(normal1_rm_warped, axis=1, keepdims=True) + 1e-20)
            normal2_rm_warped = light_to_normal(normal_to_light(normal2_rm @ G2_est_a))  @ np.linalg.inv(G1_est_a)
            normal2_rm_warped /= (np.linalg.norm(normal2_rm_warped, axis=1, keepdims=True) + 1e-20)


            err_rm_1 = np.arccos(np.clip(np.sum(normal1_rm * normal2_rm_warped, axis=-1),-1,1))
            err_rm_2 = np.arccos(np.clip(np.sum(normal2_rm * normal1_rm_warped, axis=-1),-1,1))

            idx_in_rm = (np.degrees(err_rm_1) < th_rm_deg) * (np.degrees(err_rm_2) < th_rm_deg) 

            sa_rm_1 = np.sinc(np.arccos(np.clip(-normal1_rm[:,2], -1, 1)) / np.pi) # solid angle
            ratio_area = np.linalg.norm(np.cross(normal1_rm_e1 @ G1_est_a, normal1_rm_e2 @ G1_est_a), axis=-1)
            ratio_area /= np.linalg.norm(normal1_rm_e1 @ G1_est_a, axis=-1)
            sa_rm_1 *= ratio_area # gbr transformation
            sa_rm_1 *= np.clip(-normal1_rm_undistorted[:,2], 0, 1) # normal to light dir
            #score_rm = np.sum(idx_in_rm * sa_rm_1)
            score_rm = np.sum(np.exp(-(np.degrees(err_rm_1) / th_rm_deg)**2) * np.exp(-(np.degrees(err_rm_2) / th_rm_deg)**2) * sa_rm_1)
        else:
            normal1_rm_undistorted = normal1_rm @ G1_est_a
            normal1_rm_undistorted /= (np.linalg.norm(normal1_rm_undistorted,axis=1,keepdims=True) + 1e-20)

            normal1_rm_warped = light_to_normal(normal_to_light(normal1_rm @ G1_est_a) @ R21_est_a)  @ np.linalg.inv(G2_est_a)
            normal1_rm_warped /= (np.linalg.norm(normal1_rm_warped, axis=1, keepdims=True) + 1e-20)
            normal2_rm_warped = light_to_normal(normal_to_light(normal2_rm @ G2_est_a) @ R21_est_a.T)  @ np.linalg.inv(G1_est_a)
            normal2_rm_warped /= (np.linalg.norm(normal2_rm_warped, axis=1, keepdims=True) + 1e-20)


            err_rm_1 = np.arccos(np.clip(np.sum(normal1_rm * normal2_rm_warped, axis=-1),-1,1))
            err_rm_2 = np.arccos(np.clip(np.sum(normal2_rm * normal1_rm_warped, axis=-1),-1,1))

            idx_in_rm = (np.degrees(err_rm_1) < th_rm_deg) * (np.degrees(err_rm_2) < th_rm_deg) 

            sa_rm_1 = np.sinc(np.arccos(np.clip(-normal1_rm[:,2], -1, 1)) / np.pi) # solid angle
            ratio_area = np.linalg.norm(np.cross(normal1_rm_e1 @ G1_est_a, normal1_rm_e2 @ G1_est_a), axis=-1)
            ratio_area /= np.linalg.norm(normal1_rm_e1 @ G1_est_a, axis=-1)
            sa_rm_1 *= ratio_area # gbr transformation
            sa_rm_1 *= np.clip(-normal1_rm_undistorted[:,2], 0, 1) # normal to light dir
            #score_rm = np.sum(idx_in_rm * sa_rm_1)
            score_rm = np.sum(np.exp(-(np.degrees(err_rm_1) / th_rm_deg)**2) * np.exp(-(np.degrees(err_rm_2) / th_rm_deg)**2) * sa_rm_1)

        list_eta.append(eta)
        list_score.append(score_rm * score_reg**0.1)

        if list_score[-1] == max(list_score):
            best_param = encode_param(l1_est_a, m1_est_a, n1_est_a, l2_est_a, m2_est_a, n2_est_a, R21_est_a)
            best_idx_in_rm = idx_in_rm
            best_eta = eta

    #plt.plot(list_eta, list_score)
    #plt.show()

l1_est, m1_est, n1_est, l2_est, m2_est, n2_est, R21_est = decode_param(best_param)

# translation estimation
if os.path.exists(result_dir+'/gt_intrinsic_matrices.pt'):
    intrinsics = torch.load(result_dir+'/gt_intrinsic_matrices.pt').numpy()
    K1 = intrinsics[idx1]
    K2 = intrinsics[idx2]
else:
    print('Intrinsic data not found')
    K = np.array([
        [4000, 0, 128],
        [0, 4000, 128],
        [0, 0, 1]
    ])
    K1 = K
    K2 = K

t21_est = solve_translation_perspective(pixel1, pixel2, K1, K2, R21_est)
surf_points_est = solve_surf_point_locations_perspective(pixel1, pixel2, K1, K2, R21_est, t21_est)
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
bsphere_radius = np.max(np.linalg.norm(surf_points_est - bbox_center, axis=-1))

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

print('GT Euler Angles:')
print(matrix_to_euler(gt_rot_21))
print('Est Euler Angles:')
print(matrix_to_euler(R21_est))

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

print('P1:')
print(P1_est)
print('P2:')
print(P2_est)

normals = torch.load(result_dir+'/est_normal_sfs_final.pt')
if os.path.exists(result_dir+'/est_normal_nlsfs_final.pt'):
    normals = torch.load(result_dir+'/est_normal_nlsfs_final.pt')
normal1 = normals[idx1:idx1+1] # [1,3,256,256]
normal2 = normals[idx2:idx2+1] # [1,3,256,256]

G1_est = torch.from_numpy(get_gbr_matrix(l1_est, m1_est, n1_est)).float()
G2_est = torch.from_numpy(get_gbr_matrix(l2_est, m2_est, n2_est)).float()

normal1_undistorted = (normal1.view(3,-1).contiguous().T @ G1_est).T.view(1,3,256,256)
normal1_undistorted /= torch.sqrt(torch.sum(normal1_undistorted**2, dim=1,keepdim=True) + 1e-20)
normal2_undistorted = (normal2.view(3,-1).contiguous().T @ G2_est).T.view(1,3,256,256)
normal2_undistorted /= torch.sqrt(torch.sum(normal2_undistorted**2, dim=1,keepdim=True) + 1e-20)

normals_undistorted = torch.cat([normal1_undistorted, normal2_undistorted], dim=0)
print(normals_undistorted.size())
torch.save(normals_undistorted, out_dir+'/est_normal_sfs_final_undistorted.pt')
save_normal_map(out_dir+'/est_normal_sfs_final_undistorted.png', torch.cat(normals_undistorted.unbind(0), dim=-1))

def distort_rmap(rmap, G):
    normal_grid = create_normal_grid(256,'probe')[64:192,64:192]
    n_ = (G.T[None,None,:,:] @ normal_grid[:,:,:,None])[...,0]
    n_ = n_ / torch.sqrt(torch.sum(n_**2, dim=-1, keepdim=True) + 1e-6)
    n_ = torch.stack(n_.unbind(-1), dim=0)[None].repeat(len(rmap),1,1,1)

    rmap_pad = torch.nn.functional.pad(rmap, (64,64,64,64), "constant", 0)
    print(rmap_pad.size())
    print(n_.size())
    rmap_ = sample_rmap(rmap_pad, n_, projection_mode='probe', interep_mode='bilinear')
    return rmap_ * (normal_grid[...,2] > 0).float()

rmap1_undistorted = distort_rmap(rmap1.cpu(), torch.inverse(G1_est))
rmap2_undistorted = distort_rmap(rmap2.cpu(), torch.inverse(G2_est))
rmap_undistorted = torch.cat([rmap1_undistorted, rmap2_undistorted], dim=0)
torch.save(rmap_undistorted, out_dir+'/est_rmap_final_undistorted.pt')
save_hdr_as_ldr(out_dir+'/est_rmap_final_undistorted.png', torch.cat(rmap_undistorted.unbind(0), dim=-1)[None])
save_hdr(out_dir+'/est_rmap_final_undistorted.exr', torch.cat(rmap_undistorted.unbind(0), dim=-1)[None])

rmap2_undistorted_aligned = rotate_rmap(
    rmap2_undistorted, 
    torch.eye(3)[None], 
    torch.from_numpy(R21_est.T).float()[None], 
    projection_mode='probe'
)
rmap_undistorted_aligned = torch.cat([rmap1_undistorted, rmap2_undistorted_aligned], dim=0)
torch.save(rmap_undistorted_aligned, out_dir+'/est_rmap_final_undistorted_aligned.pt')
save_hdr_as_ldr(out_dir+'/est_rmap_final_undistorted_aligned.png', torch.cat(rmap_undistorted_aligned.unbind(0), dim=-1)[None])
save_hdr(out_dir+'/est_rmap_final_undistorted_aligned.exr', torch.cat(rmap_undistorted_aligned.unbind(0), dim=-1)[None])

if True:
    import open3d as o3d
    surf_points_est = solve_surf_point_locations_perspective(pixel1, pixel2, K1, K2, R21_est, t21_est)
    surf_points_global_est = (np.linalg.inv(R1_est) @ (surf_points_est - t1_est).T).T
    pcd_est = o3d.geometry.PointCloud()
    pcd_est.points = o3d.utility.Vector3dVector(surf_points_global_est)

    o3d.io.write_point_cloud(out_dir+'/surf_points.ply', pcd_est)

assert np.linalg.norm(R21_est - R1_est @ np.linalg.inv(R2_est)) < 1e-4
assert np.linalg.norm(t21_est[:,None] - (t1_est[:,None] - R1_est @ np.linalg.inv(R2_est) @ t2_est[:,None])) < 1e-4
assert np.linalg.norm(P1_est[:3,:4] - K1 @ np.concatenate([R1_est, t1_est[:,None]], axis=1)) < 1e-4
assert np.linalg.norm(P2_est[:3,:4] - K2 @ np.concatenate([R2_est, t2_est[:,None]], axis=1)) < 1e-4

if False:
    proj_points_1_est = (P1_est[:3,:3] @ surf_points_global_est.T + P1_est[:3,3:4]).T
    proj_points_1_est = proj_points_1_est[:,:2] / proj_points_1_est[:,2:3]
    proj_points_2_est = (P2_est[:3,:3] @ surf_points_global_est.T + P2_est[:3,3:4]).T
    proj_points_2_est = proj_points_2_est[:,:2] / proj_points_2_est[:,2:3]
    plt.subplot(1,2,1)
    plt.scatter(proj_points_1_est[::100,0], proj_points_1_est[::100,1], marker='x')
    plt.scatter(pixel1[::100,0], pixel1[::100,1], marker='+')
    plt.subplot(1,2,2)
    plt.scatter(proj_points_2_est[::100,0], proj_points_2_est[::100,1], marker='x')
    plt.scatter(pixel2[::100,0], pixel2[::100,1], marker='+')
    plt.show()


result_text = ''
result_text += 'R21_error_deg: '+str(np.degrees(compute_pose_error(R21_est, gt_rot_21)))+'\n'
result_text += 'euler_angles_deg_est: '+' '.join([str(np.degrees(v)) for v in matrix_to_euler(R21_est)])+'\n'
result_text += 'euler_angles_deg_gt: '+' '.join([str(np.degrees(v)) for v in matrix_to_euler(gt_rot_21)])+'\n'
result_text += 'gbr_params_1_est: '+' '.join([str(v) for v in decode_param(best_param)[:-1][:3]])+'\n'
result_text += 'gbr_params_2_est: '+' '.join([str(v) for v in decode_param(best_param)[:-1][3:]])
with open(out_dir+'/accuracy_'+str(idx1)+'_'+str(idx2)+'.txt', 'w') as f:
    f.write(result_text)

kp1 = [cv2.KeyPoint(u,v, 1.) for u,v in rm_coord1[best_idx_in_rm].astype(np.float32)]
kp2 = [cv2.KeyPoint(u,v, 1.) for u,v in rm_coord2[best_idx_in_rm].astype(np.float32)]
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

print('Pose estimated by RANSAC')
print(R21_est)
print_pose_error(R21_est, gt_rot_21)
print('GBR parames estimated by RANSAC')
print(decode_param(best_param)[:-1])
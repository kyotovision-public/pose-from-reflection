import numpy as np
import torch
import pycalib.plot as cplt
import matplotlib.pyplot as plt
import os

from core.sfm_utils import matrix_to_euler

import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D
import trimesh

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('work_dir', type=str)
parser.add_argument('--corrs-folder-name', type=str, default=None)
parser.add_argument('--plot-gt', type=bool, default=True)
args = parser.parse_args()

work_dir=args.work_dir
corrs_folder_name=args.corrs_folder_name
plot_gt = args.plot_gt

if corrs_folder_name is None:
    if os.path.exists(f'{work_dir}/corrs_sym'):
        corrs_folder_name='corrs_sym'
    else:
        corrs_folder_name='corrs'

est_cam_file=work_dir+f'/{corrs_folder_name}/est_pose_0_1.npz'
gt_cam_file=work_dir+f'/gt_extrinsic_matrices.pt'
est_pcd_file=work_dir+f'/{corrs_folder_name}/surf_points.ply'
colmap_cam_file=None
if os.path.exists(work_dir+f'/{corrs_folder_name}/colmap_result.npz'):
    colmap_cam_file=work_dir+f'/{corrs_folder_name}/colmap_result.npz'
samurai_cam_file=None
if os.path.exists(work_dir+f'/{corrs_folder_name}/samurai_poses.npy'):
    samurai_cam_file=work_dir+f'/{corrs_folder_name}/samurai_poses.npy'

est_pcd = trimesh.load(est_pcd_file).vertices

def compute_pose_error(R_est, R_gt):
    return np.arccos(np.clip(0.5 * (np.trace(R_est @ R_gt.T) - 1.), -1., 1.))

est_pose = np.load(est_cam_file)
R1_est = est_pose['R1']
t1_est = est_pose['t1']
T1_est = np.concatenate([R1_est, t1_est[:,None]], axis=1)
R2_est = est_pose['R2']
t2_est = est_pose['t2']
T2_est = np.concatenate([R2_est, t2_est[:,None]], axis=1)

gt_pose = torch.load(gt_cam_file).numpy()
T1 = gt_pose[0]
T2 = gt_pose[1]
R21_gt = T1[:3,:3] @ np.linalg.inv(T2[:3,:3])
t21_gt = (T1@np.linalg.inv(T2))[:3,3]
R1_gt = R1_est
t1_gt = t1_est
R2_gt = R21_gt.T @ R1_gt
t2_gt = t2_est#(R21_gt.T @ (t1_gt - t21_gt)[:,None])[:,0]

# ignore translation estimates as they are unreliable
t1_est = t2_est = t1_gt = t2_gt = np.array([0.,0.,1.])

# normalize scale of pcd
est_pcd /= 10 * np.max(np.linalg.norm(est_pcd, axis=1))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(vertical_axis='y', elev=40, azim=-10)
#ax.set_box_aspect(aspect=(1, 1, 1))

ax.scatter(0, 0, 0, c='k', marker='s', label='First View')
if plot_gt:
    ax.scatter(0, 0, 0, c='g', marker='s', label='Ours')
    ax.scatter(0, 0, 0, c='b', marker='s', label='GT')
else:
    ax.scatter(0, 0, 0, c='g', marker='s', label='Second View')
if not (colmap_cam_file is None):
    ax.scatter(0, 0, 0, c='orange', marker='s', label='COLMAP (11 Views)')
if not (samurai_cam_file is None):
    ax.scatter(0, 0, 0, c='r', marker='s', label='SAMURAI')
ax.scatter(est_pcd[:,0], est_pcd[:,1], est_pcd[:,2], c='g', marker='o')
#for pt in est_pcd:
#    ax.scatter(pt[0], pt[1], pt[2], c='k', marker='o')
#ax.scatter(1, 0, 0, c='r', marker='o')
#ax.scatter(0, 1, 0, c='g', marker='o')

print('GT Euler [deg]:')
print([np.degrees(v) for v in matrix_to_euler(R21_gt)])
print('Ours Euler [deg]:')
print([np.degrees(v) for v in matrix_to_euler(R1_est @ R2_est.T)])
print('Ours Pose Error:', np.degrees(compute_pose_error(R2_est, R2_gt)), 'degrees')

#ax.set_xlim(-0.125, 1)
#ax.set_ylim(-0.125, 1)
#ax.set_zlim(-0.125, 1)

cplt.plotCamera(ax, R1_est.T, -R1_est.T @ t1_est[:,None], color='k',scale=0.1)
#cplt.plotCamera(ax, R1_gt, t1_gt[:,None], 'b',0.01)


cplt.plotCamera(ax, R2_est.T, -R2_est.T @ t2_est[:,None], color='g',scale=0.1)
if plot_gt:
    cplt.plotCamera(ax, R2_gt.T, -R2_gt.T @ t2_gt[:,None], color='b',scale=0.1)
if not (colmap_cam_file is None):
    colmap_pose = np.load(colmap_cam_file)
    R1_colmap = colmap_pose['R1']
    t1_colmap = colmap_pose['t1']
    T1_colmap = np.concatenate([R1_colmap, t1_colmap[:,None]], axis=1)
    R2_colmap = colmap_pose['R2']
    t2_colmap = colmap_pose['t2']
    T2_colmap = np.concatenate([R2_colmap, t2_colmap[:,None]], axis=1)
    R21_colmap = T1_colmap[:3,:3] @ np.linalg.inv(T2_colmap[:3,:3])
    #t21_colmap = (T1_colmap@np.linalg.inv(T2_colmap))[:3,3]
    R1_colmap = R1_est
    t1_colmap = t1_est
    R2_colmap = R21_colmap.T @ R1_colmap
    t2_colmap = t2_est#(R21_gt.T @ (t1_gt - t21_gt)[:,None])[:,0]

    print('COLMAP Pose Error:', np.degrees(compute_pose_error(R2_colmap, R2_gt)), 'degrees')
    cplt.plotCamera(ax, R2_colmap.T, -R2_colmap.T @ t2_colmap[:,None], 'orange',0.1)

if not (samurai_cam_file is None):
    samurai_pose = np.load(samurai_cam_file)
    T1_samurai = np.linalg.inv(samurai_pose[1])
    T2_samurai = np.linalg.inv(samurai_pose[2])
    R21_samurai = T1_samurai[:3,:3] @ np.linalg.inv(T2_samurai[:3,:3])
    #t21_colmap = (T1_colmap@np.linalg.inv(T2_colmap))[:3,3]
    R1_samurai = R1_est
    t1_samurai = t1_est
    R2_samurai = R21_samurai.T @ R1_samurai
    t2_samurai = t2_est#(R21_gt.T @ (t1_gt - t21_gt)[:,None])[:,0]

    print('SAMURAI Pose Error:', np.degrees(compute_pose_error(R2_samurai, R2_gt)), 'degrees')
    cplt.plotCamera(ax, R2_samurai.T, -R2_samurai.T @ t2_samurai[:,None], 'red',0.1)

ax.legend()

cplt.axisEqual3D(ax)

plt.savefig(work_dir+f'/{corrs_folder_name}/camera_result.pdf')
plt.show()





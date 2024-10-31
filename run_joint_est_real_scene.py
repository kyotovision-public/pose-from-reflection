import subprocess
import argparse
import numpy as np
from core.sfm_utils import matrix_to_euler

parser = argparse.ArgumentParser()
parser.add_argument('object_id')
parser.add_argument('--wo-data-aug', action='store_true')
parser.add_argument('--wo-gbr', action='store_true')
parser.add_argument('--wo-rm', action='store_true')
parser.add_argument('--static-camera', action='store_true')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--depth-anything', action='store_true')
args = parser.parse_args()

object_id=args.object_id
wo_data_aug = args.wo_data_aug
wo_gbr = args.wo_gbr
wo_rm = args.wo_rm
static_camera=args.static_camera

object_dir=str(object_id).zfill(3)
exp_name='test_shape_from_pose_real_scene'
corrs_dir='corrs'
if wo_data_aug:
    exp_name+='_wo_da'
    corrs_dir+='_wo_da'
if wo_gbr:
    exp_name+='_wo_gbr'
    corrs_dir+='_wo_gbr'
if wo_rm:
    exp_name+='_wo_rm'
    corrs_dir+='_wo_rm'

if args.depth_anything:
    exp_name += '_da'


src_cam_file = './run/image_matching_lightglue_ortho/'+str(object_id).zfill(2)+'/est_pose_0_1.npz'

if True:
    # SfM
    subprocess.run([
        'python', 'inference/test_image_matching_lightglue_ortho.py', str(object_id)
    ]+(['--gpu', str(args.gpu)] if not (args.gpu is None) else []))

    # DeepShaRM
    subprocess.run([
        'python', 'inference/test_joint_opt.py', 
        '--exp-name', exp_name+'_1', 
        '--config', './confs/test_joint_opt_real_scene.json',
        '--cam-file', src_cam_file,
        str(object_id)
    ]+(['--gpu', str(args.gpu)] if not (args.gpu is None) else [])+(['--depth-anything'] if args.depth_anything else []))

# 
subprocess.run([
    'python', 
    'inference/test_rmap_matching.py',
    '--result-dir', './run/'+exp_name+'_1/'+object_dir,
]+(['--gpu', str(args.gpu)] if not (args.gpu is None) else []))

src_cam_data = np.load(src_cam_file)

np.savez(
    './run/'+exp_name+'_1/'+object_dir+'/corrs/corrs_in_0_1_nm.npz',
    normal1=np.array([0.,0.,1])[None][:0],
    normal2=np.array([0.,0.,1])[None][:0],
    coord1=src_cam_data['coord1'],
    coord2=src_cam_data['coord2'],
    G21_tilde=src_cam_data['R21'],
    phi=src_cam_data['phi'],
    eta=src_cam_data['eta'],
    theta=src_cam_data['theta'],
)

subprocess.run([
    'python', 
    'inference/test_ransac_rm.py',
    '--result-dir', './run/'+exp_name+'_1/'+object_dir,
])


for i in range(2,4):
    subprocess.run([
        'python', 'inference/test_joint_opt.py', 
        '--exp-name', exp_name+'_'+str(i), 
        '--config', './confs/test_joint_opt_real_scene.json',
        '--cam-file', './run/'+exp_name+'_'+str(i-1)+'/'+object_dir+'/'+corrs_dir+'/est_pose_0_1.npz',
        '--rmap-file', './run/'+exp_name+'_'+str(i-1)+'/'+object_dir+'/'+corrs_dir+'/est_rmap_final_undistorted.pt',
        '--normal-file', './run/'+exp_name+'_'+str(i-1)+'/'+object_dir+'/'+corrs_dir+'/est_normal_sfs_final_undistorted.pt',
        #'--surf-pcd-file', './run/'+exp_name+'_'+str(i-1)+'/'+object_dir+'/'+corrs_dir+'/surf_points.ply',
        str(object_id)
    ]+(['--gpu', str(args.gpu)] if not (args.gpu is None) else [])+(['--depth-anything'] if args.depth_anything else []))

    subprocess.run([
        'python', 
        'inference/test_rmap_matching.py',
        '--result-dir', './run/'+exp_name+'_'+str(i)+'/'+object_dir
    ]+(['--gpu', str(args.gpu)] if not (args.gpu is None) else []))

    src_cam_data = np.load(src_cam_file)
    cam_data = np.load('./run/'+exp_name+'_'+str(i-1)+'/'+object_dir+'/'+corrs_dir+'/est_pose_0_1.npz')
    phi,eta,theta = matrix_to_euler(cam_data['R21'])
    np.savez(
        './run/'+exp_name+'_'+str(i)+'/'+object_dir+'/corrs/corrs_in_0_1_nm.npz',
        normal1=np.array([0.,0.,1])[None][:0],
        normal2=np.array([0.,0.,1])[None][:0],
        coord1=src_cam_data['coord1'],
        coord2=src_cam_data['coord2'],
        G21_tilde=cam_data['R21'],
        phi=phi,
        eta=eta,
        theta=theta,
    )

    subprocess.run([
        'python', 
        'inference/test_ransac_rm.py',
        '--result-dir', './run/'+exp_name+'_'+str(i)+'/'+object_dir
    ])

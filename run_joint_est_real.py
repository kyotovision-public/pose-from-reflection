import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('object_id')
parser.add_argument('--wo-data-aug', action='store_true')
parser.add_argument('--wo-gbr', action='store_true')
parser.add_argument('--wo-rm', action='store_true')
parser.add_argument('--static-camera', action='store_true')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--depth-anything', action='store_true')
parser.add_argument('--prefiltering', action='store_true')
parser.add_argument('--normal-reg', action='store_true')
args = parser.parse_args()

object_id=args.object_id
wo_data_aug = args.wo_data_aug
wo_gbr = args.wo_gbr
wo_rm = args.wo_rm
static_camera=args.static_camera
prefiltering = args.prefiltering
use_nreg = args.normal_reg

object_dir=str(object_id).zfill(3)
exp_name='test_shape_from_pose_real'
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
if prefiltering:
    exp_name+='_prefiltering'
    corrs_dir+='_prefiltering'
if use_nreg:
    exp_name+='_nreg'
    corrs_dir+='_nreg'

src_result_dir = './run/test_depth_grid_real'
if args.depth_anything:
    src_result_dir += '_da'
    exp_name += '_da'

# single-view estimation
subprocess.run([
    'python', 'inference/test_depth_grid.py', 
    '--config', './confs/test_depth_grid_real.json',
    str(object_id)
]+(['--gpu', str(args.gpu)] if not (args.gpu is None) else [])+(['--depth-anything'] if args.depth_anything else []))

# initial camera pose recovery
subprocess.run([
    'python',  'inference/est_cam_pose.py', 
    '--result-dir', src_result_dir+'/'+object_dir
]+['--wo-data-aug' for _ in range(1) if wo_data_aug]+['--wo-gbr' for _ in range(1) if wo_gbr]+['--wo-rm' for _ in range(1) if wo_rm]+['--prefiltering' for _ in range(1) if prefiltering]+['--normal-reg' for _ in range(1) if use_nreg]+['--static-camera' for _ in range(1) if static_camera]+(['--gpu', str(args.gpu)] if not (args.gpu is None) else []))

# initial multi-view consolidation
subprocess.run([
    'python', 'inference/test_joint_opt.py', 
    '--exp-name', exp_name+'_1', 
    '--config', './confs/test_joint_opt_real.json',
    '--cam-file', src_result_dir+'/'+object_dir+'/'+corrs_dir+'/est_pose_0_1.npz',
    '--rmap-file', src_result_dir+'/'+object_dir+'/'+corrs_dir+'/est_rmap_final_undistorted.pt',
    '--normal-file', src_result_dir+'/'+object_dir+'/'+corrs_dir+'/est_normal_sfs_final_undistorted.pt',
    '--surf-pcd-file', src_result_dir+'/'+object_dir+'/'+corrs_dir+'/surf_points.ply',
    str(object_id)
]+(['--gpu', str(args.gpu)] if not (args.gpu is None) else [])+(['--depth-anything'] if args.depth_anything else []))

# 2nd camera pose recovery
corrs_dir+='_sym'
subprocess.run([
    'python',  'inference/est_cam_pose.py', 
    '--result-dir', './run/'+exp_name+'_1/'+object_dir,
    '--sym'
]+['--wo-data-aug' for _ in range(1) if wo_data_aug]+['--wo-gbr' for _ in range(1) if wo_gbr]+['--wo-rm' for _ in range(1) if wo_rm]+['--prefiltering' for _ in range(1) if prefiltering]+['--normal-reg' for _ in range(1) if use_nreg]+['--static-camera' for _ in range(1) if static_camera]+(['--gpu', str(args.gpu)] if not (args.gpu is None) else []))

# alternating estimation
for i in range(2,4):
    subprocess.run([
        'python', 'inference/test_joint_opt.py', 
        '--exp-name', exp_name+'_'+str(i), 
        '--config', './confs/test_joint_opt_real.json',
        '--cam-file', './run/'+exp_name+'_'+str(i-1)+'/'+object_dir+'/'+corrs_dir+'/est_pose_0_1.npz',
        '--rmap-file', './run/'+exp_name+'_'+str(i-1)+'/'+object_dir+'/'+corrs_dir+'/est_rmap_final_undistorted.pt',
        '--normal-file', './run/'+exp_name+'_'+str(i-1)+'/'+object_dir+'/'+corrs_dir+'/est_normal_sfs_final_undistorted.pt',
        '--surf-pcd-file', './run/'+exp_name+'_'+str(i-1)+'/'+object_dir+'/'+corrs_dir+'/surf_points.ply',
        str(object_id)
    ]+(['--gpu', str(args.gpu)] if not (args.gpu is None) else [])+(['--depth-anything'] if args.depth_anything else []))

    subprocess.run([
        'python',  'inference/est_cam_pose.py', 
        '--result-dir', './run/'+exp_name+'_'+str(i)+'/'+object_dir,
        '--sym'
    ]+['--wo-data-aug' for _ in range(1) if wo_data_aug]+['--wo-gbr' for _ in range(1) if wo_gbr]+['--wo-rm' for _ in range(1) if wo_rm]+['--prefiltering' for _ in range(1) if prefiltering]+['--normal-reg' for _ in range(1) if use_nreg]+['--static-camera' for _ in range(1) if static_camera]+(['--gpu', str(args.gpu)] if not (args.gpu is None) else []))
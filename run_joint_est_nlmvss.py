import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('object_id')
parser.add_argument('--wo-data-aug', action='store_true')
parser.add_argument('--wo-gbr', action='store_true')
parser.add_argument('--wo-rm', action='store_true')
parser.add_argument('--gpu', type=int, default=None)
args = parser.parse_args()

object_id=args.object_id
wo_data_aug = args.wo_data_aug
wo_gbr = args.wo_gbr
wo_rm = args.wo_rm

object_dir=str(object_id).zfill(3)
exp_name='test_shape_from_pose_nlmvss'
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

# single-view estimation
subprocess.run([
    'python',  'inference/test_depth_grid.py', 
    '--config', './confs/test_depth_grid_nlmvss.json',
    str(object_id)
]+(['--gpu', str(args.gpu)] if not (args.gpu is None) else []))

# initial camera pose recovery
subprocess.run([
    'python',  'inference/est_cam_pose.py', 
    '--result-dir', './run/test_depth_grid_nlmvss/'+object_dir
]+['--wo-data-aug' for _ in range(1) if wo_data_aug]+['--wo-gbr' for _ in range(1) if wo_gbr]+['--wo-rm' for _ in range(1) if wo_rm]+(['--gpu', str(args.gpu)] if not (args.gpu is None) else []))

# initial multi-view consolidation
subprocess.run([
    'python', 'inference/test_joint_opt.py', 
    '--exp-name', exp_name+'_1', 
    '--config', './confs/test_joint_opt_nlmvss.json',
    '--cam-file', './run/test_depth_grid_nlmvss/'+object_dir+'/'+corrs_dir+'/est_pose_0_1.npz',
    '--rmap-file', './run/test_depth_grid_nlmvss/'+object_dir+'/'+corrs_dir+'/est_rmap_final_undistorted.pt',
    '--normal-file', './run/test_depth_grid_nlmvss/'+object_dir+'/'+corrs_dir+'/est_normal_sfs_final_undistorted.pt',
    '--surf-pcd-file', './run/test_depth_grid_nlmvss/'+object_dir+'/'+corrs_dir+'/surf_points.ply',
    str(object_id)
]+(['--gpu', str(args.gpu)] if not (args.gpu is None) else []))

# 2nd camera pose recovery
corrs_dir+='_sym'
subprocess.run([
    'python',  'inference/est_cam_pose.py', 
    '--result-dir', './run/'+exp_name+'_1/'+object_dir,
    '--sym'
]+['--wo-data-aug' for _ in range(1) if wo_data_aug]+['--wo-gbr' for _ in range(1) if wo_gbr]+['--wo-rm' for _ in range(1) if wo_rm]+(['--gpu', str(args.gpu)] if not (args.gpu is None) else []))

# alternating estimation
for i in range(2,4):
    subprocess.run([
        'python', 'inference/test_joint_opt.py', 
        '--exp-name', exp_name+'_'+str(i), 
        '--config', './confs/test_joint_opt_nlmvss.json',
        '--cam-file', './run/'+exp_name+'_'+str(i-1)+'/'+object_dir+'/'+corrs_dir+'/est_pose_0_1.npz',
        '--rmap-file', './run/'+exp_name+'_'+str(i-1)+'/'+object_dir+'/'+corrs_dir+'/est_rmap_final_undistorted.pt',
        '--normal-file', './run/'+exp_name+'_'+str(i-1)+'/'+object_dir+'/'+corrs_dir+'/est_normal_sfs_final_undistorted.pt',
        '--surf-pcd-file', './run/'+exp_name+'_'+str(i-1)+'/'+object_dir+'/'+corrs_dir+'/surf_points.ply',
        str(object_id)
    ]+(['--gpu', str(args.gpu)] if not (args.gpu is None) else []))

    subprocess.run([
        'python',  'inference/est_cam_pose.py', 
        '--result-dir', './run/'+exp_name+'_'+str(i)+'/'+object_dir,
        '--sym'
    ]+['--wo-data-aug' for _ in range(1) if wo_data_aug]+['--wo-gbr' for _ in range(1) if wo_gbr]+['--wo-rm' for _ in range(1) if wo_rm]+(['--gpu', str(args.gpu)] if not (args.gpu is None) else []))
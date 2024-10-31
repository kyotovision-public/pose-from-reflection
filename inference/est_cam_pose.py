import os
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import subprocess
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--result-dir', '-o', type=str, default='./run/test_shape_from_pose_1/079')
parser.add_argument('--wo-data-aug', action='store_true')
parser.add_argument('--wo-gbr', action='store_true')
parser.add_argument('--wo-rm', action='store_true')
parser.add_argument('--sym', action='store_true')
parser.add_argument('--static-camera', action='store_true')
parser.add_argument('--prefiltering', action='store_true')
parser.add_argument('--normal-reg', action='store_true')
parser.add_argument('--gpu', type=int, default=None)
args = parser.parse_args()

result_dir=args.result_dir
wo_data_aug = args.wo_data_aug
wo_gbr=args.wo_gbr
wo_rm=args.wo_rm
sym=args.sym
static_camera=args.static_camera
prefiltering = args.prefiltering
use_nreg = args.normal_reg

num_views = len(torch.load(result_dir+'/est_normal_final.pt'))
view_pairs = []#[(2,3), (0,1), ]
for i in range(num_views):
    for j in range(num_views):
        if i < j:
            view_pairs.append((i,j))


for view_pair in view_pairs:
    flags = [
        '--result-dir', result_dir,
        '-v1', str(sorted(view_pair)[0]),
        '-v2', str(sorted(view_pair)[1]),        
    ]
    if wo_data_aug:
        flags = flags + ['--wo-data-aug']
    if wo_gbr:
        flags = flags + ['--wo-gbr']
    if wo_rm:
        flags = flags + ['--wo-rm']
    if sym:
        flags = flags + ['--sym']
    if prefiltering:
        flags = flags + ['--prefiltering']
    if use_nreg:
        flags = flags + ['--normal-reg']
    if not (args.gpu is None):
        flags = flags + ['--gpu', str(args.gpu)]

    if wo_gbr:
        out_dir=result_dir+'/corrs_wo_gbr'
        if wo_rm:
            out_dir+='_wo_rm'
        os.makedirs(out_dir, exist_ok=True)
        src_files = [
            result_dir+'/corrs/corrs_'+str(sorted(view_pair)[0])+'_'+str(sorted(view_pair)[1])+'_nm.npz',
            result_dir+'/corrs/matches_nm_'+str(sorted(view_pair)[0])+'_'+str(sorted(view_pair)[1])+'.png',
        ]
        if all([os.path.exists(f) for f in src_files]):
            for f in src_files:
                fname = f.split('/')[-1]
                subprocess.run([
                    'cp', f, out_dir+'/'+fname
                ])
        else:
            subprocess.run([
                'python', 
                f'{project_dir}/inference/test_normal_map_matching.py',
            ]+flags)
    else:
        subprocess.run([
            'python', 
            f'{project_dir}/inference/test_normal_map_matching.py',
        ]+flags)

    if wo_gbr:
        src_files = [
            result_dir+'/corrs/corrs_'+str(sorted(view_pair)[0])+'_'+str(sorted(view_pair)[1])+'_rm.npz',
            result_dir+'/corrs/matches_rm_'+str(sorted(view_pair)[0])+'_'+str(sorted(view_pair)[1])+'.png',
        ]
        if all([os.path.exists(f) for f in src_files]):
            for f in src_files:
                fname = f.split('/')[-1]
                subprocess.run([
                    'cp', f, out_dir+'/'+fname
                ])
        else:
            subprocess.run([
                'python', 
                f'{project_dir}/inference/test_rmap_matching.py',
            ]+flags)
    else:
        subprocess.run([
            'python', 
            f'{project_dir}/inference/test_rmap_matching.py',
        ]+flags)

if wo_gbr:
    proc_list = []
    for view_pair in view_pairs:
        flags = [
            '--result-dir', result_dir,
            '-v1', str(sorted(view_pair)[0]),
            '-v2', str(sorted(view_pair)[1]),        
        ]
        if wo_rm:
            flags = flags + ['--wo-rm']
        proc = subprocess.Popen([
            'python', 
            f'{project_dir}/inference/test_ransac_wo_gbr.py',
        ]+flags)
        proc_list.append(proc)
        if len(proc_list) == 5:
            for subproc in proc_list:
                subproc.wait()
            proc_list = []

    for subproc in proc_list:
        subproc.wait()
    proc_list = []
    exit()

# ransac with conventional correspondences
proc_list = []
for view_pair in view_pairs:
    flags = [
        '--result-dir', result_dir,
        '-v1', str(sorted(view_pair)[0]),
        '-v2', str(sorted(view_pair)[1]),        
    ]
    if wo_data_aug:
        flags = flags + ['--wo-data-aug']
    if wo_gbr:
        flags = flags + ['--wo-gbr']
    if prefiltering:
        flags = flags + ['--prefiltering']
    if use_nreg:
        flags = flags + ['--normal-reg']


    script_file = f'{project_dir}/inference/test_ransac_nm.py'
    if sym:
        script_file = f'{project_dir}/inference/test_ransac_nm_sym.py'
    proc = subprocess.Popen([
        'python', 
        script_file,
    ]+flags)
    proc_list.append(proc)
    if len(proc_list) == 5:
        for subproc in proc_list:
            subproc.wait()
        proc_list = []

for subproc in proc_list:
    subproc.wait()
proc_list = []

# estimation with reflection correspondences
for view_pair in view_pairs:
    flags = [
        '--result-dir', result_dir,
        '-v1', str(sorted(view_pair)[0]),
        '-v2', str(sorted(view_pair)[1]),        
    ]
    if wo_data_aug:
        flags = flags + ['--wo-data-aug']
    if wo_gbr:
        flags = flags + ['--wo-gbr']
    if prefiltering:
        flags = flags + ['--prefiltering']
    if use_nreg:
        flags = flags + ['--normal-reg']
    if sym:
        flags = flags + ['--sym']
    if static_camera:
        flags = flags + ['--static-camera']

    proc = subprocess.Popen([
        'python', 
        f'{project_dir}/inference/test_ransac_rm.py',
    ]+flags)
    proc_list.append(proc)
    if len(proc_list) == 2:
        for subproc in proc_list:
            subproc.wait()
        proc_list = []

for subproc in proc_list:
    subproc.wait()
proc_list = []
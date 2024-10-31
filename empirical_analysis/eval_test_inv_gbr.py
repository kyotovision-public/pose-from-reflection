import numpy as np
import cv2
from tqdm import tqdm

import argparse
from glob import glob
import os

np.random.seed(10)

parser = argparse.ArgumentParser()
parser.add_argument('--method-name', type=str, default='test_inv_gbr')
args = parser.parse_args()

data_dir = './data/synthetic_correspondences'
resuld_dirs = sorted(glob('./run/'+args.method_name+'/*_*'))
out_dir='./run/eval_test_inv_gbr/'+args.method_name
os.makedirs(out_dir, exist_ok=True)

force_write = False

# plot results w.r.t. to the number of RM correspondences
for num_corrs_nm in range(101):
    # check availablity
    for num_corrs_rm in range(101):
        result_dir = './run/'+args.method_name+'/'+str(num_corrs_nm)+'_'+str(num_corrs_rm)
        if not (result_dir in resuld_dirs):
            continue
        result_files = sorted(glob(result_dir+'/*.npz'))
        if len(result_files) != 10000:
            continue

        # load results
        print('Number of Correspondences (NM,RM):', num_corrs_nm, num_corrs_rm)        

        out_file = out_dir+'/'+str(num_corrs_nm)+'_'+str(num_corrs_rm)+'.npz'
        if os.path.exists(out_file) and (not force_write):
            print('Result aldeady exists.')
            continue

        list_pose_error_deg = []
        list_compound_mat_error = []
        bar = tqdm(result_files)
        for idx_data, result_file in enumerate(bar):
            result = np.load(result_file)
            R21_est = result['R21_est']
            l1_est = result['l1_est']
            m1_est = result['m1_est']
            n1_est = result['n1_est']
            l2_est = result['l2_est']
            m2_est = result['m2_est']
            n2_est = result['n2_est']
            M_est = result['M_est']


            gt_data = np.load(data_dir+'/'+str(idx_data).zfill(5)+'.npz')
            R21_gt = gt_data['R21_gt']
            l1_gt = gt_data['l1_gt']
            m1_gt = gt_data['m1_gt']
            n1_gt = gt_data['n1_gt']
            l2_gt = gt_data['l2_gt']
            m2_gt = gt_data['m2_gt']
            n2_gt = gt_data['n2_gt']

            def get_gbr_matrix(l,m,n):
                return np.array([
                    [1., 0., 0.],
                    [0., 1., 0.],
                    [m, n, l]
                ])
            G1_gt = get_gbr_matrix(l1_gt, m1_gt, n1_gt)
            G2_gt = get_gbr_matrix(l2_gt, m2_gt, n2_gt)
            M_gt =  np.linalg.inv(G1_gt).T @ R21_gt @ G2_gt.T

            def compute_pose_error(R_est, R_gt):
                return np.arccos(np.clip(0.5 * (np.trace(R_est @ R_gt.T) - 1.), -1., 1.))
            
            pose_error_deg = np.degrees(compute_pose_error(R21_est, R21_gt))
            list_pose_error_deg.append(pose_error_deg)        

            compound_mat_error = np.sqrt(np.sum((M_est - M_gt)**2))
            list_compound_mat_error.append(compound_mat_error)

        result_dict = {
            'num_corrs_nm': num_corrs_nm,
            'num_corrs_rm': num_corrs_rm,
            'mean_pose_error_deg': np.mean(list_pose_error_deg),
            'median_pose_error_deg': np.median(list_pose_error_deg),
            'worst_pose_error_deg': np.max(list_pose_error_deg),
            'failure_rate_0.1deg': np.sum(np.array(list_pose_error_deg) > 0.1) / len(list_pose_error_deg),
            'failure_rate_1deg': np.sum(np.array(list_pose_error_deg) > 1) / len(list_pose_error_deg),
            'failure_rate_10deg': np.sum(np.array(list_pose_error_deg) > 10) / len(list_pose_error_deg),
            'mean_compound_mat_error': np.mean(list_compound_mat_error),
            'median_compound_mat_error': np.median(list_compound_mat_error),
            'max_compound_mat_error': np.max(list_compound_mat_error),
            'failure_rate_compound_0001': np.sum(np.array(list_compound_mat_error) > 1e-3) / len(list_compound_mat_error),
            'failure_rate_compound_001': np.sum(np.array(list_compound_mat_error) > 1e-2) / len(list_compound_mat_error),
            'failure_rate_compound_01': np.sum(np.array(list_compound_mat_error) > 1e-1) / len(list_compound_mat_error),
        }
        np.savez(out_file, **result_dict)

        print('Mean Pose Error:', result_dict['mean_pose_error_deg'], 'deg')
        print('Median Pose Error:', result_dict['median_pose_error_deg'], 'deg')
        print('Worst Pose Error:', result_dict['worst_pose_error_deg'], 'deg')
        print('Failure Rate(>0.1 deg):', 100 * result_dict['failure_rate_0.1deg'], '%')
        print('Failure Rate(>  1 deg):', 100 * result_dict['failure_rate_1deg'], '%')
        print('Failure Rate(> 10 deg):', 100 * result_dict['failure_rate_10deg'], '%')
        print('Mean Compound Matrix Error:', result_dict['mean_compound_mat_error'])
        print('Median Compound Matrix Error:', result_dict['median_compound_mat_error'])
        print('Max Compound Matrix Error:', result_dict['max_compound_mat_error'])
        print('Failure Rate(compound, >1e-3):', 100 * result_dict['failure_rate_compound_0001'], '%')
        print('Failure Rate(compound, >1e-2):', 100 * result_dict['failure_rate_compound_001'], '%')
        print('Failure Rate(compound, >1e-1):', 100 * result_dict['failure_rate_compound_01'], '%')


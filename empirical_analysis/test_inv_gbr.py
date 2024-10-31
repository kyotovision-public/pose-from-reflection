import os
import sys
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as ScipyRotation

import argparse
from glob import glob
import os

from core.sfm_utils import *

np.random.seed(10)

parser = argparse.ArgumentParser()
parser.add_argument('--num-corrs-rm', '-nr', type=int, default=100)
parser.add_argument('--num-corrs-nm', '-nn', type=int, default=100)
parser.add_argument('--num-corrs-pix', '-np', type=int, default=None)
parser.add_argument('--data-dir', type=str, default='./data/synthetic_correspondences')
parser.add_argument('--out-dir', '-o', type=str, default='./run/test_inv_gbr')
args = parser.parse_args()

num_corrs_rm = args.num_corrs_rm
num_corrs_nm = args.num_corrs_nm
num_corrs_pix = args.num_corrs_pix
if num_corrs_pix is None:
    num_corrs_pix = num_corrs_nm

data_dir = args.data_dir
out_dir = args.out_dir+'/'+str(num_corrs_nm)+'_'+str(num_corrs_rm)
if num_corrs_nm != num_corrs_pix:
    out_dir = args.out_dir+'/'+str(num_corrs_nm)+'_'+str(num_corrs_rm)+'_'+str(num_corrs_pix)
os.makedirs(out_dir, exist_ok=True)

min_view_distance_deg = 15
pose_tor_deg = 1
x_tor = 1e-4
ldiv_tor = 1e-4
f_tor = 1e-6
M_tor = 1e-6

assert_x = True
assert_ldiv = True
assert_f = True
assert_M = True

exit_if_asserted = False

print('Num of Normal Corrs:    ', num_corrs_nm)
print('Num of Reflection Corrs:', num_corrs_rm)
print('Num of Pixel Corrs:     ', num_corrs_pix)

num_failed_f = 0
num_failed_M = 0
num_failed = 0
pose_errors = []
M_errors = []
data_files = sorted(glob(data_dir+'/?????.npz'))
bar = tqdm(data_files)
for idx_data, data_file in enumerate(bar):
    # data loading
    data = np.load(data_file)
    normal1_rm = data['normal1_rm'][:num_corrs_rm]
    normal2_rm = data['normal2_rm'][:num_corrs_rm]
    normal1_nm = data['normal1_nm'][:num_corrs_nm]
    normal2_nm = data['normal2_nm'][:num_corrs_nm]
    pixel1 = data['pixel1'][:num_corrs_pix]
    pixel2 = data['pixel2'][:num_corrs_pix]

    #correct = False
    #phi_gt, eta_gt, theta_gt = matrix_to_euler(data['R21_gt'])
    #for phi_est, theta_est in solve_phi_theta(pixel1, pixel2):
    #    if (np.arccos(np.cos(phi_est - phi_gt)) < 1e-3) and (np.arccos(np.cos(theta_est - theta_gt)) < 1e-3):
    #        correct = True
    #if not correct:
    #    break

    result_dict = solve_pose({
        'pixel1': pixel1,
        'pixel2': pixel2,
        'normal1_nm': normal1_nm,
        'normal2_nm': normal2_nm,
        'normal1_rm': normal1_rm,
        'normal2_rm': normal2_rm,
    },
    use_two_steps=False,
    use_multi_initial_params=False,
    wo_gbr=False,
    )

    R21_est = result_dict['R21_est']
    G1_est = result_dict['G1_est']
    G2_est = result_dict['G2_est']
    M_est = result_dict['G21_est']
    l1_est = result_dict['l1_est']
    m1_est = result_dict['m1_est']
    n1_est = result_dict['n1_est']
    l2_est = result_dict['l2_est']
    m2_est = result_dict['m2_est']
    n2_est = result_dict['n2_est']

    result = result_dict['optimization_result']

    #print(t21_est0)
    #print(t21_est_ofs)
    #print(data['t21_gt'])
    #print((data['t21_gt'] - t21_est0) / t21_est_ofs)
    #exit()

    # save results
    result_dict = {
        # settings
        'method': 'scipy_least_squares',
        'num_corrs_rm': num_corrs_rm,
        'num_corrs_nm': num_corrs_nm,
        # results
        'R21_est': R21_est,
        'G1_est': G1_est,
        'G2_est': G2_est,
        'M_est': M_est,
        'l1_est': l1_est,
        'm1_est': m1_est,
        'n1_est': n1_est,
        'l2_est': l2_est,
        'm2_est': m2_est,
        'n2_est': n2_est,
    }
    np.savez(
        out_dir+'/'+str(idx_data).zfill(5)+'.npz',
        **result_dict
    )

    # comparison with gt
    R21_gt = data['R21_gt']
    l1_gt = data['l1_gt']
    m1_gt = data['m1_gt']
    n1_gt = data['n1_gt']
    l2_gt = data['l2_gt']
    m2_gt = data['m2_gt']
    n2_gt = data['n2_gt']

    param_gt = encode_param(l1_gt, m1_gt, n1_gt, l2_gt, m2_gt, n2_gt, R21_gt)
    assert np.all(np.abs(res_func(param_gt, pixel1, pixel2, normal1_rm, normal2_rm, normal1_nm, normal2_nm) < 1e-9))

    G1_gt = get_gbr_matrix(l1_gt, m1_gt, n1_gt)
    G2_gt = get_gbr_matrix(l2_gt, m2_gt, n2_gt)
    M_gt =  np.linalg.inv(G1_gt).T @ R21_gt @ G2_gt.T

    M_err = np.sqrt(np.sum((M_est - M_gt)**2))
    M_errors.append(
        M_err
    )

    max_residual_error = np.max(res_func(result.x, pixel1, pixel2, normal1_rm, normal2_rm, normal1_nm, normal2_nm)**2)

    pose_error = np.degrees(compute_pose_error(R21_est, R21_gt))
    pose_errors.append(
        pose_error
    )

    ftor_passed = max_residual_error < f_tor

    xtor_passed = abs(l1_est - l1_gt) < x_tor
    xtor_passed *= abs(l2_est - l2_gt) < x_tor
    xtor_passed *= abs(m1_est - m1_gt) < x_tor
    xtor_passed *= abs(m2_est - m2_gt) < x_tor
    xtor_passed *= abs(n1_est - n1_gt) < x_tor
    xtor_passed *= abs(n2_est - n2_gt) < x_tor
    xtor_passed *= pose_error < pose_tor_deg

    ldivtor_passed = abs(l1_est / l2_est - l1_gt / l2_gt) < ldiv_tor

    Mtor_passed = np.max(np.abs(M_est - M_gt)) < M_tor

    asserted = False
    asserted = asserted or (assert_x and (not xtor_passed))
    asserted = asserted or (assert_f and (not ftor_passed))
    asserted = asserted or (assert_ldiv and (not ldivtor_passed))
    asserted = asserted or (assert_M and (not Mtor_passed))

    if False:#not Mtor_passed:
        assert np.all(np.abs(R21_gt - euler_to_matrix(*matrix_to_euler(R21_gt))) < 1e-6)

        print(result)

        l1_init, m1_init, n1_init, l2_init, m2_init, n2_init = decode_param(param_init)[:6]
        phi_init, eta_init, theta_init = matrix_to_euler(decode_param(param_init)[-1])    
        phi_est, eta_est, theta_est = matrix_to_euler(R21_est)    
        phi_gt, eta_gt, theta_gt = matrix_to_euler(R21_gt)   
        print('euler_init:', phi_init, eta_init, theta_init)
        print('euler_est:', phi_est, eta_est, theta_est)
        print('euler_gt:', phi_gt, eta_gt, theta_gt)
        print('l_init:', l1_init, l2_init, l2_init / l1_init)
        print('l_gt:', l1_gt, l2_gt, l2_gt / l1_gt)
        print(euler_to_matrix(phi_gt, eta_gt, theta_gt))
        print(euler_to_matrix(-phi_gt, eta_gt, -theta_gt))

        if True:
            list_phi = np.arange(0, 2*np.pi, np.pi/50)
            list_theta = np.arange(0, 2*np.pi, np.pi/50)
            list_res_pix = []
            list_res_nm = []
            for phi_ in list_phi:
                for theta_ in list_theta:
                    R21_gt_ = euler_to_matrix(phi_, eta_gt, theta_)
                    param_gt_ = encode_param(l1_gt, m1_gt, n1_gt, l2_gt, m2_gt, n2_gt, R21_gt_)
                    list_res_pix.append(
                        np.sum(
                            res_func(
                                param_gt_, 
                                pixel1, pixel2, 
                                normal1_rm[:0], normal2_rm[:0], 
                                normal1_nm[:0], normal2_nm[:0]
                            )**2
                        ) / len(pixel1)
                    )
                    list_res_nm.append(
                        np.sum(
                            res_func(
                                param_gt_, 
                                pixel1[:0], pixel2[:0], 
                                normal1_rm[:0], normal2_rm[:0], 
                                normal1_nm, normal2_nm
                            )**2
                        ) / len(normal1_nm)
                    )
            list_res_pix = np.array(list_res_pix).reshape(len(list_phi), len(list_theta))
            list_res_nm = np.array(list_res_nm).reshape(len(list_phi), len(list_theta))
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10,10))
            plt.subplot(2,2,1)
            plt.imshow(list_res_pix)
            plt.title('Pixel Residuals')
            plt.xlabel('theta ($0\\rightarrow2\pi$)')
            plt.ylabel('phi ($2\pi\\rightarrow0$)')
            plt.subplot(2,2,2)
            plt.imshow(list_res_pix, vmin=0, vmax=0.01)
            plt.title('Pixel Residuals (vmax=0.01)')
            plt.xlabel('theta ($0\\rightarrow2\pi$)')
            plt.ylabel('phi ($2\pi\\rightarrow0$)')
            plt.subplot(2,2,3)
            plt.imshow(list_res_nm)
            plt.title('Normal Residuals')
            plt.xlabel('theta ($0\\rightarrow2\pi$)')
            plt.ylabel('phi ($2\pi\\rightarrow0$)')
            plt.subplot(2,2,4)
            plt.imshow(list_res_nm, vmax=0.1)
            plt.title('Normal Residuals (vmax=0.1)')
            plt.xlabel('theta ($0\\rightarrow2\pi$)')
            plt.ylabel('phi ($2\pi\\rightarrow0$)')

            plt.show()

    num_failed_f += 1 if not ftor_passed else 0
    num_failed_M += 1 if not Mtor_passed else 0
    num_failed += 1 if asserted else 0
    bar.set_postfix(
        worst_pose_errors=sorted(pose_errors, reverse=True)[:4],
        #worst_M_errors=sorted(M_errors, reverse=True)[:5],
        num_failed=num_failed,
        num_failed_f=num_failed_f,
        num_failed_M=num_failed_M
    )
    if asserted and exit_if_asserted:
        print(result)
        if not xtor_passed:
            print('Error: Estimated parameters are wrong.')
        if not ftor_passed:
            print('Error: Residual errors of the objective function is too large.')
        if not ldivtor_passed:
            print('Error: Estimated ldiv is wrong')

        print('residual mse:', np.mean(res_func(result.x, pixel1, pixel2, normal1_rm, normal2_rm, normal1_nm, normal2_nm)**2))
        print('max residual error:', max_residual_error)

        print('R21_est:')
        print(R21_est)
        print('R21_gt:')
        print(R21_gt)
        print('')

        print('l1_est:', l1_est)
        print('l1_gt:', l1_gt)
        print('m1_est:', m1_est)
        print('m1_gt:', m1_gt)
        print('n1_est:', n1_est)
        print('n1_gt:', n1_gt)
        print('')


        print('l2_est:', l2_est)
        print('l2_gt:', l2_gt)
        print('m2_est:', m2_est)
        print('m2_gt:', m2_gt)
        print('n2_est:', n2_est)
        print('n2_gt:', n2_gt)
        print('')

        print('ldiv_est:', l1_est / l2_est)
        print('ldiv_gt:', l1_gt / l2_gt)

        G1_gt = get_gbr_matrix(l1_gt, m1_gt, n1_gt)
        G2_gt = get_gbr_matrix(l2_gt, m2_gt, n2_gt)
        M_gt =  np.linalg.inv(G1_gt).T @ R21_gt @ G2_gt.T
        print('M_gt:')
        print(M_gt)

        G1_est = get_gbr_matrix(l1_est, m1_est, n1_est)
        G2_est = get_gbr_matrix(l2_est, m2_est, n2_est)
        M_est =  np.linalg.inv(G1_est).T @ R21_est @ G2_est.T
        print('M_est:')
        print(M_est)

        print(res_func(result.x, pixel1, pixel2, normal1_rm, normal2_rm, normal1_nm, normal2_nm))
        normal2_nm_warped = (M_est @ normal2_nm.T).T
        normal2_nm_warped = normal2_nm_warped / np.linalg.norm(normal2_nm_warped, axis=1, keepdims=True)
        #print(normal1_nm - normal2_nm_warped)

        normal1_nm_u = undistort_normal(normal1_nm, l1_est, m1_est, n1_est)
        normal2_nm_u = undistort_normal(normal2_nm, l2_est, m2_est, n2_est)

        #print(normal1_nm_u - normal2_nm_u)

        print(np.max(np.abs(normal1_nm_u[:,2])))

        print('M_est[i,j] / M_gt[i,j]:')
        print(M_est / M_gt)

        print(M_gt[:,2] / np.linalg.norm(M_gt[:,2]))
        print(M_est[:,2] / np.linalg.norm(M_est[:,2]))

        print('Test Failed')
        #print(res_func(param_gt, normal1_rm, normal2_rm, normal1_nm, normal2_nm))
        exit()

    # evaluate solutions

    pass

print('Test Passed')
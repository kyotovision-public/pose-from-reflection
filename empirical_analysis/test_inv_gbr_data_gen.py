import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as ScipyRotation

import os
import argparse

np.random.seed(10)

parser = argparse.ArgumentParser()
parser.add_argument('--num-corrs-rm', '-nr', type=int, default=200)
parser.add_argument('--num-corrs-nm', '-nn', type=int, default=200)
parser.add_argument('--data-dir', '-o', type=str, default='./data/synthetic_correspondences')

args = parser.parse_args()

num_tries = 10000
num_corrs_rm = args.num_corrs_rm
num_corrs_nm = args.num_corrs_nm
use_kabsch = False

data_dir = args.data_dir
os.makedirs(data_dir, exist_ok=True)


min_view_distance_deg = 15


print('Num of Corrs in NM:', num_corrs_nm)
print('Num of Corrs in RM:', num_corrs_rm)

bar = tqdm(range(num_tries))
for idx_data in bar:
    # sample random surface points
    surf_point1 = np.random.rand(num_corrs_nm, 3)

    # sample random translation
    t21_gt = np.random.rand(3)

    # sample random normals
    def uniform_sample_hemisphere(num_samples):
        cosTheta = np.random.rand(num_samples)
        sinTheta = np.sqrt(np.clip(1. - cosTheta**2, 0, 1))
        phi = 2. * np.pi * np.random.rand(num_samples)
        return np.stack([
            sinTheta * np.cos(phi),
            sinTheta * np.sin(phi),
            cosTheta
        ], axis=-1)
    normal1_rm_gt = uniform_sample_hemisphere(num_samples=num_corrs_rm)
    normal1_nm_gt = uniform_sample_hemisphere(num_samples=num_corrs_nm)

    # sample random relative pose
    R21_gt = ScipyRotation.random().as_matrix()
    while np.abs(R21_gt[2,2]) > np.cos(np.radians(min_view_distance_deg)):
        R21_gt = ScipyRotation.random().as_matrix()

    def light_dir(normal):
        return np.stack([
            2. * normal[:,0] * normal[:,2],
            2. * normal[:,1] * normal[:,2],
            2. * normal[:,2]**2 - 1.,
        ], axis=-1)
    
    def light2normal(light):
        n = light + np.array([0.,0.,1.])
        return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-20)

    surf_point2 = (surf_point1 - t21_gt) @ R21_gt
    normal2_rm_gt = light2normal(light_dir(normal1_rm_gt) @ R21_gt)
    normal2_nm_gt = normal1_nm_gt @ R21_gt
    assert np.all(np.abs(light_dir(normal1_rm_gt).T - R21_gt @ light_dir(normal2_rm_gt).T) < 1e-6)
    assert np.all(np.abs(normal1_nm_gt.T - R21_gt @ normal2_nm_gt.T) < 1e-6)

    # sample random GBR parameters
    def random_sample_gbr_params():
        m = 0.5 * np.random.randn()
        n = 0.5 * np.random.randn()
        l = np.exp(0.5 * np.random.randn())
        return l, m, n
    l1_gt, m1_gt, n1_gt = random_sample_gbr_params()
    l2_gt, m2_gt, n2_gt = random_sample_gbr_params()
    def undistort_normal(normal_gt, l, m, n):
        normal =  np.stack([
            normal_gt[:,0] + m * normal_gt[:,2],
            normal_gt[:,1] + n * normal_gt[:,2],
            l * normal_gt[:,2],
        ], axis=-1)
        return normal / np.linalg.norm(normal, axis=-1, keepdims=True)

    def distort_normal(normal, l, m, n):
        return undistort_normal(normal, 1./l, -m/l, -n/l)
    
    def get_gbr_matrix(l,m,n):
        return np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [m, n, l]
        ])

    normal1_rm = distort_normal(normal1_rm_gt, l1_gt, m1_gt, n1_gt)
    normal2_rm = distort_normal(normal2_rm_gt, l2_gt, m2_gt, n2_gt)

    normal1_nm = distort_normal(normal1_nm_gt, l1_gt, m1_gt, n1_gt)
    normal2_nm = distort_normal(normal2_nm_gt, l2_gt, m2_gt, n2_gt)

    assert np.all(np.abs(undistort_normal(normal1_rm, l1_gt, m1_gt, n1_gt) - normal1_rm_gt) < 1e-6)
    assert np.all(np.abs(undistort_normal(normal2_rm, l2_gt, m2_gt, n2_gt) - normal2_rm_gt) < 1e-6)
    assert np.all(np.abs(undistort_normal(normal1_nm, l1_gt, m1_gt, n1_gt) - normal1_nm_gt) < 1e-6)
    assert np.all(np.abs(undistort_normal(normal2_nm, l2_gt, m2_gt, n2_gt) - normal2_nm_gt) < 1e-6)

    # save generated data
    data_dict = {
        # correspondences
        'normal1_rm': normal1_rm,
        'normal2_rm': normal2_rm,
        'normal1_nm': normal1_nm,
        'normal2_nm': normal2_nm,
        "pixel1": surf_point1[:,:2],
        "pixel2": surf_point2[:,:2],
        # GT
        'R21_gt': R21_gt,
        't21_gt': t21_gt,
        "depth1_gt": surf_point1[:,2],
        "depth2_gt": surf_point2[:,2],
        'l1_gt': l1_gt,
        'm1_gt': m1_gt,
        'n1_gt': n1_gt,
        'l2_gt': l2_gt,
        'm2_gt': m2_gt,
        'n2_gt': n2_gt,
        'normal1_rm_gt': normal1_rm_gt,
        'normal2_rm_gt': normal2_rm_gt,
        'normal1_nm_gt': normal1_nm_gt,
        'normal2_nm_gt': normal2_nm_gt,
    }
    np.savez(
        data_dir+'/'+str(idx_data).zfill(5)+'.npz',
        **data_dict
    )

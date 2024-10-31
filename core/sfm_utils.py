import numpy as np
import cv2
import scipy

from numba import cuda
import math

# Note: We assume right-up-view coordinate system

def compute_pose_error(R_est, R_gt):
    return np.arccos(np.clip(0.5 * (np.trace(R_est @ R_gt.T) - 1.), -1., 1.))

# return a rotation matrix R21 that satisfies v1 = v2 @ R21.T
# (in a least squares sense)
def compute_relative_rot(v1, v2):
    X = v1.T
    Y = v2.T
    U,s,Vt = np.linalg.svd(Y@(X.T))
    H = np.diag([1.0, 1.0, np.linalg.det(Vt.T@U.T)])
    return Vt.T@H@U.T

# normal <-> light direction
def normal2light(normal):
    return np.stack([
        2. * normal[:,0] * normal[:,2],
        2. * normal[:,1] * normal[:,2],
        2. * normal[:,2]**2 - 1.,
    ], axis=-1)

def light2normal(light):
    n = light + np.array([0.,0.,1.])
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-20)

# Euler(ZXZ) <-> Rotation Matrix
def matrix_to_euler(R):
    eta = np.arccos(np.clip(R[2,2], -1., 1.))
    theta = np.arctan2(-R[2,0], R[2,1])
    phi = np.arctan2(R[0,2], -R[1,2])
    return phi, eta, theta

def euler_to_matrix(phi, eta, theta):
    return np.array([
        [np.cos(phi) * np.cos(theta) + np.sin(phi) * np.sin(theta) * np.cos(eta), np.cos(phi) * np.sin(theta) - np.sin(phi) * np.cos(theta) * np.cos(eta), np.sin(phi) * np.sin(eta)],
        [np.sin(phi) * np.cos(theta) - np.cos(phi) * np.sin(theta) * np.cos(eta), np.sin(phi) * np.sin(theta) + np.cos(phi) * np.cos(theta) * np.cos(eta), -np.cos(phi) * np.sin(eta)],
        [-np.sin(theta) * np.sin(eta), np.cos(theta) * np.sin(eta), np.cos(eta)]
    ])

# utils about GBR transformation
def undistort_normal(normal_gt, l, m, n):
    normal =  np.stack([
        normal_gt[...,0] + m * normal_gt[...,2],
        normal_gt[...,1] + n * normal_gt[...,2],
        l * normal_gt[...,2],
    ], axis=-1)
    return normal / np.clip(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-3, None)

def distort_normal(normal, l, m, n):
    return undistort_normal(normal, 1./l, -m/l, -n/l)

def get_gbr_matrix(l,m,n):
    return np.array([
        [1., 0., 0.],
        [0., 1., 0.],
        [m, n, l]
    ])


# utils for pixel correspondence
@cuda.jit
def solve_phi_theta_kernel1(x1, y1, x2, y2, f_grid):
    Nphi, Ntheta = f_grid.shape

    idx_phi = cuda.blockIdx.x
    idx_theta = cuda.blockIdx.y

    phi = idx_phi / Nphi * 2 * np.pi
    theta = idx_theta / Ntheta * 2 * np.pi

    s_num = 0.
    s_denom = 0.
    for i in range(len(x1)):
        t1 = (x1[i] * math.cos(phi) + y1[i] * math.sin(phi)) 
        t2 = (x2[i] * math.cos(theta) + y2[i] * math.sin(theta))
        s_num += t1 * t2
        s_denom += t2**2
    s = s_num / s_denom
    #s = 1 

    f = 0.
    for i in range(len(x1)):
        t1 = (x1[i] * math.cos(phi) + y1[i] * math.sin(phi)) 
        t2 = (x2[i] * math.cos(theta) + y2[i] * math.sin(theta))
        f += (t1 - s * t2)**2
    
    f_grid[idx_phi, idx_theta] = f / len(x1)

@cuda.jit
def solve_phi_theta_kernel2(f_grid, minima_grid):
    Nphi, Ntheta = f_grid.shape

    idx_phi = cuda.blockIdx.x
    idx_theta = cuda.blockIdx.y


    f0 = f_grid[idx_phi, idx_theta]
    is_local_minima = 1
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            if (i == 0) and (j == 0):
                continue
            f1 = f_grid[(idx_phi + i) % Nphi, (idx_theta + j) % Ntheta]
            if f1 < f0:
                is_local_minima = 0

    minima_grid[idx_phi, idx_theta] = is_local_minima

def solve_phi_theta(pixel1, pixel2, N=25, th_rate=1e-2):
    x1 = pixel1[:,0] - np.mean(pixel1[:,0])
    y1 = pixel1[:,1] - np.mean(pixel1[:,1])
    x2 = pixel2[:,0] - np.mean(pixel2[:,0])
    y2 = pixel2[:,1] - np.mean(pixel2[:,1])

    if True:
        root_angles = []
        A = np.stack([x1,y1,-x2,-y2], axis=1)
        U,s,Vt = np.linalg.svd(A)
        for i in range(len(s)-1,-1,-1):
            if (i != len(s)-1) and (s[i]**2 > 1e-6):
                break
            phi = np.arctan2(Vt[i][1], Vt[i][0])
            theta = np.arctan2(Vt[i][3], Vt[i][2])
            root_angles.append((phi, theta))
            root_angles.append((phi+np.pi, theta+np.pi))
        return root_angles

    x1_cuda = cuda.to_device(x1)
    y1_cuda = cuda.to_device(y1)
    x2_cuda = cuda.to_device(x2)
    y2_cuda = cuda.to_device(y2)

    f_grid_cuda = cuda.to_device(np.zeros((N,N)))
    minima_grid_cuda = cuda.to_device(np.zeros((N,N)))

    solve_phi_theta_kernel1[(N,N),(1,)](x1_cuda, y1_cuda, x2_cuda, y2_cuda, f_grid_cuda)
    solve_phi_theta_kernel2[(N,N),(1,)](f_grid_cuda, minima_grid_cuda)

    f_grid = f_grid_cuda.copy_to_host()
    minima_grid = minima_grid_cuda.copy_to_host()

    max_f = np.max(f_grid)
    root_indices = zip(*np.where(minima_grid > 0))
    possible_root_angles = [(i / N * 2 * np.pi, j / N * 2 * np.pi) for i,j in root_indices if f_grid[i,j] < th_rate * max_f]

    # refine
    def residual_pix(param):
        phi, theta = param
        t1 = (x1 * np.cos(phi) + y1 * np.sin(phi)) 
        t2 = (x2 * np.cos(theta) + y2 * np.sin(theta))

        return t1 - t2
    
    root_angles = []
    for possible_root_angle in possible_root_angles:
        result = scipy.optimize.least_squares(residual_pix, possible_root_angle)
        root_angles.append(result.x)

    if False:
        import matplotlib.pyplot as plt
        plt.subplot(1,2,1)
        plt.imshow(f_grid)#, vmax=0.01 * np.max(f_grid))
        plt.subplot(1,2,2)
        plt.imshow(minima_grid)
        plt.show()
    return root_angles

def encode_param(l1, m1, n1, l2, m2, n2, R21):
    logl1 = np.log(l1 - 1e-2)
    logl2 = np.log(l2 - 1e-2)
    rvec = cv2.Rodrigues(R21)[0][:,0]
    return logl1, m1, n1, logl2, m2, n2, rvec[0], rvec[1], rvec[2]

def decode_param(param):
    logl1, m1, n1, logl2, m2, n2 = param[:6]
    l1 = np.exp(logl1) + 1e-2
    l2 = np.exp(logl2) + 1e-2
    R21 = cv2.Rodrigues(param[6:9])[0]
    return l1, m1, n1, l2, m2, n2, R21

def res_func_pixel(param, pixel1, pixel2, optimize_scale=True):
    if len(pixel1) <= 1:
        return [] 
    phi, theta = param[:2]
    #R21 = decode_param(param)[-1]
    #phi, eta, theta = matrix_to_euler(R21)

    x1 = pixel1[:,0] - np.mean(pixel1[:,0])
    y1 = pixel1[:,1] - np.mean(pixel1[:,1])
    x2 = pixel2[:,0] - np.mean(pixel2[:,0])
    y2 = pixel2[:,1] - np.mean(pixel2[:,1])

    t1 = (x1 * np.cos(phi) + y1 * np.sin(phi))
    t2 = (x2 * np.cos(theta) + y2 * np.sin(theta))

    if optimize_scale:
        s = np.sum(t1 * t2) / np.sum(t2**2)
        t2 = s * t2

    res_pix = (t1 - t2)
    return res_pix

def res_func_rm_v(param, normal1, normal2):
    l1, m1, n1, l2, m2, n2, R21 = decode_param(param)

    G1 = get_gbr_matrix(l1, m1, n1)
    G2 = get_gbr_matrix(l2, m2, n2)

    GtN1 = normal1 @ G1
    GtN2 = normal2 @ G2

    v = np.array([0.,0.,1])

    if True:
        light1 = -v + 2 * normal1[:,2:3] * l1 / np.sum(GtN1**2, axis=-1, keepdims=True) * GtN1
        light2 = -v + 2 * normal2[:,2:3] * l2 / np.sum(GtN2**2, axis=-1, keepdims=True) * GtN2

        return np.concatenate([
            light1 - light2 @ R21.T,
            light2 - light1 @ R21,
        ]).reshape(-1)
    

    R21mI_inv = np.linalg.inv(R21 - (1 - 1e-6) * np.eye(3))

    a = 2 * normal2[:,2:3] * l2 / np.sum(GtN2**2, axis=-1, keepdims=True) * GtN2 @ (R21).T
    b = 2 * normal1[:,2:3] * l1 / np.sum(GtN1**2, axis=-1, keepdims=True) * GtN1

    c = 2 * normal1[:,2:3] * l1 / np.sum(GtN1**2, axis=-1, keepdims=True) * GtN1 @ (R21.T).T
    d = 2 * normal2[:,2:3] * l2 / np.sum(GtN2**2, axis=-1, keepdims=True) * GtN2

    print((a - b) @ R21mI_inv.T)

    return np.concatenate([
        (a - b) @ R21mI_inv.T - np.array([0., 0., 1.]),
        #c - d - np.array([0., 0., 1.]),
        #(a - b) - (c - d)
    ]).reshape(-1)



def res_func(
        param, 
        pixel1, pixel2, 
        normal1_rm, normal2_rm, 
        normal1_nm, normal2_nm, 
        target_eta=None, 
        weight_pix=None, 
        weight_nm=None
    ):
    l1, m1, n1, l2, m2, n2, R21 = decode_param(param)
    phi, eta, theta = matrix_to_euler(R21)

    res_pix = np.array([v / 10. for v in res_func_pixel((phi, theta), pixel1, pixel2)])
    if not (weight_pix is None):
        res_pix = weight_pix * res_pix

    #phi, eta, theta = matrix_to_euler(R21)

    #x1 = pixel1[:,0] - np.mean(pixel1[:,0])
    #y1 = pixel1[:,1] - np.mean(pixel1[:,1])
    #x2 = pixel2[:,0] - np.mean(pixel2[:,0])
    #y2 = pixel2[:,1] - np.mean(pixel2[:,1])

    #t1 = (x1 * np.cos(phi) + y1 * np.sin(phi))
    #t2 = (x2 * np.cos(theta) + y2 * np.sin(theta))
    #res_pix = (t1 - t2)
    
    if False:
        res_rm = res_func_rm_v(param, normal1_rm, normal2_rm)

    elif True:
        normal2_rm_warped = distort_normal(
            light2normal(
                normal2light(
                    undistort_normal(normal2_rm, l2, m2, n2)
                ) @ R21.T
            ),
            l1,m1,n1
        )
        normal1_rm_warped = distort_normal(
            light2normal(
                normal2light(
                    undistort_normal(normal1_rm, l1, m1, n1)
                ) @ R21
            ),
            l2,m2,n2
        )
        res_rm = np.concatenate([
            (normal1_rm - normal2_rm_warped).reshape(-1),
            (normal2_rm - normal1_rm_warped).reshape(-1)
        ])
    else:
        normal1_rm_u = undistort_normal(normal1_rm, l1, m1, n1)
        normal2_rm_u = undistort_normal(normal2_rm, l2, m2, n2)

        light1_u = normal2light(normal1_rm_u)
        light2_u = normal2light(normal2_rm_u)

        res_rm = (light1_u - light2_u @ R21.T).reshape(-1)

    if True:
        G1_est = get_gbr_matrix(l1, m1, n1)
        G2_est = get_gbr_matrix(l2, m2, n2)
        M21 =  np.linalg.inv(G1_est).T @ R21 @ G2_est.T
        normal2_nm_warped = (M21 @ normal2_nm.T).T
        normal2_nm_warped = normal2_nm_warped / np.clip(np.linalg.norm(normal2_nm_warped, axis=1, keepdims=True), 1e-3, None)
        M12 =  np.linalg.inv(G2_est).T @ R21.T @ G1_est.T
        normal1_nm_warped = (M12 @ normal1_nm.T).T
        normal1_nm_warped = normal1_nm_warped / np.clip(np.linalg.norm(normal1_nm_warped, axis=1, keepdims=True), 1e-3, None)
        res_nm = np.concatenate([
            (normal1_nm - normal2_nm_warped).reshape(-1),
            (normal2_nm - normal1_nm_warped).reshape(-1),
        ])
        if not (weight_nm is None):
            weight_nm_ = np.stack([weight_nm, weight_nm, weight_nm], axis=1).reshape(-1)
            res_nm = np.concatenate([weight_nm_, weight_nm_]) * res_nm
    else:
        normal1_nm_u = undistort_normal(normal1_nm, l1, m1, n1)
        normal2_nm_u = undistort_normal(normal2_nm, l2, m2, n2)

        res_nm = (normal1_nm_u - normal2_nm_u @ R21.T).reshape(-1)

    res_func_total = np.concatenate([
        res_pix / max(1,len(pixel1)), 
        res_rm / max(1,len(normal1_rm)), 
        res_nm / max(1,len(normal1_nm))
    ])
    #if len(normal1_rm) == 0:
    #    res_reg_eta = np.array([np.cos(eta) - np.radians(45)])
    #    res_func_total = np.concatenate([res_func_total, res_reg_eta])
    if not (target_eta is None):
        res_func_total = np.concatenate([
            res_func_total, 
            np.array([eta - target_eta])
        ])

    return res_func_total

def decompose_relative_gbr(G21_tilde, R21):
    if (G21_tilde[2,0]**2 + G21_tilde[2,1]**2) < 1e-3:
        print('Warning: Invalid G21_tilde')

    l1 = np.sqrt((R21[2,0]**2 + R21[2,1]**2) / (G21_tilde[2,0]**2 + G21_tilde[2,1]**2))
    l2 = l1 * np.linalg.det(G21_tilde)

    # G21_tilde[2,0] * m1 = (R21[0,0] - G21_tilde[0,0])
    # G21_tilde[2,1] * m1 = (R21[0,1] - G21_tilde[0,1])
    # G21_tilde[2,0] * n1 = (R21[1,0] - G21_tilde[1,0])
    # G21_tilde[2,1] * n1 = (R21[1,1] - G21_tilde[1,1])
    # -G21_tilde[2,2] * m1 + R21[0,0] * m2 + R21[0,1] * n2 = G21_tilde[0,2] - R21[0,2] * l2
    # G21_tilde[2,0] * m2 + G21_tilde[2,1] * n2 = G21_tilde[2,2] - R21[2,2] * np.linalg.det(G21_tilde)
    # x: [m1, n1, m2, n2]
    A = np.array([
        [G21_tilde[2,0], 0., 0., 0.],
        [G21_tilde[2,1], 0., 0., 0.],
        [0., G21_tilde[2,0], 0., 0.],
        [0., G21_tilde[2,1], 0., 0.],
        [-G21_tilde[2,2], 0., R21[0,0], R21[0,1]],
        [0., -G21_tilde[2,2], R21[1,0], R21[1,1]],
        [0., 0., G21_tilde[2,0], G21_tilde[2,1]],
    ])
    b = np.array([
        R21[0,0] - G21_tilde[0,0],
        R21[0,1] - G21_tilde[0,1],
        R21[1,0] - G21_tilde[1,0],
        R21[1,1] - G21_tilde[1,1],
        G21_tilde[0,2] - R21[0,2] * l2,
        G21_tilde[1,2] - R21[1,2] * l2,
        G21_tilde[2,2] - R21[2,2] * np.linalg.det(G21_tilde)
    ])
    result = np.linalg.lstsq(A,b, rcond=None)
    m1, n1, m2, n2 = result[0]
    return l1, m1, n1, l2, m2, n2

def decompose_invalid_relative_gbr(G21_tilde, R21):
    if (R21[2,0]**2 + R21[2,1]**2) < 1e-3:
        print('Warning: Invalid R21')

    G21_tilde /= np.linalg.det(G21_tilde)**(1/3)

    #l1 = np.sqrt((R21[2,0]**2 + R21[2,1]**2) / (G21_tilde[2,0]**2 + G21_tilde[2,1]**2))
    l1_inv = np.sqrt((G21_tilde[2,0]**2 + G21_tilde[2,1]**2) / (R21[2,0]**2 + R21[2,1]**2))
    
    # alternating estimation
    for _ in range(20):
        # R21[2,0] * l1_inv * m1 = (R21[0,0] - G21_tilde[0,0])
        # R21[2,1] * l1_inv * m1 = (R21[0,1] - G21_tilde[0,1])
        # R21[2,0] * l1_inv * n1 = (R21[1,0] - G21_tilde[1,0])
        # R21[2,1] * l1_inv * n1 = (R21[1,1] - G21_tilde[1,1])
        # -G21_tilde[2,2] * m1 + R21[0,0] * m2 + R21[0,1] * n2 + R21[0,2] * l2 = G21_tilde[0,2]
        # -G21_tilde[2,2] * n1 + R21[1,0] * m2 + R21[1,1] * n2 + R21[1,2] * l2 = G21_tilde[1,2]
        # R21[2,0] * l1_inv * m2 + R21[2,1] * l1_inv * n2 + R21[2,2] * l1_inv * l2 = G21_tilde[2,2]
        # x: [m1, n1, m2, n2]
        A = np.array([
            [R21[2,0] * l1_inv, 0., 0., 0., 0.],
            [R21[2,1] * l1_inv, 0., 0., 0., 0.],
            [0., R21[2,0] * l1_inv, 0., 0., 0.],
            [0., R21[2,1] * l1_inv, 0., 0., 0.],
            [-G21_tilde[2,2], 0., R21[0,0], R21[0,1], R21[0,2]],
            [0., -G21_tilde[2,2], R21[1,0], R21[1,1], R21[1,2]],
            [0., 0., R21[2,0] * l1_inv, R21[2,1] * l1_inv, R21[2,2] * l1_inv],
        ])
        b = np.array([
            R21[0,0] - G21_tilde[0,0],
            R21[0,1] - G21_tilde[0,1],
            R21[1,0] - G21_tilde[1,0],
            R21[1,1] - G21_tilde[1,1],
            G21_tilde[0,2],
            G21_tilde[1,2],
            G21_tilde[2,2]
        ])

        result = np.linalg.lstsq(A,b, rcond=None)
        m1, n1, m2, n2, l2 = result[0]

        # R21[2,0] * m1 * l1_inv = (R21[0,0] - G21_tilde[0,0])
        # R21[2,1] * m1 * l1_inv * m1 = (R21[0,1] - G21_tilde[0,1])
        # R21[2,0] * n1 * l1_inv = (R21[1,0] - G21_tilde[1,0])
        # R21[2,1] * n1 * l1_inv = (R21[1,1] - G21_tilde[1,1])
        # (R21[2,0] * m2 + R21[2,1] * n2 + R21[2,2] * l2) * l1_inv = G21_tilde[2,2]
        # x: [l1]
        A = np.array([
            [R21[2,0] * m1,],
            [R21[2,1] * m1,],
            [R21[2,0] * n1,],
            [R21[2,1] * n1,],
            [R21[2,0] * m2 + R21[2,1] * n2 + R21[2,2] * l2,],
        ])
        b = np.array([
            R21[0,0] - G21_tilde[0,0],
            R21[0,1] - G21_tilde[0,1],
            R21[1,0] - G21_tilde[1,0],
            R21[1,1] - G21_tilde[1,1],
            #G21_tilde[0,2],
            #G21_tilde[1,2],
            G21_tilde[2,2]
        ])
        result = np.linalg.lstsq(A,b, rcond=None)
        l1_inv = result[0][0]

        # G21_tilde[0,0] = R21[0,0] - R21[2,0] * l1_inv * m1
        # G21_tilde[0,1] = R21[0,1] - R21[2,1] * l1_inv * m1
        # G21_tilde[1,0] = R21[1,0] - R21[2,0] * l1_inv * n1
        # G21_tilde[1,1] = R21[1,1] - R21[2,1] * l1_inv * n1
        # G21_tilde[0,2] + m1 * G21_tilde[2,2] = R21[0,0] * m2 + R21[0,1] * n2 + R21[0,2] * l2
        # G21_tilde[1,2] + n1 * G21_tilde[2,2] = R21[1,0] * m2 + R21[1,1] * n2 + R21[1,2] * l2
        # G21_tilde[2,2] = R21[2,0] * l1_inv * m2 + R21[2,1] * l1_inv * n2 + R21[2,2] * l1_inv * l2
        g33 = R21[2,0] * l1_inv * m2 +  R21[2,1] * l1_inv * n2 + R21[2,2] * l1_inv * l2
        A = np.array([
            [G21_tilde[0,0],],
            [G21_tilde[0,1],],
            [G21_tilde[1,0],],
            [G21_tilde[1,1],],
            [G21_tilde[0,2],],
            [G21_tilde[1,2],],
            [G21_tilde[2,0],],
            [G21_tilde[2,1],],
            [G21_tilde[2,2],],
        ])
        b = np.array([
            R21[0,0] - R21[2,0] * l1_inv * m1,
            R21[0,1] - R21[2,1] * l1_inv * m1,
            R21[1,0] - R21[2,0] * l1_inv * n1,
            R21[1,1] - R21[2,1] * l1_inv * n1,
            R21[0,0] * m2 + R21[0,1] * n2 + R21[0,2] * l2 - m1 * g33,
            R21[1,0] * m2 + R21[1,1] * n2 + R21[1,2] * l2 - n1 * g33,
            R21[2,0] * l1_inv,
            R21[2,1] * l1_inv,
            R21[2,0] * l1_inv * m2 + R21[2,1] * l1_inv * n2 + R21[2,2] * l1_inv * l2,
        ])
        result = np.linalg.lstsq(A,b, rcond=None)
        scale = result[0][0]
        G21_tilde = scale * G21_tilde

        # compute residual
        g33 = R21[2,0] * l1_inv * m2 +  R21[2,1] * l1_inv * n2 + R21[2,2] * l1_inv * l2
        G21_tilde_est = np.array([
            R21[0,0] - R21[2,0] * l1_inv * m1,
            R21[0,1] - R21[2,1] * l1_inv * m1,
            R21[0,0] * m2 + R21[0,1] * n2 + R21[0,2] * l2 - m1 * g33,
            R21[1,0] - R21[2,0] * l1_inv * n1,
            R21[1,1] - R21[2,1] * l1_inv * n1,
            R21[1,0] * m2 + R21[1,1] * n2 + R21[1,2] * l2 - n1 * g33,
            R21[2,0] * l1_inv,
            R21[2,1] * l1_inv,
            R21[2,0] * l1_inv * m2 +  R21[2,1] * l1_inv * n2 + R21[2,2] * l1_inv * l2,
        ]).reshape(3,3)

        residual = np.sum((G21_tilde_est - G21_tilde)**2)
        #print('residual:',residual)

    return {
        'params': (1. / l1_inv, m1, n1, l2, m2, n2),
        'G21_tilde': G21_tilde_est
    }


def solve_pose(
        correspondences, 
        opencv_coord=False, 
        use_two_steps=True, 
        initial_param=None, 
        use_multi_initial_params=False,
        wo_gbr=False
    ):
    pixel1 = correspondences['pixel1'] 
    pixel2 = correspondences['pixel2'] 
    normal1_nm = correspondences['normal1_nm'] 
    normal2_nm = correspondences['normal2_nm'] 
    normal1_rm = correspondences['normal1_rm'] 
    normal2_rm = correspondences['normal2_rm']

    if opencv_coord:
        pixel1 = pixel1 * np.array([1.,-1.])
        pixel2 = pixel2 * np.array([1.,-1.])
        normal1_nm = normal1_nm * np.array([1., -1., -1.])
        normal2_nm = normal2_nm * np.array([1., -1., -1.])
        normal1_rm = normal1_rm * np.array([1., -1., -1.])
        normal2_rm = normal2_rm * np.array([1., -1., -1.])
        if not (initial_param is None):
            initial_param = (
                initial_param[0],
                -initial_param[1],
                initial_param[2],
                initial_param[3],
                -initial_param[4],
                initial_param[5],
                initial_param[6] * 
                    np.array([
                        [1., -1., -1.],
                        [-1., 1., 1.],
                        [-1., 1., 1.]
                    ]),
            )

    R21_init = compute_relative_rot(
        np.concatenate([normal2light(normal1_rm), normal1_nm], axis=0), 
        np.concatenate([normal2light(normal2_rm), normal2_nm], axis=0), 
    )
    param_init = encode_param(1.,0.,0.,1.,0.,0., R21_init)

    if wo_gbr:
        def res_func_wo_gbr(params):
            R21 = decode_param(params)[-1]
            normal1 = np.concatenate([normal2light(normal1_rm), normal1_nm], axis=0)
            normal2 = np.concatenate([normal2light(normal2_rm), normal2_nm], axis=0)
            res_normal = (normal1 - normal2 @ R21.T).reshape(-1)

            phi, eta, theta = matrix_to_euler(R21)
            res_pix = res_func_pixel((phi, theta), pixel1, pixel2)

            return np.concatenate([res_pix, res_normal])
        result = scipy.optimize.least_squares(res_func_wo_gbr, param_init)
        R21_est = decode_param(result.x)[-1]

        return {
            # settings
            'method': 'scipy_least_squares_wo_gbr',
            'num_corrs_pix': len(pixel1),
            'num_corrs_nm': len(normal1_nm),
            'num_corrs_rm': len(normal1_rm),
            'optimization_result': result,
            # results
            'R21_est': R21_est,
            'G1_est': np.eye(3),
            'G2_est': np.eye(3),
            'G21_est': R21_est,
            'l1_est': 1.,
            'm1_est': 0.,
            'n1_est': 0.,
            'l2_est': 1.,
            'm2_est': 0.,
            'n2_est': 0.,
        }

    possible_initial_params = [param_init]

    if not (initial_param is None):
        possible_initial_params = [encode_param(*initial_param)]
        print('initial params given')
    elif use_multi_initial_params:

        if (len(normal1_nm) >= 1) and (len(pixel1) >= 4):
            # estimate theta and phi
            possible_angles = solve_phi_theta(pixel1, pixel2)

            if len(possible_angles) == 0:
                print('theta-phi estimation failed')
                #print(result_est_theta_phi)
            else:
                for phi_est, theta_est in possible_angles:
                    #print(phi_est, theta_est)
                    # rough estimation of eta
                    def res_func_eta(param):
                        eta = param[0]
                        R21 = euler_to_matrix(phi_est, eta, theta_est)
                        return (normal1_nm @ R21 - normal2_nm).reshape(-1)
                    eta_est = scipy.optimize.least_squares(res_func_eta, [0,]).x[0]
                    if False:
                        list_eta = np.arange(0,2*np.pi,np.pi/100)
                        list_res = [np.sum(res_func_eta([eta,])**2) for eta in list_eta]
                        import matplotlib.pyplot as plt
                        print(eta_est % (2 * np.pi))
                        plt.plot(list_eta, list_res)
                        plt.show()
                    R21_est = euler_to_matrix(phi_est, eta_est, theta_est)
                    if np.all((normal1_nm @ R21_est)[:,2] * normal2_nm[:,2] > 0) and np.all((normal2_nm @ R21_est.T)[:,2] > 0):
                        possible_initial_params.append(encode_param(1.,0.,0.,1.,0.,0., R21_est))


        if len(possible_initial_params) > 20:
            possible_initial_params = [param_init]

        #sorted_indices = np.argsort([compute_pose_error(decode_param(p)[-1], R21_init) for p in possible_initial_params])
        possible_initial_params = sorted(possible_initial_params, key= lambda p: compute_pose_error(decode_param(p)[-1], R21_init))


    #assert np.all(np.abs(res_func(param_init, pixel1, pixel2, normal1_rm[:0], normal2_rm[:0], normal1_nm, normal2_nm)) < 1e-6)
    best_residual = np.inf
    for param_init in possible_initial_params:
        # First, initialize with normal correspondences
        if len(normal1_nm) > 0:
            result = scipy.optimize.least_squares(
                res_func, 
                param_init, 
                args=(
                    pixel1[:0], pixel2[:0], 
                    normal1_rm[:0], normal2_rm[:0], 
                    normal1_nm, normal2_nm
                ),
                #xtol=1e-15,
                #gtol=1e-15,
                #ftol=1e-15,
                #max_nfev=1000,
            )
            if result.success:
                param_init = result.x

        # 2-step initialization of eta
        if (use_two_steps) and ((len(normal1_rm) >= 1) and ((len(normal1_nm) >= 5) or ((len(normal1_nm) >= 4) and (len(pixel1) >= 4)))):
            # 2 step
            # estimate G21_tilde
            if len(pixel1) > 0:
                result = scipy.optimize.least_squares(
                    res_func, 
                    param_init, 
                    args=(
                        pixel1, pixel2, 
                        normal1_rm[:0], normal2_rm[:0], 
                        normal1_nm, normal2_nm
                    ),
                    #xtol=1e-15,
                    #gtol=1e-15,
                    #ftol=1e-15,
                    #max_nfev=1000,
                )
                if result.success:
                    param_init = result.x

            l1_init, m1_init, n1_init, l2_init, m2_init, n2_init, R21_first = decode_param(param_init)
            phi_init, eta_init, theta_init = matrix_to_euler(R21_first)
            G1_init = get_gbr_matrix(l1_init, m1_init, n1_init)
            G2_init = get_gbr_matrix(l2_init, m2_init, n2_init)
            G21_tilde_init =  np.linalg.inv(G1_init).T @ R21_first @ G2_init.T
            if (G21_tilde_init[2,0]**2 + G21_tilde_init[2,1]**2) > 1e-4:
                def res_func_second_step(param):
                    eta = param[0]
                    R21 = euler_to_matrix(phi_init, eta, theta_init)
                    l1, m1, n1, l2, m2, n2 = decompose_relative_gbr(G21_tilde_init, R21)
                    return res_func(
                        encode_param(l1, m1, n1, l2, m2, n2, R21),
                        pixel1[:0], pixel2[:0],
                        normal1_rm, normal2_rm,
                        normal1_nm[:0], normal2_nm[:0]
                    )
                
                list_eta = np.arange(0.5 * np.pi / 180, 2*np.pi, 2 * np.pi / 180)
                list_res = [np.sum(res_func_second_step((eta,))**2) for eta in list_eta]
                eta_init = list_eta[np.argmin(list_res)]
                #import matplotlib.pyplot as plt
                #plt.semilogy(list_eta, list_res)
                #plt.grid()
                #plt.show()
                if True:
                    result_second = scipy.optimize.least_squares(
                        res_func_second_step, 
                        (eta_init,), 
                        #xtol=1e-15,
                        #gtol=1e-15,
                        #ftol=1e-15,
                        #max_nfev=1000,
                    )
                    if result_second.success:
                        eta_init = result_second.x[0]

                param_init = encode_param(
                    l1_init, m1_init, n1_init, l2_init, m2_init, n2_init,
                    euler_to_matrix(phi_init, eta_init, theta_init)
                )

        # optimization with all correspondences
        result = scipy.optimize.least_squares(
            res_func, 
            param_init, 
            args=(
                pixel1, pixel2, 
                normal1_rm, normal2_rm, 
                normal1_nm, normal2_nm
            ),
            #xtol=1e-15,
            #gtol=1e-15,
            #ftol=1e-15,
            #max_nfev=1000,
        )
        if np.sum(result.fun**2) < best_residual:
            best_residual = np.sum(result.fun**2)
            best_result = result
        if (result.success) and (np.max(np.abs(result.fun)) < 1e-9):
            break
    
    result = best_result
    l1_est, m1_est, n1_est, l2_est, m2_est, n2_est, R21_est = decode_param(result.x)

    G1_est = get_gbr_matrix(l1_est, m1_est, n1_est)
    G2_est = get_gbr_matrix(l2_est, m2_est, n2_est)
    M_est =  np.linalg.inv(G1_est).T @ R21_est @ G2_est.T

    # TODO: translation estimation
    u_ = R21_est[0,0] * pixel2[:,0] + R21_est[0,1] * pixel2[:,1] - pixel1[:,0]
    v_ = R21_est[1,0] * pixel2[:,0] + R21_est[1,1] * pixel2[:,1] - pixel1[:,1]
    w = -np.sum(R21_est[1,2] * u_ - R21_est[0,2] * v_)
    t21_est_ofs = np.array([R21_est[0,2], R21_est[1,2], 0.])
    if np.abs(R21_est[1,2]) > np.abs(R21_est[0,2]):
        t21_est0 = np.array([w / R21_est[1,2], 0., 0.])
    else:
        t21_est0 = np.array([0., w / R21_est[1,2], 0.])

    if opencv_coord:
        # TODO: coordinate system conversion
        R21_est = R21_est * np.array([
            [1., -1., -1.],
            [-1., 1., 1.],
            [-1., 1., 1.]
        ])
        G1_est[2,0] *= -1
        G2_est[2,0] *= -1
        M_est =  np.linalg.inv(G1_est).T @ R21_est @ G2_est.T
        m1_est *= -1.
        m2_est *= -1.
        pass

    return {
        # settings
        'method': 'scipy_least_squares',
        'num_corrs_pix': len(pixel1),
        'num_corrs_nm': len(normal1_nm),
        'num_corrs_rm': len(normal1_rm),
        'optimization_result': result,
        # results
        'R21_est': R21_est,
        'G1_est': G1_est,
        'G2_est': G2_est,
        'G21_est': M_est,
        'l1_est': l1_est,
        'm1_est': m1_est,
        'n1_est': n1_est,
        'l2_est': l2_est,
        'm2_est': m2_est,
        'n2_est': n2_est,
    }

def solve_translation_perspective(pixel1, pixel2, K1, K2, R21, loss='linear'):
    pixel1_ = np.concatenate([pixel1, np.ones_like(pixel1[:,0:1])], axis=1)
    pixel2_ = np.concatenate([pixel2, np.ones_like(pixel2[:,0:1])], axis=1)

    def res_tran(param):
        tx, ty, tz = param / (np.linalg.norm(param) + 1e-9)

        T = np.array([
            [0., -tz, ty],
            [tz, 0, -tx],
            [-ty, tx, 0]
        ])
        E = np.linalg.inv(K1).T @ T @ R21 @ np.linalg.inv(K2)
        return np.sum(pixel1_ * (E @ pixel2_.T).T, axis=-1)
    
    t0 = np.array([0., 0., 1.,])
    result = scipy.optimize.least_squares(res_tran, t0, loss=loss)
    return result.x

def solve_rotation_and_translation_perspective(pixel1, pixel2, K1, K2):
    pixel1_ = np.concatenate([pixel1, np.ones_like(pixel1[:,0:1])], axis=1)
    pixel2_ = np.concatenate([pixel2, np.ones_like(pixel2[:,0:1])], axis=1)

    def res_tran(param):
        R21 = cv2.Rodrigues(param[:3])[0]
        tx, ty, tz = param[3:6] / (np.linalg.norm(param[3:6]) + 1e-9)

        T = np.array([
            [0., -tz, ty],
            [tz, 0, -tx],
            [-ty, tx, 0]
        ])
        E = np.linalg.inv(K1).T @ T @ R21 @ np.linalg.inv(K2)
        return np.sum(pixel1_ * (E @ pixel2_.T).T, axis=-1)
    
    t0 = np.array([0., 0., 0., 0., 0., 1.,])
    result = scipy.optimize.least_squares(res_tran, t0)

    R21 = cv2.Rodrigues(result.x[:3])[0]
    tvec = result.x[3:6] / (np.linalg.norm(result.x[3:6]) + 1e-9)
    return R21, tvec

def eval_reprojection_error_perspective(pixel1, pixel2, K1, K2, R21, t21):
    pixel1_ = np.concatenate([pixel1, np.ones_like(pixel1[:,0:1])], axis=1)
    pixel2_ = np.concatenate([pixel2, np.ones_like(pixel2[:,0:1])], axis=1)
    p1_ = pixel1_ @ np.linalg.inv(K1).T
    p2_ = pixel2_ @ np.linalg.inv(K2).T
    p2_w = p2_ @ R21.T

    surf_points = []
    for i in range(len(pixel1)):
        A = np.stack([p1_[i], -p2_w[i]], axis=1)
        x, residual = np.linalg.lstsq(A,t21, rcond=None)[:2]

        surf_points.append(
            #x[0] * p1_[i]
            #x[1] * p2_w[i] + t21
            0.5 * (x[0] * p1_[i] + x[1] * p2_w[i] + t21)
        )
    surf_points = np.stack(surf_points, axis=0)

    pixel1_reproj_ = surf_points @ K1.T
    pixel1_reproj = pixel1_reproj_[:,:2] / pixel1_reproj_[:,2:3]

    pixel2_reproj_ = (surf_points @ R21.T + t21) @ K2.T
    pixel2_reproj = pixel2_reproj_[:,:2] / pixel2_reproj_[:,2:3]

    return np.concatenate([pixel1 - pixel1_reproj, pixel2 - pixel2_reproj], axis=1)

def solve_surf_point_locations_perspective(pixel1, pixel2, K1, K2, R21, t21):
    pixel1_ = np.concatenate([pixel1, np.ones_like(pixel1[:,0:1])], axis=1)
    pixel2_ = np.concatenate([pixel2, np.ones_like(pixel2[:,0:1])], axis=1)
    p1_ = pixel1_ @ np.linalg.inv(K1).T
    p2_ = pixel2_ @ np.linalg.inv(K2).T
    p2_w = p2_ @ R21.T

    surf_points = []
    for i in range(len(pixel1)):
        A = np.stack([p1_[i], -p2_w[i]], axis=1)
        x, residual = np.linalg.lstsq(A,t21, rcond=None)[:2]

        surf_points.append(
            #x[0] * p1_[i]
            #x[1] * p2_w[i] + t21
            0.5 * (x[0] * p1_[i] + x[1] * p2_w[i] + t21)
        )
    surf_points = np.stack(surf_points, axis=0)
    return surf_points

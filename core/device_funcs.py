import numba
from numba import cuda

import math

@cuda.jit(device=True)
def scale_vector_device(scale, v):
    return scale * v[0], scale * v[1], scale * v[2]

@cuda.jit(device=True)
def add_vector_device(v1, v2):
    return v1[0]+v2[0], v1[1]+v2[1], v1[2]+v2[2]

@cuda.jit(device=True)
def dot_device(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

@cuda.jit(device=True)
def normalize_device(v):
    x,y,z = v
    l = math.sqrt(x*x+y*y+z*z)+1e-12
    return x/l, y/l, z/l

@cuda.jit(device=True)
def cross_device(v1, v2):
    x1,y1,z1 = v1
    x2,y2,z2 = v2
    x = y1*z2 - y2*z1
    y = z1*x2 - z2*x1
    z = x1*y2 - x2*y1
    return x,y,z

@cuda.jit(device=True)
def eye3_device():
    return (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

@cuda.jit(device=True)
def add_mat33_device(a,b):
    c00 = a[0][0] + b[0][0]
    c01 = a[0][1] + b[0][1]
    c02 = a[0][2] + b[0][2]
    c10 = a[1][0] + b[1][0]
    c11 = a[1][1] + b[1][1]
    c12 = a[1][2] + b[1][2]
    c20 = a[2][0] + b[2][0]
    c21 = a[2][1] + b[2][1]
    c22 = a[2][2] + b[2][2]
    return ((c00,c01,c02), (c10,c11,c12), (c20,c21,c22))

@cuda.jit(device=True)
def scale_mat33_device(s,b):
    c00 = s * b[0][0]
    c01 = s * b[0][1]
    c02 = s * b[0][2]
    c10 = s * b[1][0]
    c11 = s * b[1][1]
    c12 = s * b[1][2]
    c20 = s * b[2][0]
    c21 = s * b[2][1]
    c22 = s * b[2][2]
    return ((c00,c01,c02), (c10,c11,c12), (c20,c21,c22))

@cuda.jit(device=True)
def matmul333_device(a,b):
    c00 = a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0]
    c01 = a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1]
    c02 = a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2]
    c10 = a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0]
    c11 = a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1]
    c12 = a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2]    
    c20 = a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0]
    c21 = a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1]
    c22 = a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2]
    return ((c00,c01,c02), (c10,c11,c12), (c20,c21,c22))

@cuda.jit(device=True)
def mul_vec_mat_device(v,mat):
    c00 = v[0] * mat[0][0] + v[1] * mat[1][0] + v[2] * mat[2][0]
    c01 = v[0] * mat[0][1] + v[1] * mat[1][1] + v[2] * mat[2][1]
    c02 = v[0] * mat[0][2] + v[1] * mat[1][2] + v[2] * mat[2][2]
    return (c00,c01,c02)

@cuda.jit(device=True)
def mul_mat_vec_device(mat,v):
    c00 = v[0] * mat[0][0] + v[1] * mat[0][1] + v[2] * mat[0][2]
    c01 = v[0] * mat[1][0] + v[1] * mat[1][1] + v[2] * mat[1][2]
    c02 = v[0] * mat[2][0] + v[1] * mat[2][1] + v[2] * mat[2][2]
    return (c00,c01,c02)
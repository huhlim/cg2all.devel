import torch
import numpy as np
from libconfig import EPS

pi = torch.tensor(np.pi)

# some basic functions
v_size = lambda v: torch.linalg.norm(v, dim=-1)
v_norm = lambda v: v / v_size(v)[..., None]


def v_norm_safe(v, index=0):
    u = v.clone()
    u[..., index] = u[..., index] + EPS
    return v_norm(u)


def inner_product(v1, v2):
    return torch.sum(v1 * v2, dim=-1)


def rotate_matrix(R, X):
    return R @ X


def rotate_vector(R, X):
    return (X[..., None, :] @ R.mT)[..., 0, :]


def rotate_vector_inv(R, X):
    R_inv = torch.inverse(R)
    return rotate_vector(R_inv, X)


def angle_sign(x):
    s = torch.sign(x)
    s[s == 0] = 1.0
    return s


def torsion_angle(R: torch.Tensor) -> torch.Tensor:
    torsion_axis = v_norm(R[..., 2, :] - R[..., 1, :])
    v0 = v_norm(R[..., 0, :] - R[..., 1, :])
    v1 = v_norm(R[..., 3, :] - R[..., 2, :])
    n0 = v_norm_safe(torch.linalg.cross(v0, torsion_axis, dim=-1))
    n1 = v_norm_safe(torch.linalg.cross(v1, torsion_axis, dim=-1))
    angle = torch.acos(torch.clamp(inner_product(n0, n1), -1.0+EPS, 1.0-EPS))
    sign = angle_sign(inner_product(v0, n1))
    return angle * sign

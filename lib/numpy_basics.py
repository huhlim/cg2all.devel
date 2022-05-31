# %%
# load modules
import numpy as np

# %%
# Some basic functions
v_size = lambda v: np.linalg.norm(v, axis=-1)
v_norm = lambda v: v / v_size(v)[..., None]


def inner_product(v1, v2):
    return np.sum(v1 * v2, axis=-1)


def angle_sign(x):
    if isinstance(x, np.ndarray):
        s = np.sign(x)
        s[s == 0] = 1.0
        return s
    elif x >= 0:
        return 1.0
    else:
        return -1.0


# %%
# Some geometry functions
def bond_length(R) -> float:
    return v_size(R[..., 1, :] - R[..., 0, :])


# bond_angle: returns the angle consist of three atoms
def bond_angle(R) -> float:
    v1 = R[..., 0, :] - R[..., 1, :]
    v2 = R[..., 2, :] - R[..., 1, :]
    return np.arccos(np.clip(inner_product(v_norm(v1), v_norm(v2)), -1.0, 1.0))


# torsion_angle: returns the torsion angle consist of four atoms
def torsion_angle(R) -> float:
    torsion_axis = v_norm(R[..., 2, :] - R[..., 1, :])
    v0 = v_norm(R[..., 0, :] - R[..., 1, :])
    v1 = v_norm(R[..., 3, :] - R[..., 2, :])
    n0 = v_norm(np.cross(v0, torsion_axis))
    n1 = v_norm(np.cross(v1, torsion_axis))
    angle = np.arccos(np.clip(inner_product(n0, n1), -1.0, 1.0))
    sign = angle_sign(inner_product(v0, n1))
    return angle * sign


# %%
# Algorithm 21. Rigid from 3 points using the Gram-Schmidt process
def rigid_from_3points(x):
    v0 = x[2] - x[1]
    v1 = x[0] - x[1]
    e0 = v_norm(v0)
    u1 = v1 - e0 * e0.T.dot(v1)
    e1 = v_norm(u1)
    e2 = np.cross(e0, e1)
    R = np.vstack((e0, e1, e2))
    t = x[1]
    return (R, t)


# %%
# Translate and rotate a set of coordinates
def translate_and_rotate(x, R, t):
    # return R.dot(x.T).T + t
    return np.moveaxis(R.dot(x.T), -1, -2) + t[..., None, :]


# %%
# Rotate around the x-axis
def rotate_x(t_ang):
    R = np.array(
        [
            [1, 0, 0],
            [0, np.cos(t_ang), -np.sin(t_ang)],
            [0, np.sin(t_ang), np.cos(t_ang)],
        ]
    )
    t = np.zeros(3)
    return (R, t)

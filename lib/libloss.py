#!/usr/bin/env python

import torch

from residue_constants import (
    ATOM_INDEX_N,
    ATOM_INDEX_CA,
    ATOM_INDEX_C,
    BOND_LENGTH0,
    BOND_ANGLE0,
    TORSION_ANGLE0,
)

from libconfig import DTYPE, EPS


# some basic functions
v_size = lambda v: torch.linalg.norm(v, dim=-1)
v_norm = lambda v: v / v_size(v)[..., None]


# MSE loss for comparing coordinates
def loss_f_mse_R(R, R_ref, R_mask):
    dr_sq = torch.sum(torch.pow(R - R_ref, 2) * R_mask[..., None])
    return dr_sq / R.size(0)


# MSE loss for comparing C-alpha coordinates
def loss_f_mse_R_CA(R, R_ref):
    return torch.mean(torch.pow(R[:, ATOM_INDEX_CA, :] - R_ref[:, ATOM_INDEX_CA, :], 2))


# Bonded energy penalties
def loss_f_bonded_energy(R, is_continuous, weight_s=(1.0, 1.0, 1.0)):
    if weight_s[0] == 0.0:
        return 0.0

    bonded = is_continuous[1:]
    n_bonded = torch.sum(bonded)

    # vector: -C -> N
    v1 = R[1:, ATOM_INDEX_N, :] - R[:-1, ATOM_INDEX_C, :]
    #
    # bond lengths
    d1 = v_size(v1)
    bond_energy = torch.sum(torch.pow(d1 - BOND_LENGTH0, 2) * bonded) / n_bonded
    print ("bond energy:", bond_energy)
    if weight_s[1] == 0.0:
        return bond_energy * weight_s[0]
    #
    # vector: -CA -> -C
    v0 = R[:-1, ATOM_INDEX_C, :] - R[:-1, ATOM_INDEX_CA, :]
    # vector: N -> CA
    v2 = R[1:, ATOM_INDEX_CA, :] - R[1:, ATOM_INDEX_N, :]
    #
    d0 = v_size(v0)
    d2 = v_size(v2)
    #
    # bond angles
    def bond_angle(v1, v2):
        # torch.acos is unstable around -1 and 1 -> added EPS
        return torch.acos(torch.clamp(torch.inner(v1, v2), -1.0 + EPS, 1.0 - EPS))

    v0 = v0 / d0[..., None]
    v1 = v1 / d1[..., None]
    v2 = v2 / d2[..., None]
    a01 = bond_angle(-v0, v1)
    a12 = bond_angle(-v1, v2)
    angle_energy = torch.pow(a01 - BOND_ANGLE0[0], 2) + torch.pow(
        a12 - BOND_ANGLE0[1], 2
    )
    angle_energy = torch.sum(angle_energy * bonded) / n_bonded
    print ("angle_energy", angle_energy)

    if weight_s[2] == 0.0:
        return bond_energy * weight_s[0] + angle_energy * weight_s[1]
    #
    # torsion angles without their signs
    def torsion_angle_without_sign(v0, v1, v2):
        n0 = v_norm(torch.cross(v2, v1))
        n1 = v_norm(torch.cross(-v0, v1))
        angle = bond_angle(n0, n1)
        return angle  # between 0 and pi

    t_ang = torsion_angle_without_sign(v0, v1, v2)
    d_ang = torch.minimum(t_ang - TORSION_ANGLE0[0], TORSION_ANGLE0[1] - t_ang)
    torsion_energy = torch.sum(torch.pow(d_ang, 2) * bonded) / n_bonded
    return (
        bond_energy * weight_s[0]
        + angle_energy * weight_s[1]
        + torsion_energy * weight_s[2]
    )

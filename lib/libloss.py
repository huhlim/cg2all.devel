#!/usr/bin/env python

from residue_constants import BACKBONE_ATOM_s

import torch
from numpy import deg2rad

from libconfig import DTYPE


atom_index_N = BACKBONE_ATOM_s.index("N")
atom_index_CA = BACKBONE_ATOM_s.index("CA")
atom_index_C = BACKBONE_ATOM_s.index("C")
atom_index_O = BACKBONE_ATOM_s.index("O")


# some basic functions
v_size = lambda v: torch.linalg.norm(v, dim=-1)
v_norm = lambda v: v / v_size(v)[..., None]


# MSE loss for comparing coordinates
def loss_f_mse_R(R, R_ref, R_mask):
    dr_sq = torch.sum(torch.pow(R - R_ref, 2) * R_mask[...,None])
    return dr_sq / R.size(0)


# MSE loss for comparing C-alpha coordinates
def loss_f_mse_R_CA(R, R_ref):
    return torch.mean(torch.pow(R[:, atom_index_CA, :] - R_ref[:, atom_index_CA, :], 2))


# Bonded energy penalties
BOND_LENGTH0 = 0.1345
BOND_ANGLE0 = (deg2rad(120.0), deg2rad(116.5))
TORSION_ANGLE0 = (deg2rad(0.0), deg2rad(180.0))


def loss_f_bonded_energy(R, weight_s=(1.0, 1.0, 1.0)):
    if weight_s[0] == 0.0:
        return 0.0

    # vector: -CA -> -C
    v0 = R[:-1, atom_index_C, :] - R[:-1, atom_index_CA, :]
    # vector: -C -> N
    v1 = R[1:, atom_index_N, :] - R[:-1, atom_index_C, :]
    # vector: N -> CA
    v2 = R[1:, atom_index_CA, :] - R[1:, atom_index_N, :]
    #
    # bond lengths
    d0 = v_size(v0)
    d1 = v_size(v1)
    d2 = v_size(v2)
    bond_energy = torch.mean(torch.pow(d1 - BOND_LENGTH0, 2))
    if weight_s[1] == 0.0:
        return bond_energy * weight_s[0]
    #
    # bond angles
    def bond_angle(v1, v2):
        return torch.acos(torch.clamp(torch.inner(v1, v2), -1.0, 1.0))

    v0 = v0 / d0[..., None]
    v1 = v1 / d1[..., None]
    v2 = v2 / d2[..., None]
    a01 = bond_angle(-v0, v1)
    a12 = bond_angle(-v1, v2)
    angle_energy = torch.mean(torch.pow(a01 - BOND_ANGLE0[0], 2)) + torch.mean(
        torch.pow(a12 - BOND_ANGLE0[1], 2)
    )
    if weight_s[2] == 0.0:
        return bond_energy * weight_s[0] + angle_energy * weight_s[1]
    #
    # torsion angles without their signs
    def torsion_angle_without_sign(v0, v1, v2):
        n0 = v_norm(torch.cross(v2, v1))
        n1 = v_norm(torch.cross(-v0, v1))
        angle = torch.abs(bond_angle(n0, n1))
        return angle  # between 0 and pi

    t_ang = torsion_angle_without_sign(v0, v1, v2)
    d_ang = torch.minimum(t_ang - TORSION_ANGLE0[0], TORSION_ANGLE0[1] - t_ang)
    torsion_energy = torch.mean(torch.pow(d_ang, 2))
    return (
        bond_energy * weight_s[0]
        + angle_energy * weight_s[1]
        + torsion_energy * weight_s[2]
    )

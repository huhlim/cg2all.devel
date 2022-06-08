#!/usr/bin/env python

import torch
import torch_cluster

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
def loss_f_mse_R(R, R_ref, mask):
    dr_sq = torch.sum(torch.pow(R - R_ref, 2) * mask[..., None])
    # return dr_sq / R.size(0)
    return dr_sq / mask.sum()


def loss_f_rigid_body(R: torch.Tensor, R_ref: torch.Tensor) -> torch.Tensor:
    loss_translation = torch.mean(
        torch.pow(R[:, ATOM_INDEX_CA, :] - R_ref[:, ATOM_INDEX_CA, :], 2)
    )
    #
    v0 = v_norm(R[:, ATOM_INDEX_C, :] - R[:, ATOM_INDEX_CA, :])
    v0_ref = v_norm(R_ref[:, ATOM_INDEX_C, :] - R_ref[:, ATOM_INDEX_CA, :])
    v1 = v_norm(R[:, ATOM_INDEX_N, :] - R[:, ATOM_INDEX_CA, :])
    v1_ref = v_norm(R_ref[:, ATOM_INDEX_N, :] - R_ref[:, ATOM_INDEX_CA, :])
    loss_rotation = torch.mean(
        torch.pow(1.0 - torch.inner(v0, v0_ref), 2)
        + torch.mean(torch.pow(1.0 - torch.inner(v1, v1_ref), 2))
    )
    #
    return loss_translation + loss_rotation


# Bonded energy penalties
def loss_f_bonded_energy(R, is_continuous, weight_s=(1.0, 0.0, 0.0)):
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


def loss_f_torsion_angle(sc, sc0, mask, norm_weight=0.01):
    torsion = sc.reshape(sc.size(0), -1, 2)
    norm = torch.linalg.norm(torsion, dim=2)
    #
    sc_cos = torch.cos(sc0)
    sc_sin = torch.sin(sc0)
    #
    # loss norm is to make torsion angle prediction stable
    loss_norm = torch.sum(torch.pow(norm - 1.0, 2) * mask)
    loss_cos = torch.sum(torch.pow(torsion[:, :, 0] / norm - sc_cos, 2) * mask)
    loss_sin = torch.sum(torch.pow(torsion[:, :, 1] / norm - sc_sin, 2) * mask)
    loss = (loss_cos + loss_sin + loss_norm * norm_weight) / sc.size(0)
    return loss


def loss_f_distance_matrix(R, batch, radius=1.0):
    R_ref = batch.output_xyz[:, ATOM_INDEX_CA]
    edge_src, edge_dst = torch_cluster.radius_graph(
            R_ref, radius, batch=batch.batch)
    d_ref = v_size(R_ref[edge_dst] - R_ref[edge_src])
    d = v_size(R[edge_dst, ATOM_INDEX_CA] - R[edge_src, ATOM_INDEX_CA])
    #
    loss = torch.nn.functional.mse_loss(d, d_ref)
    return loss


#def loss_f_distogram(_R, batch):
#    loss = 0.0
#    for idx, data in enumerate(batch.to_data_list()):
#        R_ref = data.output_xyz[:, ATOM_INDEX_CA, :]
#        start = int(batch._slice_dict["output_atom_mask"][idx])
#        end = int(batch._slice_dict["output_atom_mask"][idx + 1])
#        d_ref = R_to_dist(R_ref)
#        h_ref = dist_to_distogram(d_ref, return_index=True)[None, :]
#        #
#        R = _R[start:end, ATOM_INDEX_CA, :]
#        d = R_to_dist(R)
#        h = torch.moveaxis(dist_to_distogram(d, return_index=False), -1, 0)[None, :]
#        #
#        loss += torch.nn.functional.cross_entropy(h, h_ref)
#    return loss
#
#
#def R_to_dist(R: torch.Tensor) -> torch.Tensor:
#    dr = R[:, None] - R[None, :]
#    return v_size(dr)
#
#
#def dist_to_distogram(
#    d: torch.Tensor, d_min=0.2, d_max=1.0, d_bin=0.05, return_index=False
#) -> torch.Tensor:
#    if return_index:
#        idx = torch.floor(
#            (torch.clip(d, min=d_min, max=d_max - EPS) - d_min) / d_bin
#        ).type(torch.long)
#        return idx
#    else:
#        n_bin = int((d_max - d_min) / d_bin)
#        d0 = (
#            torch.arange(d_min, d_max + EPS, d_bin, device=d.device, dtype=DTYPE)
#            + d_bin * 0.5
#        )
#        delta_d = torch.clip(d[:, :, None], min=d_min) - d0[None, None, :]
#        h = torch.exp(-0.5 * torch.pow(delta_d / (2.0 * d_bin), 2))
#        h_sum = torch.sum(h[:, :, :-1], dim=-1)
#        h = h / torch.clip(h_sum[:, :, None], min=1.0)
#        h_last = 1.0 - torch.clip(h_sum, min=0.0, max=1.0)
#        h[:, :, -1] += h_last - h[:, :, -1]
#        return h

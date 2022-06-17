#!/usr/bin/env python

import torch
import torch_cluster
import torch_geometric

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


def v_norm_safe(v):
    u = v.clone()
    u[..., 0] = u[..., 0] + EPS
    return v_norm(u)


def inner_product(v1, v2):
    return torch.sum(v1 * v2, dim=-1)


# MSE loss for comparing coordinates
def loss_f_mse_R(R, R_ref, mask):
    dr_sq = torch.sum(torch.pow(R - R_ref, 2) * mask[..., None])
    return dr_sq / mask.sum()


def loss_f_rigid_body(R: torch.Tensor, R_ref: torch.Tensor) -> torch.Tensor:
    # deviation of the backbone rigids, N, CA, C
    loss_translation = torch.mean(torch.pow(R[:, :3, :] - R_ref[:, :3, :], 2))
    #
    v0 = v_norm(R[:, ATOM_INDEX_C, :] - R[:, ATOM_INDEX_CA, :])
    v0_ref = v_norm(R_ref[:, ATOM_INDEX_C, :] - R_ref[:, ATOM_INDEX_CA, :])
    v1 = v_norm(R[:, ATOM_INDEX_N, :] - R[:, ATOM_INDEX_CA, :])
    v1_ref = v_norm(R_ref[:, ATOM_INDEX_N, :] - R_ref[:, ATOM_INDEX_CA, :])
    loss_rotation_0 = torch.mean(torch.pow(1.0 - inner_product(v0, v0_ref), 2))
    loss_rotation_1 = torch.mean(torch.pow(1.0 - inner_product(v1, v1_ref), 2))
    #
    return loss_translation + loss_rotation_0 + loss_rotation_1


# distance between two quaternions
def loss_f_quaternion(
    bb: torch.Tensor, bb1: torch.Tensor, q_ref: torch.Tensor, norm_weight: float = 0.01
) -> torch.Tensor:
    # d(q1, q2) = 1.0 - <q1, q2>^2
    loss_quat_dist = torch.mean(1.0 - torch.sum(bb[:, :4] * q_ref, dim=-1))
    #
    if bb1 is not None:
        quat_norm = torch.linalg.norm(bb1[:, :4], dim=-1)
        loss_quat_norm = torch.mean(torch.pow(quat_norm - 1.0, 2))
    else:
        loss_quat_norm = 0.0
    return loss_quat_dist + loss_quat_norm * norm_weight


def loss_f_FAPE_CA(
    batch: torch_geometric.data.Batch,
    R: torch.Tensor,
    bb: torch.Tensor,
    d_clamp: float = 1.0,
) -> torch.Tensor:
    def rotate_vector_inv(R, X):
        R_inv = torch.inverse(R)
        return torch.einsum("...ij,...j", R_inv, X)

    R_ref = batch.output_xyz[:, ATOM_INDEX_CA]
    bb_ref = batch.correct_bb
    #
    batch_size = batch.batch.max().item() + 1
    loss = torch.zeros(batch_size, device=R.device, dtype=DTYPE)
    n_residue = torch.zeros(batch_size, device=R.device, dtype=DTYPE)
    for i in range(R.size(0)):
        batch_index = batch.batch[i]
        selected = batch.batch == batch_index
        r = rotate_vector_inv(bb[i, :3], R[selected, ATOM_INDEX_CA] - bb[i, 3])
        r_ref = rotate_vector_inv(bb_ref[i, :3], R_ref[selected] - bb_ref[i, 3])
        dr = r - r_ref
        d = torch.clamp(
            torch.sqrt(torch.pow(dr, 2).sum(dim=-1) + EPS**2), max=d_clamp
        )
        loss[batch_index] = loss[batch_index] + torch.mean(d)
        n_residue[batch_index] = n_residue[batch_index] + 1.0
    return torch.mean(loss / n_residue)


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
        return torch.acos(torch.clamp(inner_product(v1, v2), -1.0 + EPS, 1.0 - EPS))

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


def loss_f_torsion_angle(sc, sc1, sc_ref, mask, norm_weight=0.01):
    # loss norm is to make torsion angle prediction stable
    if sc1 is not None:
        norm = v_size(sc1)
        loss_norm = torch.sum(torch.pow(norm - 1.0, 2) * mask)
    else:
        loss_norm = 0.0
    #
    sc_cos = torch.cos(sc_ref)
    sc_sin = torch.sin(sc_ref)
    #
    loss_cos = torch.sum(torch.pow(sc[:, :, 0] - sc_cos, 2) * mask)
    loss_sin = torch.sum(torch.pow(sc[:, :, 1] - sc_sin, 2) * mask)
    loss = (loss_cos + loss_sin + loss_norm * norm_weight) / sc.size(0)
    return loss


def loss_f_distance_matrix(R, batch, radius=1.0):
    R_ref = batch.output_xyz[:, ATOM_INDEX_CA]
    edge_src, edge_dst = torch_cluster.radius_graph(
        R_ref, radius, batch=batch.batch, max_num_neighbors=R_ref.size(0) - 1
    )
    d_ref = v_size(R_ref[edge_dst] - R_ref[edge_src])
    d = v_size(R[edge_dst, ATOM_INDEX_CA] - R[edge_src, ATOM_INDEX_CA])
    #
    loss = torch.nn.functional.mse_loss(d, d_ref)
    return loss

#!/usr/bin/env python

import torch
import dgl

from residue_constants import (
    PROLINE_INDEX,
    ATOM_INDEX_N,
    ATOM_INDEX_CA,
    ATOM_INDEX_C,
    ATOM_INDEX_PRO_CD,
    ATOM_INDEX_CYS_SG,
    BOND_LENGTH0,
    BOND_LENGTH_PROLINE_RING,
    BOND_LENGTH_DISULFIDE,
    BOND_ANGLE0,
    TORSION_ANGLE0,
    RIGID_GROUPS_DEP,
)

from libconfig import DTYPE, EPS
from libcg import get_residue_center_of_mass
from torch_basics import v_size, v_norm, v_norm_safe, inner_product


def loss_f(batch, ret, loss_weight, loss_prev=None):
    R = ret["R"]
    opr_bb = ret["opr_bb"]
    #
    loss = {}
    if loss_weight.get("rigid_body", 0.0) > 0.0:
        loss["rigid_body"] = (
            loss_f_rigid_body(R, batch.ndata["output_xyz"]) * loss_weight.rigid_body
        )
    if loss_weight.get("mse_R", 0.0) > 0.0:
        loss["mse_R"] = (
            loss_f_mse_R(R, batch.ndata["output_xyz"], batch.ndata["pdb_atom_mask"])
            * loss_weight.mse_R
        )
    if loss_weight.get("v_cntr", 0.0) > 0.0:
        loss["v_cntr"] = (
            loss_f_v_cntr(R, batch.ndata["atomic_mass"], batch.ndata["v_cntr"]) * loss_weight.v_cntr
        )
    if loss_weight.get("FAPE_CA", 0.0) > 0.0:
        loss["FAPE_CA"] = loss_f_FAPE_CA(batch, R, opr_bb, d_clamp=2.0) * loss_weight.FAPE_CA
    if loss_weight.get("FAPE_all", 0.0) > 0.0:
        loss["FAPE_all"] = loss_f_FAPE_all(batch, R, opr_bb, d_clamp=2.0) * loss_weight.FAPE_all
    if loss_weight.get("rotation_matrix", 0.0) > 0.0:
        loss["rotation_matrix"] = (
            loss_f_rotation_matrix(ret["bb"], ret["bb0"], batch.ndata["correct_bb"])
            * loss_weight.rotation_matrix
        )
    if loss_weight.get("bonded_energy", 0.0) > 0.0:
        loss["bonded_energy"] = (
            loss_f_bonded_energy(R, batch.ndata["continuous"]) + loss_f_bonded_energy_aux(batch, R)
        ) * loss_weight.bonded_energy
    if loss_weight.get("distance_matrix", 0.0) > 0.0:
        loss["distance_matrix"] = loss_f_distance_matrix(R, batch) * loss_weight.distance_matrix
    if loss_weight.get("torsion_angle", 0.0) > 0.0:
        loss["torsion_angle"] = (
            loss_f_torsion_angle(
                ret["sc"], ret["sc0"], batch.ndata["correct_torsion"], batch.ndata["torsion_mask"]
            )
            * loss_weight.torsion_angle
        )
    if loss_weight.get("atomic_clash", 0.0) > 0.0:
        loss["atomic_clash"] = loss_f_atomic_clash(R, batch) * loss_weight.atomic_clash
    #
    if loss_prev is not None:
        for k, v in loss_prev.items():
            if k in loss:
                loss[k] += v
            else:
                loss[k] = v
    return loss


# MSE loss for comparing coordinates
def loss_f_mse_R(R, R_ref, mask):
    dr_sq = torch.sum(torch.abs(R - R_ref) * mask[..., None])
    return dr_sq / mask.sum()


def loss_f_v_cntr(R, mass, v_cntr_ref):
    r_cntr = get_residue_center_of_mass(R, mass)
    v_cntr = v_norm_safe(r_cntr - R[:, ATOM_INDEX_CA])
    return torch.mean(torch.abs(1.0 - inner_product(v_cntr, v_cntr_ref)))


def loss_f_rigid_body(R: torch.Tensor, R_ref: torch.Tensor) -> torch.Tensor:
    # deviation of the backbone rigids, N, CA, C
    loss_translation = torch.mean(torch.abs(R[:, :3, :] - R_ref[:, :3, :]))
    #
    v0 = v_norm(R[:, ATOM_INDEX_C, :] - R[:, ATOM_INDEX_CA, :])
    v0_ref = v_norm(R_ref[:, ATOM_INDEX_C, :] - R_ref[:, ATOM_INDEX_CA, :])
    v1 = v_norm(R[:, ATOM_INDEX_N, :] - R[:, ATOM_INDEX_CA, :])
    v1_ref = v_norm(R_ref[:, ATOM_INDEX_N, :] - R_ref[:, ATOM_INDEX_CA, :])
    loss_rotation_0 = torch.mean(torch.abs(1.0 - inner_product(v0, v0_ref)))
    loss_rotation_1 = torch.mean(torch.abs(1.0 - inner_product(v1, v1_ref)))
    #
    return loss_translation + loss_rotation_0 + loss_rotation_1


def loss_f_rotation_matrix(
    bb: torch.Tensor, bb0: torch.Tensor, bb_ref: torch.Tensor, norm_weight: float = 0.01
) -> torch.Tensor:
    loss_bb = torch.mean(torch.abs(bb[:, :2] - bb_ref[:, :2]))
    #
    if bb0 is not None and norm_weight > 0.0:
        loss_bb_norm_1 = torch.mean(torch.abs(v_size(bb0[:, 0]) - 1.0))
        loss_bb_norm_2 = torch.mean(torch.abs(v_size(bb0[:, 1]) - 1.0))
        return loss_bb + (loss_bb_norm_1 + loss_bb_norm_2) * norm_weight
    else:
        return loss_bb


def loss_f_FAPE_CA(
    batch: dgl.DGLGraph, R: torch.Tensor, bb: torch.Tensor, d_clamp: float = 2.0
) -> torch.Tensor:
    def rotate_vector_inv(R, X):
        R_inv = torch.inverse(R)
        return (X[..., None, :] @ R.mT)[..., 0, :]

    first = 0
    loss = torch.zeros(batch.batch_size, device=R.device, dtype=DTYPE)
    for batch_index, data in enumerate(dgl.unbatch(batch)):
        n_residue = data.num_nodes()
        last = first + n_residue
        #
        _R = R[first:last, ATOM_INDEX_CA]
        _bb = bb[first:last]
        R_ref = data.ndata["output_xyz"][:, ATOM_INDEX_CA]
        bb_ref = data.ndata["correct_bb"]
        #
        for i in range(n_residue):
            r = rotate_vector_inv(_bb[i, :3], _R - _bb[i, 3])
            r_ref = rotate_vector_inv(bb_ref[i, :3], R_ref - bb_ref[i, 3])
            dr = r - r_ref
            d = torch.clamp(torch.sqrt(torch.pow(dr, 2).sum(dim=-1) + EPS**2), max=d_clamp)
            loss[batch_index] = loss[batch_index] + torch.mean(d)
        #
        first = last
        #
    return torch.mean(loss / batch.batch_num_nodes())


def loss_f_FAPE_all(
    batch: dgl.DGLGraph, R: torch.Tensor, bb: torch.Tensor, d_clamp: float = 2.0
) -> torch.Tensor:
    def rotate_vector_inv(R, X):
        R_inv = torch.inverse(R)
        return (X[..., None, :] @ R.mT)[..., 0, :]

    first = 0
    loss = torch.zeros(batch.batch_size, device=R.device, dtype=DTYPE)
    for batch_index, data in enumerate(dgl.unbatch(batch)):
        n_residue = data.num_nodes()
        last = first + n_residue
        #
        mask = data.ndata["pdb_atom_mask"] > 0.0
        _R = R[first:last]
        _bb = bb[first:last]
        R_ref = data.ndata["output_xyz"]
        bb_ref = data.ndata["correct_bb"]
        #
        for i in range(n_residue):
            r = rotate_vector_inv(_bb[i, :3], _R - _bb[i, 3])
            r_ref = rotate_vector_inv(bb_ref[i, :3], R_ref - bb_ref[i, 3])
            dr = (r - r_ref)[mask]
            d = torch.clamp(torch.sqrt(torch.pow(dr, 2).sum(dim=-1) + EPS**2), max=d_clamp)
            loss[batch_index] = loss[batch_index] + torch.mean(d)
        #
        first = last
        #
    return torch.mean(loss / batch.batch_num_nodes())


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
    bond_energy = torch.sum(torch.abs(d1 - BOND_LENGTH0) * bonded) / n_bonded
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
    angle_energy = torch.abs(a01 - BOND_ANGLE0[0]) + torch.abs(a12 - BOND_ANGLE0[1])
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
    torsion_energy = torch.sum(torch.abs(d_ang) * bonded) / n_bonded
    return bond_energy * weight_s[0] + angle_energy * weight_s[1] + torsion_energy * weight_s[2]


def loss_f_bonded_energy_aux(batch, R):
    # proline ring closure
    proline = batch.ndata["residue_type"] == PROLINE_INDEX
    if torch.any(proline):
        R_pro_N = R[proline, ATOM_INDEX_N]
        R_pro_CD = R[proline, ATOM_INDEX_PRO_CD]
        d_pro = v_size(R_pro_N - R_pro_CD)
        bond_energy_pro = torch.mean(torch.abs(d_pro - BOND_LENGTH_PROLINE_RING))
        # bond_energy_pro = torch.sum(torch.abs(d_pro - BOND_LENGTH_PROLINE_RING)) / R.size(0)
    else:
        bond_energy_pro = 0.0

    # disulfide bond
    bond_energy_ssbond = 0.0
    for batch_index in range(batch.batch_size):
        data = dgl.slice_batch(batch, batch_index, store_ids=True)
        if not torch.any(data.ndata["ssbond_index"] >= 0):
            continue
        #
        _R = R[data.ndata["_ID"]]
        cys_1_index = data.ndata["ssbond_index"]
        disu = cys_1_index >= 0
        cys_0_index = data.nodes()[disu]
        cys_1_index = cys_1_index[disu]
        R_cys_0 = _R[cys_0_index, ATOM_INDEX_CYS_SG]
        R_cys_1 = _R[cys_1_index, ATOM_INDEX_CYS_SG]
        d_ssbond = v_size(R_cys_1 - R_cys_0)
        bond_energy_ssbond = bond_energy_ssbond + torch.mean(
            torch.abs(d_ssbond - BOND_LENGTH_DISULFIDE)
        )
        # bond_energy_ssbond = bond_energy_ssbond + torch.sum(torch.abs(d_ssbond - BOND_LENGTH_DISULFIDE)) / R.size(0)

    return bond_energy_pro + bond_energy_ssbond


def loss_f_torsion_angle_v0(sc, sc0, sc_ref, mask, norm_weight: float = 0.01):
    sc_cos = torch.cos(sc_ref)
    sc_sin = torch.sin(sc_ref)
    #
    loss_cos = torch.sum(torch.abs(sc[:, :, 0] - sc_cos) * mask)
    loss_sin = torch.sum(torch.abs(sc[:, :, 1] - sc_sin) * mask)

    if sc0 is not None and norm_weight > 0.0:
        norm = v_size(sc0)
        loss_norm = torch.sum(torch.abs(norm - 1.0) * mask)
        loss = loss_cos + loss_sin + loss_norm * norm_weight
    else:
        loss = loss_cos + loss_sin
    loss = loss / sc.size(0)
    return loss


def loss_f_torsion_angle(sc, sc0, sc_ref, mask, norm_weight: float = 0.01):
    sc_cos = torch.cos(sc_ref)
    sc_sin = torch.sin(sc_ref)
    loss = torch.sum((1.0 - ((sc[..., 0] * sc_cos) + (sc[..., 1] * sc_sin))) * mask)
    #
    if sc0 is not None and norm_weight > 0.0:
        norm = v_size(sc0)
        loss_norm = torch.sum(torch.abs(norm - 1.0) * mask)
        loss = loss + loss_norm * norm_weight
    loss = loss / sc.size(0)
    return loss


def loss_f_distance_matrix(R, batch, radius=1.0):
    R_ref = batch.ndata["output_xyz"][:, ATOM_INDEX_CA]
    graph = dgl.radius_graph(R_ref, radius)
    edge_dst, edge_src = graph.edges()
    d_ref = v_size(R_ref[edge_dst] - R_ref[edge_src])
    d = v_size(R[edge_dst, ATOM_INDEX_CA] - R[edge_src, ATOM_INDEX_CA])
    #
    loss = torch.nn.functional.mse_loss(d, d_ref)
    return loss


def loss_f_atomic_clash(R, batch, lj=False):
    energy = 0.0
    for batch_index in range(batch.batch_size):
        data = dgl.slice_batch(batch, batch_index, store_ids=True)
        _R = R[data.ndata["_ID"]]
        #
        for i in range(1, data.num_nodes()):
            curr_residue_type = data.ndata["residue_type"][i]
            prev_residue_type = data.ndata["residue_type"][i - 1]
            mask_i = data.ndata["output_atom_mask"][i] > 0.0
            mask_j = data.ndata["output_atom_mask"][:i] > 0.0
            mask = mask_j[..., None] & mask_i[None, None, :]
            #
            # excluding BB(prev) - BB(curr)
            curr_bb = RIGID_GROUPS_DEP[curr_residue_type] < 3
            if curr_residue_type == PROLINE_INDEX:
                curr_bb[:7] = True  # BB + CD, HD1, HD2
            prev_bb = RIGID_GROUPS_DEP[prev_residue_type] < 3
            bb_pair = prev_bb[:, None] & curr_bb[None, :]
            mask[-1][bb_pair] = False
            #
            ssbond = data.ndata["ssbond_index"][i]
            if ssbond >= 0:
                mask[ssbond, ATOM_INDEX_CYS_SG, ATOM_INDEX_CYS_SG] = False
            #
            dr = _R[:i][..., None, :] - _R[i][None, None, :]
            dist = v_size(dr)[mask]
            #
            radius_i = data.ndata["atomic_radius"][i, :, 0, 1]
            radius_j = data.ndata["atomic_radius"][:i, :, 0, 1]
            radius_sum = (radius_j[..., None] + radius_i[None, None, :])[mask]
            #
            if lj:
                epsilon_i = batch.ndata["atomic_radius"][i, :, 0, 0]
                epsilon_j = batch.ndata["atomic_radius"][:i, :, 0, 0]
                epsilon = torch.sqrt(epsilon_j[..., None] * epsilon_i[None, None, :])[mask]
                #
                x = torch.pow(radius_sum / dist, 6)
                energy_i = epsilon * (x**2 - 2 * x)
            else:
                # radius_sum = radius_sum * 2 ** (-1 / 6)
                energy_i = torch.pow(-torch.clamp(dist - radius_sum, max=0.0), 2)
            energy = energy + energy_i.sum()
    energy = energy / R.size(0)
    return energy


def test():
    from libconfig import BASE
    import libcg
    import functools
    from libdata import PDBset

    base_dir = BASE / "pdb.processed"
    pdblist = base_dir / "loss_test"
    cg_model = libcg.CalphaBasedModel
    #
    train_set = PDBset(
        base_dir,
        pdblist,
        cg_model,
        get_structure_information=True,
    )
    train_loader = dgl.dataloading.GraphDataLoader(
        train_set, batch_size=2, shuffle=False, num_workers=1
    )
    batch = next(iter(train_loader))
    #
    native = dgl.slice_batch(batch, 0)
    model = dgl.slice_batch(batch, 1)
    #
    R_ref = native.ndata["output_xyz"].clone()
    bb_ref = native.ndata["correct_bb"].clone()
    R = model.ndata["output_xyz"].clone()
    bb = model.ndata["correct_bb"].clone()

    # loss = loss_f_bonded_energy_aux(native, R)
    # print(loss)
    # loss = loss_f_bonded_energy_aux(native, R_ref)
    # print(loss)
    # loss = loss_f_atomic_clash(R, model)
    # print(loss)
    # loss = loss_f_atomic_clash(R_ref, native)
    # print(loss)
    loss = loss_f_FAPE_all(native, R, bb)
    print(loss)
    loss = loss_f_FAPE_all(native, R_ref, bb_ref)
    print(loss)


if __name__ == "__main__":
    test()

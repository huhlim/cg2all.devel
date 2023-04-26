#!/usr/bin/env python

import os
import sys
import copy
import torch
import mdtraj
import numpy as np
import dgl

from libconfig import DTYPE, DATA_HOME, EPS
from libdata import resSeq_to_number

from libloss import (
    loss_f_bonded_energy,
    loss_f_bonded_energy_aux,
    loss_f_torsion_energy,
    loss_f_atomic_clash,
    CoarseGrainedGeometryEnergy,
)
from libter import patch_termini
from torch_basics import v_size, torsion_angle
from residue_constants import (
    residue_s,
    MAX_ATOM,
    GLYCINE_INDEX,
    PROLINE_INDEX,
    ATOM_INDEX_N,
    ATOM_INDEX_CA,
    ATOM_INDEX_C,
    ATOM_INDEX_PRO_CD,
    ATOM_INDEX_CYS_CB,
    ATOM_INDEX_CYS_SG,
    par_dihed_s,
)

from openmm.unit import *
from openmm.openmm import *
from openmm.app import *

KE_CONST = 33.2063684  # kcal/mol * ...
GAS_CONST = 8.31446261815  # J/mol/K
GAS_CONST_sq = np.sqrt(0.001 * GAS_CONST)  # sqrt(kJ/mol/K)
Cal_to_J = 4.184
J_to_Cal = 1.0 / Cal_to_J
np.set_printoptions(linewidth=500, edgeitems=100)

# UNITS
# - mass: amu
# - distance: nm
# - time: ps
# - velocity: nm/ps
# - acceleration: nm/ps^2
# - energy: kcal/mol
# - force: kcal/mol/nm


class MDdata(object):
    def __init__(self, pdb_fn, cg_model, r_cg=None, v_cg=None, radius=1.0, dtype=DTYPE):
        super().__init__()
        #
        self.pdb_fn = pdb_fn
        #
        self.radius = radius
        self.dtype = dtype
        #
        self.cg = cg_model(self.pdb_fn, is_all=True)
        #
        if r_cg is None:
            self.r_cg = torch.as_tensor(self.cg.R_cg[0], dtype=self.dtype)
        else:
            self.r_cg = torch.as_tensor(r_cg, dtype=self.dtype)
        self.r_cg.requires_grad = True
        #
        if v_cg is not None:
            self.v_cg = torch.as_tensor(v_cg, dtype=self.dtype)
        #
        self.set_mass()

    def set_mass(self):
        cg_model_name = self.cg.__class__.__name__
        if cg_model_name in ["CalphaBasedModel", "ResidueBasedModel"]:
            self.cg_mass = torch.sum(
                torch.as_tensor(self.cg.atomic_mass, dtype=self.dtype), 1, keepdim=True
            )
        else:
            raise NotImplementedError

    def set_velocities(self, temperature):
        v_cg = torch.randn_like(self.r_cg)
        v_cg = v_cg * GAS_CONST_sq * torch.sqrt(temperature / self.cg_mass[..., None])
        p = torch.sum(v_cg * self.cg_mass[..., None], (0, 1))
        v_cntr = p / torch.sum(self.cg_mass)
        v_cg = v_cg - v_cntr[None, None, :]
        self.v_cg = v_cg

    def convert_to_batch(self, r_cg):
        valid_residue = self.cg.atom_mask_cg[:, 0] > 0.0
        pos = r_cg[valid_residue, :]
        geom_s = self.cg.get_geometry(pos, self.cg.atom_mask_cg, self.cg.continuous[0])
        #
        node_feat = self.cg.geom_to_feature(geom_s, self.cg.continuous, dtype=self.dtype)
        data = dgl.radius_graph(pos[:, 0], self.radius, self_loop=False)
        data.ndata["pos"] = pos[:, 0]
        data.ndata["node_feat_0"] = node_feat["0"][..., None]  # shape=(N, 16, 1)
        data.ndata["node_feat_1"] = node_feat["1"]  # shape=(N, 4, 3)
        #
        edge_src, edge_dst = data.edges()
        data.edata["rel_pos"] = pos[edge_dst, 0] - pos[edge_src, 0]
        #
        data.ndata["chain_index"] = torch.as_tensor(self.cg.chain_index, dtype=torch.long)
        resSeq, resSeqIns = resSeq_to_number(self.cg.resSeq)
        data.ndata["resSeq"] = torch.as_tensor(resSeq, dtype=torch.long)
        data.ndata["resSeqIns"] = torch.as_tensor(resSeqIns, dtype=torch.long)
        data.ndata["residue_type"] = torch.as_tensor(self.cg.residue_index, dtype=torch.long)
        data.ndata["continuous"] = torch.as_tensor(self.cg.continuous[0], dtype=self.dtype)
        data.ndata["output_atom_mask"] = torch.as_tensor(self.cg.atom_mask, dtype=self.dtype)  #
        #
        # aa-specific
        data.ndata["output_xyz"] = torch.as_tensor(self.cg.R[0], dtype=self.dtype)
        data.ndata["heavy_atom_mask"] = torch.as_tensor(self.cg.atom_mask_heavy, dtype=self.dtype)
        data.ndata["atomic_radius"] = torch.as_tensor(self.cg.atomic_radius, dtype=self.dtype)
        data.ndata["atomic_mass"] = torch.as_tensor(self.cg.atomic_mass, dtype=self.dtype)
        data.ndata["atomic_charge"] = torch.as_tensor(self.cg.atomic_charge, dtype=self.dtype)
        #
        ssbond_index = torch.full((data.num_nodes(),), -1, dtype=torch.long)
        for cys_i, cys_j in self.cg.ssbond_s:
            if cys_i < cys_j:  # because of loss_f_atomic_clash
                ssbond_index[cys_j] = cys_i
            else:
                ssbond_index[cys_i] = cys_j
        data.ndata["ssbond_index"] = ssbond_index
        #
        edge_feat = torch.zeros((data.num_edges(), 3), dtype=self.dtype)  # bonded / ssbond / space
        #
        # bonded
        pair_s = [(i - 1, i) for i, cont in enumerate(self.cg.continuous[0]) if cont]
        pair_s = torch.as_tensor(pair_s, dtype=torch.long)
        has_edges = data.has_edges_between(pair_s[:, 0], pair_s[:, 1])
        pair_s = pair_s[has_edges]
        eid = data.edge_ids(pair_s[:, 0], pair_s[:, 1])
        edge_feat[eid, 0] = 1.0
        eid = data.edge_ids(pair_s[:, 1], pair_s[:, 0])
        edge_feat[eid, 0] = 1.0
        #
        # ssbond
        if len(self.cg.ssbond_s) > 0:
            pair_s = torch.as_tensor(self.cg.ssbond_s, dtype=torch.long)
            has_edges = data.has_edges_between(pair_s[:, 0], pair_s[:, 1])
            pair_s = pair_s[has_edges]
            eid = data.edge_ids(pair_s[:, 0], pair_s[:, 1])
            edge_feat[eid, 1] = 1.0
            eid = data.edge_ids(pair_s[:, 1], pair_s[:, 0])
            edge_feat[eid, 1] = 1.0
        #
        # space
        edge_feat[edge_feat.sum(dim=-1) == 0.0, 2] = 1.0
        data.edata["edge_feat_0"] = edge_feat[..., None]

        return data


def loss_f_nonbonded(
    batch: dgl.DGLGraph,
    R: torch.Tensor,
    RIGID_OPs,
    energy_clamp=0.0,
    g_radius=1.4,
):
    # c ~ average number of edges per node
    # time: O(Nxc) vs. O(N^2) for loss_f_atomic_clash_old
    # memory: O(Nxc) vs. O(N) for loss_f_atomic_clash_old
    #   (e.g., 70 MB vs. 20 MB for 855 aa.)
    # this can be approximate if radius is small
    #
    _RIGID_GROUPS_DEP = RIGID_OPs[1][1]
    #
    def get_pairs(data, R, g_radius):
        g = dgl.radius_graph(R, g_radius, self_loop=False)
        #
        edges = g.edges()
        ssbond = torch.zeros_like(edges[0], dtype=bool)
        #
        cys_i = torch.nonzero(data.ndata["ssbond_index"] >= 0)[:, 0]
        if cys_i.size(0) > 0:
            cys_j = data.ndata["ssbond_index"][cys_i]
            try:
                eids = g.edge_ids(cys_i, cys_j)
                ssbond[eids] = True
            except:
                pass
        #
        subset = edges[0] > edges[1]
        edges = (edges[0][subset], edges[1][subset])
        ssbond = ssbond[subset]
        #
        return edges, ssbond

    #
    energy = 0.0
    for batch_index in range(batch.batch_size):
        data = dgl.slice_batch(batch, batch_index, store_ids=True)
        _R = R[data.ndata["_ID"]]
        (i, j), ssbond = get_pairs(data, _R[:, ATOM_INDEX_CA], g_radius=g_radius)
        #
        mask_i = data.ndata["output_atom_mask"][i] > 0.0
        mask_j = data.ndata["output_atom_mask"][j] > 0.0
        mask = mask_j[:, :, None] & mask_i[:, None, :]
        #
        # find consecutive residue pairs (i == j + 1)
        y = i == j + 1
        curr_residue_type = data.ndata["residue_type"][i[y]]
        prev_residue_type = data.ndata["residue_type"][j[y]]
        curr_bb = _RIGID_GROUPS_DEP[curr_residue_type] < 3
        curr_bb[curr_residue_type == PROLINE_INDEX, :7] = True
        prev_bb = _RIGID_GROUPS_DEP[prev_residue_type] < 3
        bb_pair = prev_bb[:, :, None] & curr_bb[:, None, :]
        mask[y] = mask[y] & (~bb_pair)
        #
        mask[ssbond, ATOM_INDEX_CYS_CB:, ATOM_INDEX_CYS_CB:] = False
        mask[ssbond, ATOM_INDEX_CYS_SG, ATOM_INDEX_CA] = False
        mask[ssbond, ATOM_INDEX_CA, ATOM_INDEX_CYS_SG] = False
        #
        dr = _R[j][:, :, None] - _R[i][:, None, :]
        dist = v_size(dr)
        #
        charge_i = data.ndata["atomic_charge"][i, :]
        charge_j = data.ndata["atomic_charge"][j, :]
        charge = charge_j[:, :, None] * charge_i[:, None, :] * mask
        #
        epsilon_i = data.ndata["atomic_radius"][i, :, 0, 0]
        epsilon_j = data.ndata["atomic_radius"][j, :, 0, 0]
        epsilon = torch.sqrt(epsilon_j[:, :, None] * epsilon_i[:, None, :]) * mask
        #
        radius_i = data.ndata["atomic_radius"][i, :, 0, 1]
        radius_j = data.ndata["atomic_radius"][j, :, 0, 1]
        radius_sum = radius_j[:, :, None] + radius_i[:, None, :]
        #
        dist0 = torch.clamp(radius_sum * 0.8, min=EPS)
        dist = (torch.nn.functional.elu((dist / dist0) - 1.0, alpha=0.2) + 1.0) * dist0
        dist = torch.clamp(dist, min=EPS)
        x6 = torch.pow(radius_sum / dist, 6)
        vdw_ij = epsilon * (x6.square() - 2.0 * x6)
        #
        elec_ij = KE_CONST * charge / dist / (40.0 * dist)
        #
        energy = energy + vdw_ij.sum() + elec_ij.sum()
    return energy


class BackboneTorsionEnergy(object):
    def __init__(self, data, device="cpu"):
        for residue in residue_s.values():
            residue.find_1_N_pair(4)
        #
        bonded = []
        atom_index = 0
        for residue_name in data.cg.residue_name:
            ref_res = residue_s[residue_name]
            _bonded = {}
            for i in [2, 3, 4]:
                X = np.array(copy.deepcopy(ref_res.bonded_pair_s[i]), dtype=int)
                _bonded[i] = (X + atom_index).tolist()
            bonded.append(_bonded)
            atom_index += MAX_ATOM
        #
        for i_res, continuous in enumerate(data.cg.continuous[1]):
            if not continuous:
                continue
            #
            atom_index_i = i_res * MAX_ATOM
            atom_index_j = (i_res + 1) * MAX_ATOM
            #
            bonded[i_res][2].append([atom_index_i + ATOM_INDEX_C, atom_index_j + ATOM_INDEX_N])
            BackboneTorsionEnergy.update_bonded_between_residues(
                bonded[i_res], bonded[i_res + 1], 3
            )
            BackboneTorsionEnergy.update_bonded_between_residues(
                bonded[i_res], bonded[i_res + 1], 4
            )
        #
        atom_index = []
        param_s = []
        for X in bonded:
            for index in X[4]:
                if index[1] % MAX_ATOM not in [ATOM_INDEX_N, ATOM_INDEX_CA, ATOM_INDEX_C]:
                    continue
                if index[2] % MAX_ATOM not in [ATOM_INDEX_N, ATOM_INDEX_CA, ATOM_INDEX_C]:
                    continue

                type_s = []
                for i in index:
                    i_res = i // MAX_ATOM
                    i_atm = i % MAX_ATOM
                    residue_name = data.cg.residue_name[i_res]
                    atom_type = residue_s[residue_name].atom_type_s[i_atm]
                    type_s.append(atom_type)
                type_s = tuple(type_s)
                type_rev_s = type_s[::-1]
                type_x = tuple(["X", type_s[1], type_s[2], "X"])
                type_rev_x = type_x[::-1]
                #
                if type_s in par_dihed_s:
                    par = par_dihed_s[type_s]
                elif type_rev_s in par_dihed_s:
                    par = par_dihed_s[type_rev_s]
                elif type_x in par_dihed_s:
                    par = par_dihed_s[type_x]
                elif type_rev_x in par_dihed_s:
                    par = par_dihed_s[type_rev_x]
                else:
                    raise KeyError(type_s)
                #
                for p in par:
                    if p[0] == 0.0:
                        continue
                    atom_index.append(index)
                    param_s.append(p)
        #
        self.atom_index = torch.as_tensor(atom_index, dtype=torch.long, device=device)
        self.param_s = torch.as_tensor(param_s, dtype=DTYPE, device=device).T

    #
    def __call__(self, R):
        r = R.view((-1, 3))[self.atom_index]
        t_ang = torsion_angle(r)
        enr = self.param_s[0] * (1.0 + torch.cos(self.param_s[1] * t_ang - self.param_s[2]))
        return enr.sum()

    @staticmethod
    def update_bonded_between_residues(curr_res, next_res, N):
        bond_s = curr_res[2] + next_res[2]
        concat_s = curr_res[N - 1] + next_res[N - 1]
        exist_s = curr_res[N] + next_res[N]
        #
        pair_s = []
        for prev in concat_s:
            for bond in bond_s:
                pair = None
                if prev[-1] == bond[0] and prev[-2] != bond[1]:
                    pair = prev + [bond[1]]
                elif prev[-1] == bond[1] and prev[-2] != bond[0]:
                    pair = prev + [bond[0]]
                elif prev[0] == bond[1] and prev[1] != bond[0]:
                    pair = [bond[0]] + prev
                elif prev[0] == bond[0] and prev[1] != bond[1]:
                    pair = [bond[1]] + prev
                #
                if pair is None:
                    continue
                if pair in pair_s or pair[::-1] in pair_s:
                    continue
                if pair in exist_s or pair[::-1] in exist_s:
                    continue
                pair_s.append(pair)
        #
        curr_res[N].extend(pair_s)


class CMAPTorsionEnergy(object):
    def __init__(self, data, device="cpu"):
        self.read_CMAP_data(device)
        #
        self.type_s = []
        self.atom_index = []
        for i_res, res_type in enumerate(data.cg.residue_index):
            has_prev = data.cg.continuous[0][i_res]
            has_next = data.cg.continuous[1][i_res]
            if (not has_prev) or (not has_next):
                continue
            #
            if res_type == PROLINE_INDEX:
                curr_res = 1
            elif res_type == GLYCINE_INDEX:
                curr_res = 2
            else:
                curr_res = 0
            if data.cg.residue_index[i_res + 1] == PROLINE_INDEX:
                next_res = 1
            else:
                next_res = 0
            cmap_type = curr_res * 2 + next_res
            self.type_s.append(cmap_type)
            #
            index0 = (i_res - 1) * MAX_ATOM
            index1 = i_res * MAX_ATOM
            index2 = (i_res + 1) * MAX_ATOM
            phi_index = [
                index0 + ATOM_INDEX_C,
                index1 + ATOM_INDEX_N,
                index1 + ATOM_INDEX_CA,
                index1 + ATOM_INDEX_C,
            ]
            psi_index = [
                index1 + ATOM_INDEX_N,
                index1 + ATOM_INDEX_CA,
                index1 + ATOM_INDEX_C,
                index2 + ATOM_INDEX_N,
            ]
            self.atom_index.append((phi_index, psi_index))
            #
        self.type_s = torch.as_tensor(self.type_s, dtype=torch.long, device=device)
        self.atom_index = torch.as_tensor(self.atom_index, dtype=torch.long, device=device)

    def __call__(self, R):
        cmap = self.cmap_s[self.type_s]
        #
        r = R.view((-1, 3))[self.atom_index]
        t_ang = torsion_angle(r)
        xy = t_ang / 12.0 - 1.0
        enr = torch.nn.functional.grid_sample(
            cmap[:, None, :, :], xy[:, None, None, :], align_corners=True, mode="bicubic"
        )
        return enr.sum()

    def read_CMAP_data(self, device):
        cmap_type_s = []
        cmap_s = []
        with open(DATA_HOME / "cmap.dat") as fp:
            for line in fp:
                line = line.strip()
                if line.startswith("TYPE"):
                    x = line.strip().split()
                    cmap_type_s.append(tuple(x[1:]))
                    cmap = []
                    cmap_s.append(cmap)
                elif line.startswith("!") or line == "":
                    continue
                else:
                    cmap.extend(line.split())
        #
        cmap_s = np.array(cmap_s, dtype=float).reshape((-1, 24, 24))
        _cmap = np.zeros((len(cmap_s), 25, 25))
        _cmap[:, :24, :24] = cmap_s
        _cmap[:, -1, :24] = cmap_s[:, 0, :]
        _cmap[:, :24, -1] = cmap_s[:, :, 0]
        _cmap[:, -1, -1] = cmap_s[:, 0, 0]
        self.cmap_s = torch.as_tensor(_cmap, dtype=torch.float, device=device)
        self.cmap_type_s = cmap_type_s


class DCDReporter(mdtraj.reporters.DCDReporter):
    def __init__(self, file, interval):
        super().__init__(file, interval)

    def report(self, conf):
        args = (conf.xyz[0] * 10.0,)
        kwargs = {}
        self._traj_file.write(*args, **kwargs)
        if hasattr(self._traj_file, "flush"):
            self._traj_file.flush()


class MolecularMechanicsForceField(object):
    def __init__(self, data, model, model_type, device, **kwargs):
        self.RIGID_OPs = model.RIGID_OPs
        self.TORSION_PARs = model.TORSION_PARs
        self.geometry_energy = CoarseGrainedGeometryEnergy(model_type, device)
        self.backbone_torsion = BackboneTorsionEnergy(data, device=device)
        self.cmap_energy = CMAPTorsionEnergy(data, device=device)
        self.use_constraint = model_type == "CalphaBasedModel"
        #
        self.weight = {}
        self.weight["bond"] = kwargs.get("bond_weight", 0.0)
        self.weight["torsion"] = kwargs.get("torsion_weight", 0.0)
        self.weight["backbone"] = kwargs.get("backbone_weight", 1.0)
        self.weight["nb"] = kwargs.get("nb_weight", 1.0)
        self.weight["cg"] = kwargs.get("cg_weight", 1.0)

    def __call__(self, batch, ret):
        R = ret["R"]
        #
        loss = {}
        if self.weight.get("bond", 0.0) > 0.0:
            loss["bond"] = loss_f_bonded_energy(batch, R) + loss_f_bonded_energy_aux(batch, R)
            loss["bond"] = loss["bond"] * R.size(0)
        if self.weight.get("torsion", 0.0) > 0.0:
            loss["torsion"] = loss_f_torsion_energy(batch, R, ret["ss"], self.TORSION_PARs)
            loss["torsion"] = loss["torsion"] * R.size(0)
        if self.weight.get("backbone", 0.0) > 0.0:
            loss["backbone"] = self.backbone_torsion(R)
            loss["backbone"] = loss["backbone"] + self.cmap_energy(R)
        #
        if self.use_constraint:
            cg_bonded_weight = [0.0, 1.0]
        else:
            cg_bonded_weight = [1.0, 1.0]

        loss["nb"] = loss_f_nonbonded(batch, R, self.RIGID_OPs)
        loss["cg"] = self.geometry_energy.eval_bonded(
            batch, weight=cg_bonded_weight
        ) + self.geometry_energy.eval_vdw(batch)
        #
        loss_sum = 0.0
        for name, value in loss.items():
            loss_sum = loss_sum + value * self.weight[name]
        return loss_sum


class Constraint(object):
    def __init__(self, data, dtype=DTYPE):
        self.max_iteration = 100
        self.tolerance = 0.001
        self.dtype = dtype
        #
        self.inverse_mass = 1.0 / data.cg_mass[..., 0]
        cg_model_name = data.cg.__class__.__name__
        if cg_model_name not in ["CalphaBasedModel", "ResidueBasedModel"]:
            raise NotImplementedError
        #
        self.set_constraint_CCMA(data)

    def set_constraint_CCMA(self, data, element_cutoff=0.02):
        self.angle = 1.851863  # rad
        self.d0 = 0.380573  # nm
        #
        pair_s = []
        dist0 = []
        for i, has_prev in enumerate(data.cg.continuous[0]):
            if has_prev:
                pair_s.append((i - 1, i))
                dist0.append(self.d0)
        n_pair = len(pair_s)
        matrix = np.eye(n_pair)
        reduced_mass = []
        for j, pair_j in enumerate(pair_s):
            inverse_mass = self.inverse_mass[pair_j[0]], self.inverse_mass[pair_j[1]]
            reduced_mass.append(0.5 / (inverse_mass[0] + inverse_mass[1]))
            for k, pair_k in enumerate(pair_s):
                if j == k:
                    continue
                if pair_j[1] == pair_k[0]:
                    bead_a = pair_j[0]
                    bead_b = pair_j[1]
                    bead_c = pair_k[1]
                    scale = inverse_mass[1] / (inverse_mass[0] + inverse_mass[1])
                elif pair_j[0] == pair_k[1]:
                    bead_a = pair_j[1]
                    bead_b = pair_j[0]
                    bead_c = pair_k[0]
                    scale = inverse_mass[0] / (inverse_mass[0] + inverse_mass[1])
                else:
                    continue
                matrix[j, k] = scale * np.cos(self.angle)
        #
        matrix_inv = np.linalg.inv(matrix)
        matrix_inv[matrix_inv < element_cutoff] = 0.0
        #
        self.n_pair = n_pair
        self.pair_s = torch.as_tensor(pair_s, dtype=torch.long)
        self.dist0 = torch.as_tensor(dist0, dtype=self.dtype)
        self.reduced_mass = torch.as_tensor(reduced_mass, dtype=self.dtype)
        self.matrix = torch.as_tensor(matrix_inv, dtype=self.dtype)

    def apply(self, x, xp, is_velocity):
        if not is_velocity:
            tolerance = (
                (1.0 - 2.0 * self.tolerance + self.tolerance**2),
                (1.0 + 2.0 * self.tolerance + self.tolerance**2),
            )
            lowerTol = tolerance[0] * self.dist0.square()
            upperTol = tolerance[1] * self.dist0.square()
        #
        II = self.pair_s[:, 0]
        JJ = self.pair_s[:, 1]
        #
        r_ij = x[II] - x[JJ]
        d_ij2 = r_ij.square().sum(dim=-1)
        #
        for iteration in range(self.max_iteration):
            rp_ij = xp[II] - xp[JJ]
            if is_velocity:
                rrpr = (rp_ij * r_ij).sum(dim=-1)
                delta = 2.0 * self.reduced_mass * rrpr / d_ij2
                n_converged = (torch.abs(delta) <= self.tolerance).type(torch.long).sum()
            else:
                rp2 = rp_ij.square().sum(dim=-1)
                diff = rp2 - self.dist0**2
                rrpr = (rp_ij * r_ij).sum(dim=-1)
                delta = self.reduced_mass * diff / rrpr
                n_converged = ((rp2 >= lowerTol) & (rp2 <= upperTol)).type(torch.long).sum()
            if n_converged == self.n_pair:
                break
            #
            scale = torch.matmul(self.matrix, delta)
            dr = scale[:, None] * r_ij
            xp[II] = xp[II] - dr * self.inverse_mass[II][:, None]
            xp[JJ] = xp[JJ] + dr * self.inverse_mass[JJ][:, None]
        return xp

    def update_position(self, x, v):
        return self.apply(x, x.clone().detach(), False)

    def update_velocity(self, x, v):
        return self.apply(x, v.clone().detach(), True)


class LangevinIntegratorTorch(object):
    def __init__(self, time_step, temperature, gamma, constraint=None):
        self.dt = time_step
        self.temperature = temperature
        self.gamma = gamma
        #
        self.alpha = np.exp(-self.gamma * self.dt)
        self.beta = np.sqrt(1.0 - np.exp(-2.0 * self.gamma * self.dt))
        #
        self.splitting = "V R O R V".split()
        # self.splitting = "V R R R O R R R V".split()
        self.n_ORV = {opr: float(self.splitting.count(opr)) for opr in ["O", "R", "V"]}

        self.constraint = constraint

    def update_R(self, x, v, m, f):
        dx = self.dt / self.n_ORV["R"] * v
        x.add_(dx)
        #
        if self.constraint:
            x1 = x.clone().detach()
            x = self.constraint.update_position(x, None)
            dx = x - x1
            dv = dx / (self.dt / self.n_ORV["R"])
            v.add_(dv)
            v = self.constraint.update_velocity(x, v)
        return x, v

    def update_V(self, x, v, m, f):
        dv = self.dt / self.n_ORV["V"] * f / m[..., None]
        v.add_(dv)
        #
        if self.constraint:
            v = self.constraint.update_velocity(x, v)
        return x, v

    def update_O(self, x, v, m, f):
        R = GAS_CONST_sq * torch.sqrt(self.temperature / m[..., None])
        v.mul_(self.alpha)
        v.add_(self.beta * R * torch.randn_like(v))
        #
        if self.constraint:
            v = self.constraint.update_velocity(x, v)
        return x, v

    @staticmethod
    def get_kinetic_energy(m, v):
        ke = torch.sum(0.5 * m[..., None] * v**2).detach().item()
        return ke

    @staticmethod
    def get_temperature(m, v):
        # only for 1-bead cg model
        ke = LangevinIntegratorTorch.get_kinetic_energy(m, v)
        T = ke * 2.0 / 3.0 / m.size(0) / GAS_CONST * 1000.0
        return T, ke


class MDsimulator(object):
    def __init__(self, model, force_field, integrator, device):
        self.model = model
        self.force_field = force_field
        self.integrator = integrator
        self.device = device

    def step(self, data, n_step):
        mask = torch.as_tensor(data.cg.atom_mask_cg > 0.0)
        mass = data.cg_mass[mask]
        #
        for _ in range(n_step):
            for update in self.integrator.splitting:
                r_cg = data.r_cg[mask]
                v_cg = data.v_cg[mask]
                #
                if update == "R":
                    with torch.no_grad():
                        r_cg, v_cg = self.integrator.update_R(r_cg, v_cg, mass, None)
                        _r_cg = data.r_cg.clone().detach()
                        _r_cg[mask] = r_cg
                        data.r_cg.copy_(_r_cg)
                        data.r_cg.grad = None
                    grad_calculated = False
                    #
                elif update == "V":
                    if data.r_cg.grad is None:
                        batch = data.convert_to_batch(data.r_cg).to(self.device)
                        ret = self.model.forward(batch)[0]
                        energy = self.force_field(batch, ret)
                        energy.backward()
                    #
                    with torch.no_grad():
                        force_cg = -data.r_cg.grad.data[mask] * Cal_to_J
                        force_size_clip = 1e4
                        force_size = v_size(force_cg)
                        clip = force_size > force_size_clip
                        force_cg[clip] = (
                            force_cg[clip] * force_size_clip / force_size[clip][:, None]
                        )
                        r_cg, v_cg = self.integrator.update_V(r_cg, v_cg, mass, force_cg)
                    #
                elif update == "O":
                    with torch.no_grad():
                        r_cg, v_cg = self.integrator.update_O(r_cg, v_cg, mass, None)
                #
                if self.integrator.constraint or update != "R":
                    data.v_cg[mask] = v_cg
                    # temperature = self.integrator.get_temperature(mass, v_cg)
        #
        temperature, kinetic_energy = self.integrator.get_temperature(mass, v_cg)
        return temperature, energy, kinetic_energy


def main():
    import libcg

    pdb_fn = "../cg2all.md/pdb/1ubq_A.pdb"
    cg_model = libcg.ResidueBasedModel

    data = MDdata(pdb_fn, cg_model)
    # bb_enr = BackboneTorsionEnergy(data)
    # batch = data.convert_to_batch(data.r_cg)
    # print(bb_enr(batch, torch.as_tensor(data.cg.R[0])))
    cmap_enr = CMAPTorsionEnergy(data)
    r = torch.tensor(data.cg.R[0], dtype=DTYPE, requires_grad=True)
    cmap_enr(r)


if __name__ == "__main__":
    main()

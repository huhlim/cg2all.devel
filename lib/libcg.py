#!/usr/bin/env python

import mdtraj
import numpy as np
import torch
import dgl

from libconfig import DTYPE
from libpdb import PDB
from sklearn.decomposition import PCA
from torch_basics import v_size, inner_product, torsion_angle, one_hot_encoding, acos_safe
from residue_constants import (
    MAX_RESIDUE_TYPE,
    ATOM_INDEX_CA,
    ATOM_INDEX_N,
    ATOM_INDEX_C,
    read_martini_topology,
)


class ResidueBasedModel(PDB):
    MAX_BEAD = 1
    n_node_scalar = 17
    n_node_vector = 4
    n_edge_scalar = 3
    n_edge_vector = 0

    def __init__(self, pdb_fn, dcd_fn=None, **kwarg):
        super().__init__(pdb_fn, dcd_fn, **kwarg)
        self.max_bead_type = MAX_RESIDUE_TYPE
        self.convert_to_cg()

    def convert_to_cg(self):
        self.top_cg = self.top.subset(self.top.select("name CA"))
        self.bead_index = self.residue_index[:, None]  # deprecated
        #
        self.R_cg = np.zeros((self.n_frame, self.n_residue, 1, 3))
        self.atom_mask_cg = np.zeros((self.n_residue, 1), dtype=float)
        #
        mass_weighted_R = self.R * self.atomic_mass[None, ..., None]
        R_cg = mass_weighted_R.sum(axis=-2) / self.atomic_mass.sum(axis=-1)[None, ..., None]
        #
        self.R_cg = R_cg[..., None, :]
        self.atom_mask_cg = self.atom_mask_pdb[:, (ATOM_INDEX_CA,)]

    @staticmethod
    def convert_to_cg_tensor(r: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        mass_weighted_R = r * mass[..., None]
        R_cg = (mass_weighted_R.sum(dim=1) / mass.sum(dim=1)[..., None])[..., None, :]
        return R_cg

    def write_cg(self, R, pdb_fn, dcd_fn=None):
        mask = np.where(self.atom_mask_cg)
        xyz = R[:, mask[0], mask[1], :]
        #
        traj = mdtraj.Trajectory(xyz[:1], self.top_cg)
        traj.save(pdb_fn)
        #
        if dcd_fn is not None:
            traj = mdtraj.Trajectory(xyz, self.top_cg)
            traj.save(dcd_fn)

    @staticmethod
    def get_geometry(r: torch.Tensor, continuous: torch.Tensor):
        device = r.device
        #
        not_defined = continuous == 0.0
        geom_s = {}
        #
        # n_neigh
        n_neigh = torch.zeros(r.shape[0], dtype=DTYPE, device=device)
        graph = dgl.radius_graph(r, 1.0)
        n_neigh = graph.in_degrees(graph.nodes())
        geom_s["n_neigh"] = n_neigh[:, None]

        # bond vectors
        geom_s["bond_length"] = {}
        geom_s["bond_vector"] = {}
        for shift in [1, 2]:
            dr = torch.zeros((r.shape[0] + shift, 3), dtype=DTYPE, device=device)
            b_len = torch.zeros(r.shape[0] + shift, dtype=DTYPE, device=device)
            #
            dr[shift:-shift] = r[:-shift, :] - r[shift:, :]
            b_len[shift:-shift] = v_size(dr[shift:-shift])
            #
            dr[shift:-shift] /= b_len[shift:-shift, None]
            b_len = torch.clamp(b_len, max=1.0)
            #
            for s in range(shift):
                dr[s : -shift + s][not_defined] = 0.0
                b_len[s : -shift + s][not_defined] = 1.0
            #
            geom_s["bond_length"][shift] = (b_len[:-shift], b_len[shift:])
            geom_s["bond_vector"][shift] = (dr[:-shift], -dr[shift:])

        # bond angles
        v1 = geom_s["bond_vector"][1][0]
        v2 = geom_s["bond_vector"][1][1]
        cosine = inner_product(v1, v2)
        sine = 1.0 - cosine**2
        cosine[not_defined] = 0.0
        sine[not_defined] = 0.0
        cosine[:-1][not_defined[1:]] = 0.0
        sine[:-1][not_defined[1:]] = 0.0
        cosine[-1] = 0.0
        sine[-1] = 0.0
        geom_s["bond_angle"] = (cosine, sine)

        # dihedral angles
        R = torch.stack([r[0:-3], r[1:-2], r[2:-1], r[3:]]).reshape(-1, 4, 3)
        t_ang = torsion_angle(R)
        cosine = torch.cos(t_ang)
        sine = torch.sin(t_ang)
        for i in range(3):
            cosine[: -(i + 1)][not_defined[i + 1 : -3]] = 0.0
            sine[: -(i + 1)][not_defined[i + 1 : -3]] = 0.0
        sc = torch.zeros((r.shape[0], 4, 2), device=device)
        for i in range(4):
            sc[i : i + cosine.shape[0], i, 0] = cosine
            sc[i : i + sine.shape[0], i, 1] = sine
        geom_s["dihedral_angle"] = sc
        return geom_s

    @staticmethod
    def geom_to_feature(geom_s, continuous: torch.Tensor, dtype=DTYPE) -> torch.Tensor:
        # features for each residue
        f_in = {"0": [], "1": []}
        #
        # 0d
        f_in["0"].append(geom_s["n_neigh"])  # 1
        f_in["0"].append(torch.as_tensor(continuous.T, dtype=dtype))  # 2
        #
        f_in["0"].append(geom_s["bond_length"][1][0][:, None])  # 4
        f_in["0"].append(geom_s["bond_length"][1][1][:, None])
        f_in["0"].append(geom_s["bond_length"][2][0][:, None])
        f_in["0"].append(geom_s["bond_length"][2][1][:, None])
        #
        f_in["0"].append(geom_s["bond_angle"][0][:, None])  # 2
        f_in["0"].append(geom_s["bond_angle"][1][:, None])
        #
        f_in["0"].append(geom_s["dihedral_angle"].reshape(-1, 8))  # 8
        #
        f_in["0"] = torch.as_tensor(torch.cat(f_in["0"], axis=1), dtype=dtype)  # 17
        #
        # 1d: unit vectors from adjacent residues to the current residue
        f_in["1"].append(geom_s["bond_vector"][1][0])
        f_in["1"].append(geom_s["bond_vector"][1][1])
        f_in["1"].append(geom_s["bond_vector"][2][0])
        f_in["1"].append(geom_s["bond_vector"][2][1])
        f_in["1"] = torch.as_tensor(torch.stack(f_in["1"], axis=1), dtype=dtype)  # 4
        #
        return f_in


class CalphaBasedModel(ResidueBasedModel):
    def __init__(self, pdb_fn, dcd_fn=None, **kwarg):
        super().__init__(pdb_fn, dcd_fn, **kwarg)

    def convert_to_cg(self):
        self.top_cg = self.top.subset(self.top.select("name CA"))
        self.bead_index = self.residue_index[:, None]  # deprecated
        #
        self.R_cg = self.R[:, :, (ATOM_INDEX_CA,), :]
        self.atom_mask_cg = self.atom_mask_pdb[:, (ATOM_INDEX_CA,)]

    @staticmethod
    def convert_to_cg_tensor(r: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        R_cg = r[:, (ATOM_INDEX_CA,), :]
        return R_cg


class Martini(PDB):
    MAX_BEAD = 5
    n_node_scalar = 17
    n_node_vector = 8
    n_edge_scalar = 3
    n_edge_vector = 0

    def __init__(self, pdb_fn, dcd_fn=None, martini_top=None):
        assert martini_top is not None
        #
        super().__init__(pdb_fn, dcd_fn)
        self.convert_to_cg(martini_top)

    def create_top_cg(self, martini_top):
        top = self.top.subset(self.top.select("name CA"))
        serial = 0
        for residue in top.residues:
            bb = residue.atom(0)
            for atom in residue.atoms:
                serial += 1
                atom.serial = serial
                atom.name = "BB"
            #
            n_sc = max(martini_top[self.residue_index[residue.index]])
            for i in range(n_sc):
                top.add_atom(f"SC{i+1}", bb.element, residue)
            serial += n_sc
        return top

    def convert_to_cg(self, martini_top):
        self.top_cg = self.create_top_cg(martini_top)
        #
        self.R_cg = np.zeros((self.n_frame, self.n_residue, self.MAX_BEAD, 3))
        self.atom_mask_cg = np.zeros((self.n_residue, self.MAX_BEAD), dtype=float)
        #
        for i_res in range(self.n_residue):
            index = martini_top[self.residue_index[i_res]]
            #
            mass_weighted_R = self.R[:, i_res] * self.atomic_mass[None, i_res, :, None]
            mass_weighted_R[:, index == -1] = 0.0
            for i_frame in range(self.n_frame):
                np.add.at(self.R_cg[i_frame, i_res], index, mass_weighted_R[i_frame])
                #
            mass_sum = np.zeros(self.MAX_BEAD)
            mass = self.atomic_mass[i_res].copy()
            mass[index == -1] = 0.0
            np.add.at(mass_sum, index, mass)
            #
            self.R_cg[:, i_res] /= np.maximum(EPS, mass_sum[None, :, None])
            self.atom_mask_cg[i_res, mass_sum > EPS] = 1.0

    def write_cg(self, R, pdb_fn, dcd_fn=None):
        mask = np.where(self.atom_mask_cg)
        xyz = R[:, mask[0], mask[1], :]
        #
        traj = mdtraj.Trajectory(xyz[:1], self.top_cg)
        traj.save(pdb_fn)
        #
        if dcd_fn is not None:
            traj = mdtraj.Trajectory(xyz, self.top_cg)
            traj.save(dcd_fn)

    @staticmethod
    def get_geometry(_r: torch.Tensor, _mask: torch.Tensor, continuous: torch.Tensor):
        device = _r.device
        r = _r[:, 0]  # BB
        r_sc = _r[:, 1:]  # SC
        #
        not_defined = continuous == 0.0
        geom_s = {}
        #
        # n_neigh
        n_neigh = torch.zeros(r.shape[0], dtype=DTYPE, device=device)
        graph = dgl.radius_graph(r, 1.0)
        n_neigh = graph.in_degrees(graph.nodes())
        geom_s["n_neigh"] = n_neigh[:, None]

        # BB --> SC
        geom_s["sc_vector"] = (r_sc - r[:, None, :]) * _mask[:, 1:, None]

        # bond vectors
        geom_s["bond_length"] = {}
        geom_s["bond_vector"] = {}
        for shift in [1, 2]:
            dr = torch.zeros((r.shape[0] + shift, 3), dtype=DTYPE, device=device)
            b_len = torch.zeros(r.shape[0] + shift, dtype=DTYPE, device=device)
            #
            dr[shift:-shift] = r[:-shift, :] - r[shift:, :]
            b_len[shift:-shift] = v_size(dr[shift:-shift])
            #
            dr[shift:-shift] /= b_len[shift:-shift, None]
            b_len = torch.clamp(b_len, max=1.0)
            #
            for s in range(shift):
                dr[s : -shift + s][not_defined] = 0.0
                b_len[s : -shift + s][not_defined] = 1.0
            #
            geom_s["bond_length"][shift] = (b_len[:-shift], b_len[shift:])
            geom_s["bond_vector"][shift] = (dr[:-shift], -dr[shift:])

        # bond angles
        v1 = geom_s["bond_vector"][1][0]
        v2 = geom_s["bond_vector"][1][1]
        cosine = inner_product(v1, v2)
        sine = 1.0 - cosine**2
        cosine[not_defined] = 0.0
        sine[not_defined] = 0.0
        cosine[:-1][not_defined[1:]] = 0.0
        sine[:-1][not_defined[1:]] = 0.0
        cosine[-1] = 0.0
        sine[-1] = 0.0
        geom_s["bond_angle"] = (cosine, sine)

        # dihedral angles
        R = torch.stack([r[0:-3], r[1:-2], r[2:-1], r[3:]]).reshape(-1, 4, 3)
        t_ang = torsion_angle(R)
        cosine = torch.cos(t_ang)
        sine = torch.sin(t_ang)
        for i in range(3):
            cosine[: -(i + 1)][not_defined[i + 1 : -3]] = 0.0
            sine[: -(i + 1)][not_defined[i + 1 : -3]] = 0.0
        sc = torch.zeros((r.shape[0], 4, 2), device=device)
        for i in range(4):
            sc[i : i + cosine.shape[0], i, 0] = cosine
            sc[i : i + sine.shape[0], i, 1] = sine
        geom_s["dihedral_angle"] = sc
        return geom_s

    @staticmethod
    def geom_to_feature(geom_s, continuous: torch.Tensor, dtype=DTYPE) -> torch.Tensor:
        # features for each residue
        f_in = {"0": [], "1": []}
        #
        # 0d
        f_in["0"].append(geom_s["n_neigh"])  # 1
        f_in["0"].append(torch.as_tensor(continuous.T, dtype=dtype))  # 2
        #
        f_in["0"].append(geom_s["bond_length"][1][0][:, None])  # 4
        f_in["0"].append(geom_s["bond_length"][1][1][:, None])
        f_in["0"].append(geom_s["bond_length"][2][0][:, None])
        f_in["0"].append(geom_s["bond_length"][2][1][:, None])
        #
        f_in["0"].append(geom_s["bond_angle"][0][:, None])  # 2
        f_in["0"].append(geom_s["bond_angle"][1][:, None])
        #
        f_in["0"].append(geom_s["dihedral_angle"].reshape(-1, 8))  # 8
        #
        f_in["0"] = torch.as_tensor(torch.cat(f_in["0"], axis=1), dtype=dtype)  # 17
        #
        # 1d: unit vectors from adjacent residues to the current residue
        f_in["1"].append(geom_s["bond_vector"][1][0][:,None,:])
        f_in["1"].append(geom_s["bond_vector"][1][1][:,None,:])
        f_in["1"].append(geom_s["bond_vector"][2][0][:,None,:])
        f_in["1"].append(geom_s["bond_vector"][2][1][:,None,:])
        f_in["1"].append(geom_s["sc_vector"])
        f_in["1"] = torch.as_tensor(torch.stack(f_in["1"], axis=1), dtype=dtype)  # 4
        #
        return f_in


def get_residue_center_of_mass(r: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    # r: (n_residue, MAX_ATOM, 3)
    # mass: (n_residue, MAX_ATOM)
    mass_weighted_R = r * mass[..., None]
    cntr = mass_weighted_R.sum(dim=1) / mass.sum(dim=1)[..., None]
    return cntr


def get_backbone_angles(R):
    r_N = R[:, ATOM_INDEX_N]
    r_CA = R[:, ATOM_INDEX_CA]
    r_C = R[:, ATOM_INDEX_C]
    #
    R_phi = torch.stack([r_C[:-1], r_N[1:], r_CA[1:], r_C[1:]], dim=1)
    phi = torsion_angle(R_phi)
    R_psi = torch.stack([r_N[:-1], r_CA[:-1], r_C[:-1], r_N[1:]], dim=1)
    psi = torsion_angle(R_psi)
    #
    tor_s = torch.zeros((R.size(0), 3, 2), device=R.device)
    tor_s[1:, 0, 0] = phi
    tor_s[:-1, 0, 1] = psi
    tor_s[:-1, 1, :] = tor_s[1:, 0]
    tor_s[1:, 2, :] = tor_s[:-1, 0]
    #
    out = torch.cat([torch.cos(tor_s), torch.sin(tor_s)], dim=-1).view(-1, 12)
    return out


def main():
    pdb = Martini("martini/1ab1_A.pdb")
    print(pdb.MAX_BEAD)
    return
    r_cg = torch.as_tensor(pdb.R_cg[0], dtype=DTYPE)
    pos = r_cg[pdb.atom_mask_cg > 0.0, :]
    pdb.get_geometry(pos, pdb.continuous[0])


if __name__ == "__main__":
    main()

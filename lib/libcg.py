#!/usr/bin/env python

# %%
import mdtraj
import numpy as np
import torch
import torch_cluster

from libconfig import DTYPE
from libpdb import PDB
from sklearn.decomposition import PCA
from torch_basics import v_size, inner_product, torsion_angle
from residue_constants import MAX_RESIDUE_TYPE, ATOM_INDEX_CA

# %%
class ResidueBasedModel(PDB):
    def __init__(self, pdb_fn, dcd_fn=None):
        super().__init__(pdb_fn, dcd_fn)
        self.max_bead_type = MAX_RESIDUE_TYPE
        self.convert_to_cg()

    def convert_to_cg(self):
        self.top_cg = self.top.subset(self.top.select("name CA"))
        self.bead_index = self.residue_index[:, None]
        #
        self.R_cg = np.zeros((self.n_frame, self.n_residue, 1, 3))
        self.atom_mask_cg = np.zeros((self.n_residue, 1), dtype=np.float16)
        #
        mass_weighted_R = self.R * self.atomic_mass[None, ..., None]
        R_cg = mass_weighted_R.sum(axis=-2) / self.atomic_mass.sum(axis=-1)[None, ..., None]
        #
        self.R_cg = R_cg[..., None, :]
        self.atom_mask_cg = self.atom_mask[:, (ATOM_INDEX_CA,)]

    @staticmethod
    def convert_to_cg_tensor(r: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        # r: (n_residue, MAX_ATOM, 3)
        # mass: (n_residue, MAX_ATOM)
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
    def get_geometry(r: torch.Tensor, continuous: torch.Tensor, mask: torch.Tensor):
        not_defined = continuous == 0.0
        geom_s = {}
        #
        # pca
        geom_s["pca"] = torch.pca_lowrank(r.view(-1, 3))[-1].T[:2]

        # n_neigh
        n_neigh = torch.zeros((r.shape[0], 1), dtype=DTYPE)
        edge_src, edge_dst = torch_cluster.radius_graph(
            r[mask > 0.0],
            1.0,
        )
        n_neigh.index_add_(0, edge_src, torch.ones_like(edge_src, dtype=DTYPE))
        n_neigh.index_add_(0, edge_dst, torch.ones_like(edge_dst, dtype=DTYPE))
        n_neigh = n_neigh / 2.0
        geom_s["n_neigh"] = n_neigh

        # bond vectors
        geom_s["bond_length"] = {}
        geom_s["bond_vector"] = {}
        for shift in [1, 2]:
            b_len = torch.zeros(r.shape[0] + shift, dtype=DTYPE)
            dr = torch.zeros((r.shape[0] + shift, 3), dtype=DTYPE)
            #
            dr[shift:-shift] = r[:-shift, 0, :] - r[shift:, 0, :]
            b_len[shift:-shift] = v_size(dr[shift:-shift])
            #
            dr[shift:-shift] /= b_len[shift:-shift, None]
            dr[:-shift][not_defined] = 0.0
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
        sc = torch.zeros((r.shape[0], 4, 2))
        for i in range(4):
            sc[i : i + cosine.shape[0], i, 0] = cosine
            sc[i : i + sine.shape[0], i, 1] = sine
        geom_s["dihedral_angle"] = sc
        return geom_s

    @staticmethod
    def geom_to_feature(geom_s, noise_size: torch.Tensor, dtype=DTYPE) -> torch.Tensor:
        # features for each residue
        f_in = [[], []]  # 0d, 1d
        # 0d
        f_in[0].append(geom_s["n_neigh"])  # 1
        #
        f_in[0].append(geom_s["bond_length"][1][0][:, None])  # 4
        f_in[0].append(geom_s["bond_length"][1][1][:, None])
        f_in[0].append(geom_s["bond_length"][2][0][:, None])
        f_in[0].append(geom_s["bond_length"][2][1][:, None])
        #
        f_in[0].append(geom_s["bond_angle"][0][:, None])  # 2
        f_in[0].append(geom_s["bond_angle"][1][:, None])
        #
        f_in[0].append(geom_s["dihedral_angle"].reshape(-1, 8))  # 8
        #
        # noise-level
        f_in[0].append(noise_size[:, None])  # 1
        #
        f_in[0] = torch.as_tensor(np.concatenate(f_in[0], axis=1), dtype=dtype)  # 16x0e = 16
        n_scalar = f_in[0].size(1)  # 16
        #
        # 1d: unit vectors from adjacent residues to the current residue
        f_in[1].append(geom_s["bond_vector"][1][0])
        f_in[1].append(geom_s["bond_vector"][1][1])
        f_in[1].append(geom_s["bond_vector"][2][0])
        f_in[1].append(geom_s["bond_vector"][2][1])
        f_in[1] = torch.as_tensor(np.concatenate(f_in[1], axis=1), dtype=dtype)  # 4x1o = 12
        n_vector = int(f_in[1].size(1) // 3)  # 4
        #
        f_in = torch.cat(
            [
                f_in[0],
                f_in[1].reshape(f_in[1].shape[0], -1),
            ],
            dim=1,
        )  # 16x0e + 4x1o = 28
        return f_in, n_scalar, n_vector


class CalphaBasedModel(ResidueBasedModel):
    def __init__(self, pdb_fn, dcd_fn=None):
        super().__init__(pdb_fn, dcd_fn)

    def convert_to_cg(self):
        self.top_cg = self.top.subset(self.top.select("name CA"))
        self.bead_index = self.residue_index[:, None]
        #
        self.R_cg = self.R[:, :, (ATOM_INDEX_CA,), :]
        self.atom_mask_cg = self.atom_mask[:, (ATOM_INDEX_CA,)]

    @staticmethod
    def convert_to_cg_tensor(r: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        R_cg = r[:, :, (ATOM_INDEX_CA,), :]
        return R_cg


# %%
class Martini(PDB):
    def __init__(self, pdb_fn, dcd_fn=None):
        super().__init__(pdb_fn, dcd_fn)
        self.convert_to_cg()
        #

    def convert_to_cg(self):
        raise NotImplementedError


def main():
    pdb = ResidueBasedModel("pdb.processed/1HEO.pdb")


if __name__ == "__main__":
    main()

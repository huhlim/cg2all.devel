#!/usr/bin/env python

# %%
import mdtraj
import numpy as np
import torch
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
        R_cg = (
            mass_weighted_R.sum(axis=-2)
            / self.atomic_mass.sum(axis=-1)[None, ..., None]
        )
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
    def get_geometry(r: torch.Tensor, continuous: torch.Tensor):
        not_defined = continuous == 0.0
        geom_s = {}
        #
        # pca
        geom_s["pca"] = torch.pca_lowrank(r.view(-1, 3))[-1].T[:2]

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
    pdb.to_tensor()
    pdb.get_geometry(pdb.R_cg[0], pdb.continuous)


if __name__ == "__main__":
    main()

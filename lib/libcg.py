#!/usr/bin/env python

# %%
import mdtraj
import numpy as np
from libpdb import PDB
from numpy_basics import v_size, v_norm, inner_product, torsion_angle
from residue_constants import MAX_RESIDUE_TYPE, AMINO_ACID_s

# %%
class ResidueBasedModel(PDB):
    def __init__(self, pdb_fn, dcd_fn=None, center_of_mass=True):
        super().__init__(pdb_fn, dcd_fn)
        self.max_bead_type = MAX_RESIDUE_TYPE
        self.center_of_mass = center_of_mass
        self.convert_to_cg()
        #

    def convert_to_cg(self):
        self.top_cg = self.top.subset(self.top.select("name CA"))
        self.bead_index = self.residue_index[:, None]
        #
        if self.center_of_mass:
            self.R_cg = np.zeros((self.n_frame, self.n_residue, 1, 3))
            self.atom_mask_cg = np.zeros((self.n_residue, 1), dtype=np.float16)
            #
            for residue in self.top.residues:
                i_res = residue.index
                #
                index = np.array([atom.index for atom in residue.atoms])
                mass = np.array([atom.element.mass for atom in residue.atoms])
                #
                mass_weighted_xyz = mass[None, :, None] * self.traj.xyz[:, index, :]
                xyz = mass_weighted_xyz.sum(axis=1) / mass.sum()
                #
                self.R_cg[:, i_res, 0, :] = xyz
                self.atom_mask_cg[i_res, 0] = 1.0
        else:
            self.R_cg = self.R[:, :, (1,), :]
            self.atom_mask_cg = self.atom_mask[:, (1,)]

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

    def get_local_geometry(self, r):
        not_defined = self.continuous == 0.0
        geom_s = {}
        #
        # bond vectors
        geom_s["bond_length"] = {}
        geom_s["bond_vector"] = {}
        for shift in [1, 2]:
            b_len = np.zeros(r.shape[0] + shift)
            dr = np.zeros((r.shape[0] + shift, 3))
            dr[shift:-shift] = r[:-shift, 0, :] - r[shift:, 0, :]
            b_len[shift:-shift] = v_size(dr[shift:-shift])
            dr[shift:-shift] /= b_len[shift:-shift, None]
            dr[:-shift][not_defined] = 0.0
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
        R = np.array([r[0:-3], r[1:-2], r[2:-1], r[3:]]).reshape(-1, 4, 3)
        t_ang = torsion_angle(R)
        cosine = np.cos(t_ang)
        sine = np.sin(t_ang)
        for i in range(3):
            cosine[: -(i + 1)][not_defined[i + 1 : -3]] = 0.0
            sine[: -(i + 1)][not_defined[i + 1 : -3]] = 0.0
        sc = np.zeros((r.shape[0], 4, 2))
        for i in range(4):
            sc[i : i + cosine.shape[0], i, 0] = cosine
            sc[i : i + sine.shape[0], i, 1] = sine
        geom_s["dihedral_angle"] = sc
        return geom_s


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
    print(pdb.residue_index[14])
    print(AMINO_ACID_s[pdb.residue_index[14]])


if __name__ == "__main__":
    main()

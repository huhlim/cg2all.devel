#!/usr/bin/env python

#%%
import mdtraj
import numpy as np
from libpdb import PDB

#%%
class ResidueBasedModel(PDB):
    def __init__(self, pdb_fn, dcd_fn=None, center_of_mass=True):
        super().__init__(pdb_fn, dcd_fn)
        self.center_of_mass = center_of_mass
        self.convert_to_cg()
        #
    def convert_to_cg(self):
        self.top_cg = self.top.subset(self.top.select("name CA"))
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
                mass_weighted_xyz = mass[None,:,None] * self.traj.xyz[:,index,:]
                xyz = mass_weighted_xyz.sum(axis=1) / mass.sum()
                #
                self.R_cg[:,i_res,0,:] = xyz
                self.atom_mask_cg[i_res,0] = 1.
        else:
            self.R_cg = self.R[:,:,(1,),:]
            self.atom_mask_cg = self.atom_mask[:,(1,)]

    def write_cg(self, R, pdb_fn, dcd_fn=None):
        mask = np.where(self.atom_mask_cg)
        xyz = R[:,mask[0],mask[1],:]
        #
        traj = mdtraj.Trajectory(xyz[:1], self.top_cg)
        traj.save(pdb_fn)
        #
        if dcd_fn is not None:
            traj = mdtraj.Trajectory(xyz, self.top_cg)
            traj.save(dcd_fn)


# %%
class Martini(PDB):
    def __init__(self, pdb_fn, dcd_fn=None):
        super().__init__(pdb_fn, dcd_fn)
        self.convert_to_cg()
        #
    def convert_to_cg(self):
        raise NotImplementedError
#!/usr/bin/env python

#%%
# load modules
import numpy as np
import mdtraj
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

#%%
# define dataset
class PDBset(Dataset):
    def __init__(self, basedir, pdblist, n_neigh=16, center_of_mass=False):
        super().__init__()

        self.basedir = basedir
        self.pdb_s = []
        with open(pdblist) as fp:
            for line in fp:
                x = line.strip().split()
                self.pdb_s.append((x[0], int(x[1])))
        self.n_pdb = len(self.pdb_s)
        self.n_residues = sum([x[1] for x in self.pdb_s])
        self.n_neigh = n_neigh
        self.center_of_mass = center_of_mass

        # generate a map: index -> (pdb_id, residue_index)
        self.map = {}
        index = 0
        for pdb_id, n_residue in self.pdb_s:
            for i in range(n_residue):
                self.map[index] = (pdb_id, i)
                index += 1
    def __len__(self):
        return self.n_residues
    def get_cg_representation(self, all_atom: mdtraj.Trajectory) -> mdtraj.Trajectory:
        # get the center of mass of each residue
        cg = all_atom.atom_slice(all_atom.top.select("name CA"))
        if not self.center_of_mass:
            return cg
        masses = np.array([atom.element.mass for atom in all_atom.top.atoms], dtype=np.float32)
        residue_index = np.array([atom.residue.index for atom in all_atom.top.atoms], dtype=np.int16)
        xyz = np.zeros_like(cg.xyz[0])
        mass_sum = np.zeros_like(cg.xyz[0,:,0])
        np.add.at(xyz, (residue_index,), all_atom.xyz[0] * masses[:, None])
        np.add.at(mass_sum, (residue_index,), masses)
        xyz /= mass_sum[:,None]
        cg.xyz = xyz
        return cg
    def get_neighbor_list(self, cg: mdtraj.Trajectory, residue_index: int) -> np.ndarray:
        # get a neighbor list of the residue
        xyz = cg.xyz[0]
        dij = np.sqrt(np.sum((xyz[:,:] - xyz[residue_index,:])**2, axis=-1))
        neigh_s = np.argsort(dij)[:self.n_neigh]
        return neigh_s
    def __getitem__(self, index):
        # find the pdb_id and residue_index
        pdb_id, residue_index = self.map[index]
        #
        # load the pdb file
        pdb_fn = f"{self.basedir}/{pdb_id}.pdb"
        pdb = mdtraj.load(pdb_fn)
        aa = pdb.atom_slice(pdb.top.select("protein"))
        aa_index = aa.top.select(f"resid {residue_index}")
        cg = self.get_cg_representation(aa)
        neigh_s = self.get_neighbor_list(cg, residue_index)
        print (neigh_s)
        #
        sample = {}
        sample['input_xyz'] = cg.xyz[0, neigh_s]
        sample['output_xyz'] = aa.xyz[0, aa_index]
        return sample

test = PDBset("../pdb", "../pdb/pdblist")#, center_of_mass=True)
print (test[0])
#%%
def collate_fn(data):
    print (data)
    raise

#%%
def main():
    base_dir = "pdb"
    pdb_list = f"{base_dir}/pdblist"
    train_set = PDBset(base_dir, pdb_list)
    
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, collate_fn=collate_fn)
    for batch in train_loader:
        print (batch)
        break
    train_loader = DataLoader(train_set, shuffle=True, batch_size=2, num_workers=2, collate_fn=collate_fn)
    for x in train_loader:
        print (x)

if __name__ == '__main__':
    main()

# %%

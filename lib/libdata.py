#!/usr/bin/env python

#%%
import numpy as np
import pathlib
import functools

import torch
import torch_geometric

import libcg
from libconfig import BASE, DTYPE

#%%
class PDBset(torch_geometric.data.Dataset):
    def __init__(self, basedir, pdblist, cg_model):
        super().__init__()
        #
        self.basedir = pathlib.Path(basedir)
        self.pdb_s = []
        with open(pdblist) as fp:
            for line in fp:
                self.pdb_s.append(line.strip())
        #
        self.n_pdb = len(self.pdb_s)
        self.cg_model = cg_model
    def __len__(self):
        return self.n_pdb
    def __getitem__(self, index):
        pdb_id = self.pdb_s[index]
        pdb_fn = self.basedir / f"{pdb_id}.pdb"
        #
        cg = self.cg_model(pdb_fn)
        frame_index = np.random.randint(cg.n_frame)
        #
        data = torch_geometric.data.Data(pos=torch.tensor(cg.R_cg[frame_index][cg.atom_mask_cg==1.], dtype=DTYPE))
        #
        # one-hot encoding of residue type
        index = cg.bead_index[cg.atom_mask_cg==1.]
        data.x = torch.tensor(np.eye(cg.max_bead_type)[index], dtype=DTYPE)
        #
        data.residue_index = torch.tensor(cg.residue_index, dtype=DTYPE)
        data.output_atom_mask = torch.tensor(cg.atom_mask, dtype=DTYPE)
        data.output_xyz = torch.tensor(cg.R[frame_index], dtype=DTYPE)
        return data

# %%
def test():
    base_dir = BASE / 'pdb.processed'
    pdblist = BASE / 'pdb/pdblist'
    cg_model = functools.partial(libcg.ResidueBasedModel, center_of_mass=True)
    #
    train_set = PDBset(base_dir, pdblist, cg_model)
    train_loader = torch_geometric.loader.DataLoader(train_set, batch_size=2, shuffle=True, num_workers=2)
    for batch in train_loader:
        print (batch)

if __name__ == '__main__':
    test()

# %%

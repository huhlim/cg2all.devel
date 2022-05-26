#!/usr/bin/env python

#%%
import numpy as np
import pathlib

import torch
from torch.utils.data import DataLoader, Dataset

import libcg

#%%
class PDBset(Dataset):
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
        sample = {}
        sample['residue_index'] = cg.residue_index
        #
        sample['input_xyz'] = cg.R_cg[frame_index]
        sample['input_atom_mask'] = cg.atom_mask_cg
        #
        sample['output_xyz'] = cg.R[frame_index]
        sample['output_atom_mask'] = cg.atom_mask



# %%

#!/usr/bin/env python

# %%
import numpy as np
import pathlib
import functools

import torch
import torch_geometric

import libcg
from libconfig import BASE, DTYPE

# %%
class PDBset(torch_geometric.data.Dataset):
    def __init__(
        self,
        basedir,
        pdblist,
        cg_model,
        noise_level=0.0,
        get_structure_information=False,
    ):
        super().__init__()
        #
        self.basedir = pathlib.Path(basedir)
        self.pdb_s = []
        with open(pdblist) as fp:
            for line in fp:
                if line.startswith("#"):
                    continue
                self.pdb_s.append(line.strip())
        #
        self.n_pdb = len(self.pdb_s)
        self.cg_model = cg_model
        self.noise_level = noise_level
        self.get_structure_information = get_structure_information

    def __len__(self):
        return self.n_pdb

    def __getitem__(self, index):
        pdb_id = self.pdb_s[index]
        pdb_fn = self.basedir / f"{pdb_id}.pdb"
        #
        cg = self.cg_model(pdb_fn)
        frame_index = np.random.randint(cg.n_frame)
        #
        r = cg.R_cg[frame_index]
        dr = np.zeros((r.shape[0] + 1, 3))  # shape=(Nres+1, 3)
        dr[1:-1] = r[:-1, 0, :] - r[1:, 0, :]
        dr[1:-1] /= np.linalg.norm(dr[1:-1], axis=1)[:, None]
        dr[:-1][cg.continuous == 0.0] = 0.0
        if self.noise_level > 0.0:
            r += np.random.normal(scale=self.noise_level, size=r.shape)
        #
        data = torch_geometric.data.Data(
            pos=torch.tensor(r[cg.atom_mask_cg == 1.0], dtype=DTYPE)
        )
        #
        # features for each residue
        f_in = [[], []]  # 0d, 1d
        # 0d
        # one-hot encoding of residue type
        index = cg.bead_index[cg.atom_mask_cg == 1.0]
        f_in[0].append(np.eye(cg.max_bead_type)[index])
        # noise-level
        f_in[0].append(np.full((r.shape[0], 1), self.noise_level))
        f_in[0] = torch.tensor(np.concatenate(f_in[0], axis=1), dtype=DTYPE)
        #
        # 1d: unit vectors from adjacent residues to the current residue
        f_in[1].append(dr[:-1, None])
        f_in[1].append(-dr[1:, None])
        f_in[1] = torch.tensor(np.concatenate(f_in[1], axis=1), dtype=DTYPE)
        data.f_in = f_in
        #
        data.residue_type = torch.tensor(cg.residue_index, dtype=int)
        data.continuous = torch.tensor(cg.continuous, dtype=DTYPE)
        data.output_atom_mask = torch.tensor(cg.atom_mask, dtype=DTYPE)
        data.output_xyz = torch.tensor(cg.R[frame_index], dtype=DTYPE)
        #
        if self.get_structure_information:
            cg.get_structure_information()
            data.correct_bb = torch.tensor(cg.bb[frame_index], dtype=DTYPE)
            data.correct_torsion = torch.tensor(cg.torsion[frame_index], dtype=DTYPE)
        return data


# %%
def test():
    base_dir = BASE / "pdb.processed"
    pdblist = BASE / "pdb/pdblist"
    cg_model = functools.partial(libcg.ResidueBasedModel, center_of_mass=True)
    #
    train_set = PDBset(base_dir, pdblist, cg_model, noise_level=0.5)
    train_loader = torch_geometric.loader.DataLoader(
        train_set, batch_size=2, shuffle=True, num_workers=1
    )
    for batch in train_loader:
        print(batch)
        return


if __name__ == "__main__":
    test()

# %%

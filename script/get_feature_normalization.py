#!/usr/bin/env python

import os
import sys
import pathlib
import functools

import numpy as np
import torch
import torch_geometric

import tqdm

sys.path.insert(0, "lib")
from libdata import PDBset
from libcg import ResidueBasedModel


N_PROC = int(os.getenv("OMP_NUM_THREADS", "8"))


def main():
    base_dir = pathlib.Path("./")
    pdb_dir = base_dir / "pdb.pisces"
    pdblist_train = pdb_dir / "targets.train"
    pdblist_test = pdb_dir / "targets.test"
    pdblist_val = pdb_dir / "targets.valid"
    #
    cg_model = functools.partial(ResidueBasedModel, center_of_mass=True)
    _PDBset = functools.partial(
        PDBset,
        cg_model=cg_model,
        noise_level=0.0,
        get_structure_information=True,
        normalize=False,
    )
    #
    batch_size = 16
    train_set = _PDBset(pdb_dir, pdblist_train)
    train_loader = torch_geometric.loader.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=N_PROC,
    )
    f_in = []
    for data in tqdm.tqdm(train_loader):
        print(data.f_in.shape)
        f_in.append(data.f_in)
    f_in = torch.cat(f_in, dim=0).numpy()
    mean = f_in.mean(axis=0)
    std = f_in.std(axis=0)
    out = np.array([mean, std])
    np.save(pdb_dir / "transform.npy", out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import os
import sys
import time
import pathlib
import functools

import torch
import torch_geometric

from libconfig import BASE
from libdata import PDBset
from libcg import ResidueBasedModel
from libmodel import CONFIG, Model


def main():
    base_dir = BASE / "pdb.processed"
    pdblist = BASE / "pdb/pdblist"
    cg_model = functools.partial(ResidueBasedModel, center_of_mass=True)
    #
    train_set = PDBset(base_dir, pdblist, cg_model)
    train_loader = torch_geometric.loader.DataLoader(
        train_set, batch_size=2, shuffle=True, num_workers=2
    )
    batch = next(iter(train_loader))
    # for batch in train_loader:
    #     print (batch)
    #
    model = Model(CONFIG)
    t0 = time.time()
    out = model(batch)
    print(time.time() - t0)
    print(out["bb"].size())


if __name__ == "__main__":
    main()

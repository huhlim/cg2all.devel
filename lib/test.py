#!/usr/bin/env python

import os
import sys
import time
import pathlib
import functools

import torch_geometric

from libconfig import BASE
from libdata import PDBset
from libcg import ResidueBasedModel
import liblayer


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
    t0 = time.time()
    layer = liblayer.ConvLayer("22x0e", "20x0e + 20x1o", radius=1.0, l_max=2)
    out = layer(batch, batch.x)
    print(time.time() - t0)
    t0 = time.time()
    layer = liblayer.ConvLayer("20x0e + 20x1o", "20x0e + 20x1o", radius=1.0, l_max=2)
    out = layer(batch, out)
    print(time.time() - t0)


if __name__ == "__main__":
    main()

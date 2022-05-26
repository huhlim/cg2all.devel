#!/usr/bin/env python

import os
import sys
import pathlib
import functools

import torch_geometric

from libconfig import BASE
from libdata import PDBset
from libcg import ResidueBasedModel
import liblayer

def main():
    base_dir = BASE / 'pdb.processed'
    pdblist = BASE / 'pdb/pdblist'
    cg_model = functools.partial(ResidueBasedModel, center_of_mass=True)
    #
    train_set = PDBset(base_dir, pdblist, cg_model)
    train_loader = torch_geometric.loader.DataLoader(train_set, batch_size=2, shuffle=True, num_workers=2)
    batch = next(iter(train_loader))
    # for batch in train_loader:
    #     print (batch)
#
    layer = liblayer.ConvLayer(f"22x0e", "20x0e + 20x1o", radius=0.4, l_max=2)
    out = layer(batch, batch.x)
    print (out.size())

if __name__ == '__main__':
    main()
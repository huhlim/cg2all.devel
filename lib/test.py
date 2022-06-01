#!/usr/bin/env python

import os
import sys
import time
import pathlib
import functools

import torch
import torch_geometric

from libconfig import BASE, DTYPE
from libdata import PDBset
from libcg import ResidueBasedModel
from libmodel import CONFIG, Model

torch.autograd.set_detect_anomaly(True)

def main():
    base_dir = BASE / "pdb.processed"
    pdblist = BASE / "pdb/pdblist"
    cg_model = functools.partial(ResidueBasedModel, center_of_mass=True)
    #
    train_set = PDBset(base_dir, pdblist, cg_model, get_structure_information=True)
    train_loader = torch_geometric.loader.DataLoader(
        train_set, batch_size=1, shuffle=True, num_workers=2
    )
    batch = next(iter(train_loader))
    # for batch in train_loader:
    #     print (batch)
    #
    model = Model(CONFIG)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    optimizer.zero_grad()
    #
    out = model(batch)
    loss = out['R'].sum()
    loss.backward()
    #
    optimizer.step()

if __name__ == "__main__":
    main()

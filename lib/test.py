#!/usr/bin/env python

import os
import sys
import time
import pathlib
import functools

import numpy as np

import torch
import torch_geometric

from libconfig import BASE, DTYPE
from libdata import PDBset
from libcg import ResidueBasedModel
from libloss import loss_f_mse_R, loss_f_mse_R_CA, loss_f_bonded_energy
from libmodel import CONFIG, Model

# torch.autograd.set_detect_anomaly(True)


def loss_f(R, batch):
    loss_0 = loss_f_mse_R_CA(R, batch.output_xyz) * 1.0
    loss_1 = loss_f_mse_R(R, batch.output_xyz, batch.output_atom_mask) * 0.05
    loss_2 = loss_f_bonded_energy(R, batch.continuous, weight_s=(1.0, 0.0, 0.0)) * 0.1
    return loss_0 + loss_1 + loss_2


def main():
    base_dir = BASE / "pdb.processed"
    pdblist = BASE / "pdb/pdblist"
    cg_model = functools.partial(ResidueBasedModel, center_of_mass=True)
    #
    train_set = PDBset(base_dir, pdblist, cg_model, get_structure_information=True)
    train_loader = torch_geometric.loader.DataLoader(
        train_set, batch_size=5, shuffle=True, num_workers=1
    )
    # batch = next(iter(train_loader))
    # for x,y in zip(batch.continuous, batch.batch):
    #     print (x,y)
    # return
    #
    model = Model(CONFIG)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    #
    for epoch in range(100):
        t0 = time.time()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_f(out["R"], batch)
            loss.backward()
            optimizer.step()
            np.savez(f"out_{i}.npz", y=out["R"].detach().numpy(), y0=batch.output_xyz.detach().numpy())
            print(f"loss: ({epoch} {i})", loss)
        print (f"epoch: {epoch}", time.time() - t0)


if __name__ == "__main__":
    main()

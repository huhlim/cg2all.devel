#!/usr/bin/env python

import os
import sys
import copy
import pathlib
import functools

import numpy as np

import torch
import torch_geometric

sys.path.insert(0, "lib")
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
        train_set, batch_size=1, shuffle=True, num_workers=1
    )
    batch = next(iter(train_loader))
    #
    config = copy.deepcopy(CONFIG)
    config.update_from_flattened_dict(
        {
            "backbone.loss_weight.rigid_body": 1.0,
            "backbone.loss_weight.distogram": 1.0,
            "sidechain.loss_weight.torsion_angle": 0.1,
            "loss_weight.mse_R": 0.1,
            "loss_weight.rigid_body": 1.0,
            "loss_weight.distogram": 1.0,
        }
    )
    #
    model = Model(config, compute_loss=True)
    model = model.to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    #
    optimizer.zero_grad()
    out, loss = model(batch.to("cuda"))
    loss_sum = torch.tensor(0.0, device="cuda")
    for module_name, loss_per_module in loss.items():
        for loss_name, loss_value in loss_per_module.items():
            loss_sum += loss_value
            print(module_name, loss_name, loss_value)
    print(loss_sum)
    loss_sum.backward()
    optimizer.step()


if __name__ == "__main__":
    main()

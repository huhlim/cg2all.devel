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
from libdata import PDBset, create_trajectory_from_batch
from libcg import ResidueBasedModel
from libmodel import CONFIG, Model

torch.autograd.set_detect_anomaly(True)


def main():
    base_dir = BASE / "pdb.processed"
    pdblist = base_dir / "pdblist"
    cg_model = functools.partial(ResidueBasedModel, center_of_mass=True)
    #
    train_set = PDBset(
        base_dir, pdblist, cg_model, get_structure_information=True, cached=True
    )
    train_loader = torch_geometric.loader.DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )
    batch = next(iter(train_loader))
    #
    config = copy.deepcopy(CONFIG)
    config.update_from_flattened_dict(
        {
            "globals.num_recycle": 2,
            "feature_extraction.layer_type": "SE3Transformer",
            "globals.loss_weight.rigid_body": 1.0,
            "globals.loss_weight.FAPE_CA": 5.0,
        }
    )
    #
    model = Model(config, compute_loss=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    #
    for i in range(2):
        optimizer.zero_grad()
        out, loss, metrics = model(batch.to(device))
        loss_sum = torch.tensor(0.0, device=device)
        for module_name, loss_per_module in loss.items():
            for loss_name, loss_value in loss_per_module.items():
                loss_sum += loss_value
                print(module_name, loss_name, loss_value)
        print(loss_sum)
        loss_sum.backward(retain_graph=True)

        optimizer.step()
    #
    traj_s = create_trajectory_from_batch(batch, out["R"], write_native=True)
    traj_s[0].save("test.pdb")


if __name__ == "__main__":
    main()

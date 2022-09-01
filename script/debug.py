#!/usr/bin/env python

import os
import sys
import copy
import pathlib
import functools

import numpy as np

import torch
import dgl

sys.path.insert(0, "lib")
from libconfig import BASE
from libdata import PDBset, create_trajectory_from_batch
from libcg import ResidueBasedModel, CalphaBasedModel
from libmodel import Model, set_model_config

torch.autograd.set_detect_anomaly(True)


def main():
    base_dir = BASE / "pdb.processed"
    pdblist = base_dir / "pdblist"
    cg_model = ResidueBasedModel
    #
    config = set_model_config({}, cg_model)
    #
    train_set = PDBset(
        base_dir,
        pdblist,
        cg_model,
        radius=config.globals.radius,
        get_structure_information=True,
    )
    train_loader = dgl.dataloading.GraphDataLoader(
        train_set, batch_size=2, shuffle=False, num_workers=1
    )
    batch = next(iter(train_loader))
    #
    model = Model(config, cg_model, compute_loss=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    #
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    #
    optimizer.zero_grad()
    out, loss, metrics = model(batch.to(device))
    loss_sum = torch.tensor(0.0, device=device)
    for module_name, loss_per_module in loss.items():
        for loss_name, loss_value in loss_per_module.items():
            loss_sum += loss_value
            print(module_name, loss_name, loss_value)
    print(loss_sum)
    loss_sum.backward()
    optimizer.step()


if __name__ == "__main__":
    main()

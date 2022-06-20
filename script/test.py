#!/usr/bin/env python

import os
import sys
import copy
import pathlib
import functools

import GPUtil

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
    train_set = PDBset(base_dir, pdblist, cg_model, get_structure_information=True, cached=True)
    train_loader = torch_geometric.loader.DataLoader(
        train_set, batch_size=8, shuffle=True, num_workers=1, pin_memory=True, 
    )
    batch = next(iter(train_loader))
    #
    config = copy.deepcopy(CONFIG)
    config.update_from_flattened_dict(
        {
            "globals.num_recycle": 2,
            "globals.loss_weight.rigid_body": 1.0,
            "globals.loss_weight.FAPE_CA": 5.0,
            "globals.loss_weight.bonded_energy": 1.0,
            "globals.loss_weight.distance_matrix": 100.0,
            "backbone.loss_weight.rigid_body": 0.5,
            "backbone.loss_weight.FAPE_CA": 2.5,
            "backbone.loss_weight.bonded_energy": 0.5,
            "backbone.loss_weight.distance_matrix": 50.0,
            "feature_extraction.layer_type": "SE3Transformer",
        }
    )
    #
    model = Model(config, compute_loss=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("After Sending Model")
    GPUtil.showUtilization(all=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    #
    for i in range(2):
        for batch in train_loader:
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            out, loss, metrics = model(batch.to(device))
            print("After Forward-pass")
            GPUtil.showUtilization(all=True)
            torch.cuda.empty_cache()
            print("After Forward-pass")
            GPUtil.showUtilization(all=True)
            loss_sum = torch.tensor(0.0, device=device)
            for module_name, loss_per_module in loss.items():
                for loss_name, loss_value in loss_per_module.items():
                    loss_sum += loss_value
                    print(module_name, loss_name, loss_value)
            print("After Evaluating Loss")
            GPUtil.showUtilization(all=True)
            print(loss_sum)
            loss_sum.backward()
            print("After Loss backward")
            GPUtil.showUtilization(all=True)

            optimizer.step()
            print("After Optimizer.step()")
            GPUtil.showUtilization(all=True)


if __name__ == "__main__":
    main()

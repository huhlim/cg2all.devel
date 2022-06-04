#!/usr/bin/env python

import os
import sys
import time
import pathlib
import functools

import numpy as np

import torch
import torch_geometric
import pytorch_lightning as pl

sys.path.insert(0, "lib")
from libconfig import BASE, DTYPE
from libdata import PDBset, create_trajectory_from_batch
from libcg import ResidueBasedModel
from libloss import loss_f_mse_R, loss_f_mse_R_CA, loss_f_bonded_energy
import libmodel


class Model(pl.LightningModule):
    def __init__(self, _config):
        super().__init__()
        #
        self.feature_extraction_module = libmodel.BaseModule(_config.feature_extraction)
        self.backbone_module = libmodel.BaseModule(_config.backbone)
        self.sidechain_module = libmodel.BaseModule(_config.sidechain)

    def forward(self, batch: torch_geometric.data.Batch):
        f_out = self.feature_extraction_module(batch, batch.f_in)
        #
        ret = {}
        ret["bb"] = self.backbone_module(batch, f_out)
        ret["sc"] = self.sidechain_module(batch, f_out)
        ret["R"] = libmodel.build_structure(batch, ret["bb"], ret["sc"])
        return ret

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer

    def training_step(self, batch, batch_idx):
        return loss_f(self.forward(batch)["R"], batch)

    def test_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss = loss_f(out["R"], batch)
        return {"test_loss": loss, "out": out}

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss = loss_f(out["R"], batch)
        if self.current_epoch % 10 == 9 and batch_idx == 0:
            traj_s = create_trajectory_from_batch(batch, out["R"], write_native=True)
            for i, traj in enumerate(traj_s):
                traj.save(f"validation_step_{self.current_epoch}_{i}.pdb")
        return {"val_loss": loss}


def loss_f(R, batch):
    loss_0 = loss_f_mse_R_CA(R, batch.output_xyz) * 1.0
    loss_1 = loss_f_mse_R(R, batch.output_xyz, batch.output_atom_mask) * 0.05
    loss_2 = loss_f_bonded_energy(R, batch.continuous, weight_s=(1.0, 0.0, 0.0)) * 0.1
    return loss_0 + loss_1 + loss_2


def main():
    base_dir = pathlib.Path("/home/huhlim/work/ml/db/pisces")
    pdb_dir = base_dir / "pdb"
    pdblist_train = base_dir / "targets.train"
    pdblist_test = base_dir / "targets.test"
    pdblist_val = base_dir / "targets.valid"
    #
    cg_model = functools.partial(ResidueBasedModel, center_of_mass=True)
    #
    train_set = PDBset(
        pdb_dir, pdblist_train, cg_model, get_structure_information=False
    )
    train_loader = torch_geometric.loader.DataLoader(
        train_set, batch_size=16, shuffle=True, num_workers=8
    )
    val_set = PDBset(pdb_dir, pdblist_val, cg_model, get_structure_information=False)
    val_loader = torch_geometric.loader.DataLoader(
        val_set, batch_size=16, shuffle=False, num_workers=8
    )
    test_set = PDBset(pdb_dir, pdblist_test, cg_model, get_structure_information=False)
    test_loader = torch_geometric.loader.DataLoader(
        test_set, batch_size=16, shuffle=False, num_workers=8
    )
    model = Model(libmodel.CONFIG)
    #
    trainer = pl.Trainer(
        max_epochs=100, check_val_every_n_epoch=1, accelerator="auto", profiler="simple"
    )
    trainer.test(model, test_loader)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()

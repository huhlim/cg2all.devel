#!/usr/bin/env python

import os
import sys
import copy
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
import libmodel


class Model(pl.LightningModule):
    def __init__(self, _config, compute_loss=False):
        super().__init__()
        self.save_hyperparameters(_config.to_dict())
        self.model = libmodel.Model(_config, compute_loss=compute_loss)

    def forward(self, batch: torch_geometric.data.Batch):
        return self.model.forward(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.LinearLR(optimizer, 0.1, 1.0, 10),
                torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98),
            ],
            [10],  # milestone
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def get_loss_sum(self, loss):
        loss_sum = 0.0
        loss_s = {}
        for module_name, loss_per_module in loss.items():
            for loss_name, loss_value in loss_per_module.items():
                loss_sum += loss_value
                loss_s[f"{module_name}-{loss_name}"] = loss_value.item()
        if isinstance(loss_sum, torch.Tensor):
            loss_s["sum"] = loss_sum.item()
        else:
            loss_s["sum"] = loss_sum
        return loss_sum, loss_s

    def training_step(self, batch, batch_idx):
        out, loss, metric = self.forward(batch)
        loss_sum, loss_s = self.get_loss_sum(loss)
        #
        self.log("train_loss", loss_s, batch_size=batch.num_graphs)
        self.log("train_metric", metric, batch_size=batch.num_graphs)
        return {"loss": loss_sum, "metric": metric, "out": out}

    def test_step(self, batch, batch_idx):
        out, loss, metric = self.forward(batch)
        loss_sum, loss_s = self.get_loss_sum(loss)
        #
        traj_s = create_trajectory_from_batch(batch, out["R"], write_native=True)
        log_dir = pathlib.Path(self.logger.log_dir)
        for i, traj in enumerate(traj_s):
            out_f = log_dir / f"test_{self.current_epoch//5}_{i}.pdb"
            try:
                traj.save(out_f)
            except:
                sys.stderr.write(f"Failed to write {out_f}\n")
        #
        self.log("test_loss", loss_s, batch_size=batch.num_graphs)
        self.log("test_metric", metric, prog_bar=True, batch_size=batch.num_graphs)
        return {"loss": loss_sum, "metric": metric, "out": out}

    def validation_step(self, batch, batch_idx):
        out, loss, metric = self.forward(batch)
        loss_sum, loss_s = self.get_loss_sum(loss)
        #
        log_dir = pathlib.Path(self.logger.log_dir)
        if batch_idx == 0 and (1 + self.current_epoch) % 5 == 0:
            traj_s = create_trajectory_from_batch(batch, out["R"], write_native=True)
            for i, traj in enumerate(traj_s):
                out_f = log_dir / f"val_{self.current_epoch//5}_{i}.pdb"
                try:
                    traj.save(out_f)
                except:
                    sys.stderr.write(f"Failed to write {out_f}\n")
        self.log("val_loss", loss_s, batch_size=batch.num_graphs)
        self.log("val_metric", metric, prog_bar=True, batch_size=batch.num_graphs)
        return {"loss": loss_sum, "metric": metric, "out": out}


def main():
    hostname = os.getenv("HOSTNAME", "local")
    if hostname == "markov.bch.msu.edu" or hostname.startswith("gpu"):
        base_dir = pathlib.Path("./")
        pdb_dir = base_dir / "pdb.pisces"
        pdblist_train = pdb_dir / "targets.train"
        pdblist_test = pdb_dir / "targets.test"
        pdblist_val = pdb_dir / "targets.valid"
    else:
        base_dir = pathlib.Path("./")
        pdb_dir = base_dir / "pdb.processed"
        pdblist_train = pdb_dir / "pdblist"
        pdblist_test = pdb_dir / "pdblist"
        pdblist_val = pdb_dir / "pdblist"
    #
    cg_model = functools.partial(ResidueBasedModel, center_of_mass=True)
    #
    batch_size = 1
    train_set = PDBset(pdb_dir, pdblist_train, cg_model, get_structure_information=True)
    train_loader = torch_geometric.loader.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=8
    )
    val_set = PDBset(pdb_dir, pdblist_val, cg_model, get_structure_information=True)
    val_loader = torch_geometric.loader.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=8
    )
    test_set = PDBset(pdb_dir, pdblist_test, cg_model, get_structure_information=True)
    test_loader = torch_geometric.loader.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=8
    )
    #
    config = copy.deepcopy(libmodel.CONFIG)
    config.update_from_flattened_dict(
        {
            "feature_extraction.layer_type": "SE3Transformer",
            "backbone.layer_type": "SE3Transformer",
            "sidechain.layer_type": "SE3Transformer",
            "feature_extraction.num_recycle": 1,
            "backbone.num_recycle": 1,
            "sidechain.num_recycle": 1,
            "feature_extraction.num_layers": 3,
            "backbone.num_layers": 3,
            "sidechain.num_layers": 3,
            "backbone.loss_weight.rigid_body": 0.5,
            "backbone.loss_weight.distogram": 0.05,
            "backbone.activation": ["relu", "relu"],
            "sidechain.activation": ["relu", "relu"],
            "sidechain.loss_weight.torsion_angle": 0.05,
            "loss_weight.mse_R": 0.1,
            "loss_weight.rigid_body": 1.0,
            "loss_weight.distogram": 0.1,
            "loss_weight.torsion_angle": 0.1,
            "loss_weight.bonded_energy": 0.05,
        }
    )
    model = Model(config, compute_loss=True)
    # trainer = pl.Trainer(
    #     max_epochs=100,
    #     check_val_every_n_epoch=1,
    #     accelerator="auto",
    #     profiler="simple",
    # )
    trainer = pl.Trainer(
        check_val_every_n_epoch=1,
        accelerator="auto",
        log_every_n_steps=1,
        max_epochs=1,
    )
    trainer.fit(model, train_loader, val_loader)
    # trainer.test(model, test_loader)


if __name__ == "__main__":
    main()

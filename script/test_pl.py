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
from libdata import PDBset, create_trajectory_from_batch
from libcg import ResidueBasedModel
import libmodel


class Model(pl.LightningModule):
    def __init__(self, _config, compute_loss=False, checkpoint=False):
        super().__init__()
        self.save_hyperparameters(_config.to_dict())
        self.model = libmodel.Model(
            _config, compute_loss=compute_loss, checkpoint=checkpoint
        )

    def forward(self, batch: torch_geometric.data.Batch):
        return self.model.forward(batch)

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        # return optimizer
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.LinearLR(optimizer, 0.1, 1.0, 10),
                torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0),
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
        if torch.isnan(loss_sum):
            raise ValueError(out, loss_s, metric)
        #
        self.log(
            "train_loss",
            loss_s,
            batch_size=batch.num_graphs,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "train_metric",
            metric,
            batch_size=batch.num_graphs,
            on_epoch=True,
            on_step=False,
        )
        return {"loss": loss_sum, "metric": metric, "out": out}

    def test_step(self, batch, batch_idx):
        out, loss, metric = self.forward(batch)
        loss_sum, loss_s = self.get_loss_sum(loss)
        #
        traj_s = create_trajectory_from_batch(batch, out["R"], write_native=True)
        log_dir = pathlib.Path(self.logger.log_dir)
        for i, traj in enumerate(traj_s):
            out_f = log_dir / f"test_{self.current_epoch}_{i}.pdb"
            try:
                traj.save(out_f)
            except:
                sys.stderr.write(f"Failed to write {out_f}\n")
        #
        self.log(
            "test_loss",
            loss_s,
            batch_size=batch.num_graphs,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "test_metric",
            metric,
            prog_bar=True,
            batch_size=batch.num_graphs,
            on_epoch=True,
            on_step=False,
        )
        return {"loss": loss_sum, "metric": metric, "out": out}

    def validation_step(self, batch, batch_idx):
        out, loss, metric = self.forward(batch)
        loss_sum, loss_s = self.get_loss_sum(loss)
        #
        log_dir = pathlib.Path(self.logger.log_dir)
        if batch_idx == 0:
            traj_s = create_trajectory_from_batch(batch, out["R"], write_native=True)
            for i, traj in enumerate(traj_s):
                out_f = log_dir / f"val_{self.current_epoch}_{i}.pdb"
                try:
                    traj.save(out_f)
                except:
                    sys.stderr.write(f"Failed to write {out_f}\n")
        self.log(
            "val_loss",
            loss_s,
            batch_size=batch.num_graphs,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "val_metric",
            metric,
            prog_bar=True,
            batch_size=batch.num_graphs,
            on_epoch=True,
            on_step=False,
        )
        return {"loss": loss_sum, "metric": metric, "out": out}


def main():
    hostname = os.getenv("HOSTNAME", "local")
    if hostname == "markov.bch.msu.edu" or hostname.startswith("gpu"):  # and False:
        base_dir = pathlib.Path("./")
        pdb_dir = base_dir / "pdb.pisces"
        pdblist_train = pdb_dir / "targets.train"
        pdblist_test = pdb_dir / "targets.test"
        pdblist_val = pdb_dir / "targets.valid"
        cached = False
        pin_memory = False
    else:
        base_dir = pathlib.Path("./")
        pdb_dir = base_dir / "pdb.processed"
        pdblist_train = pdb_dir / "pdblist"
        pdblist_test = pdb_dir / "pdblist"
        pdblist_val = pdb_dir / "pdblist"
        cached = True
        pin_memory = True
    #
    cg_model = functools.partial(ResidueBasedModel, center_of_mass=True)
    _PDBset = functools.partial(
        PDBset,
        cg_model=cg_model,
        noise_level=0.0,
        get_structure_information=True,
        cached=cached,
    )
    #
    batch_size = 4
    train_set = _PDBset(pdb_dir, pdblist_train)
    train_loader = torch_geometric.loader.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=pin_memory,
    )
    val_set = _PDBset(pdb_dir, pdblist_val)
    val_loader = torch_geometric.loader.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=pin_memory,
    )
    test_set = _PDBset(pdb_dir, pdblist_test)
    test_loader = torch_geometric.loader.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=pin_memory,
    )
    #
    config = copy.deepcopy(libmodel.CONFIG)
    config.update_from_flattened_dict(
        {
            "globals.num_recycle": 1,
            "feature_extraction.layer_type": "SE3Transformer",
            "globals.loss_weight.rigid_body": 1.0,
            "globals.loss_weight.FAPE_CA": 5.0,
            # "globals.loss_weight.bonded_energy": 1.0,
            # "globals.loss_weight.rotation_matrix": 1.0,
            # "globals.loss_weight.torsion_angle": 0.2,
        }
    )
    model = Model(config, compute_loss=True, checkpoint=True)
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator="auto",
        gradient_clip_val=1.0,
        check_val_every_n_epoch=1,
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import os
import sys
import copy
import pathlib
import functools
import subprocess as sp

import numpy as np

import torch
import torch_geometric
import pytorch_lightning as pl

sys.path.insert(0, "lib")
from libdata import PDBset, create_trajectory_from_batch
from libcg import ResidueBasedModel
import libmodel


IS_DEVELOP = True
N_PROC = int(os.getenv("N_PROC", "8"))


class Model(pl.LightningModule):
    def __init__(self, _config, compute_loss=False, checkpoint=False, memcheck=False):
        super().__init__()
        self.save_hyperparameters(_config.to_dict())
        self.model = libmodel.Model(
            _config, compute_loss=compute_loss, checkpoint=checkpoint
        )
        self.memcheck = memcheck

    def forward(self, batch: torch_geometric.data.Batch):
        return self.model.forward(batch)

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()

    def on_train_batch_start(self, batch, batch_idx):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            if self.memcheck:
                torch.cuda.reset_peak_memory_stats()
                self.n_residue = batch.pos.size(0)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.memcheck and self.device.type == "cuda":
            max_memory = torch.cuda.max_memory_allocated() / 1024**2  # in MBytes
            self.log(
                "memory",
                {"n_residue": float(self.n_residue), "max_memory": max_memory},
                prog_bar=True,
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
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
            prog_bar=IS_DEVELOP,
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
        log_dir = pathlib.Path(self.logger.log_dir)
        #
        if self.current_epoch == 0 and batch_idx == 0:
            self.model.test_equivariance(batch)
            #
            sp.call(["cp", "lib/libmodel.py", log_dir])
            sp.call(["cp", __file__, log_dir])
        #
        out, loss, metric = self.forward(batch)
        loss_sum, loss_s = self.get_loss_sum(loss)
        #
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
    if IS_DEVELOP:
        pl.seed_everything(25, workers=True)
    #
    hostname = os.getenv("HOSTNAME", "local")
    if (
        hostname == "markov.bch.msu.edu"
        or hostname.startswith("gpu")
        and (not IS_DEVELOP)
    ):
        base_dir = pathlib.Path("./")
        pdb_dir = base_dir / "pdb.pisces"
        pdblist_train = pdb_dir / "targets.train"
        pdblist_test = pdb_dir / "targets.test"
        pdblist_val = pdb_dir / "targets.valid"
        pin_memory = False
        batch_size = 8
    else:
        base_dir = pathlib.Path("./")
        pdb_dir = base_dir / "pdb.processed"
        pdblist_train = pdb_dir / "pdblist"
        pdblist_test = pdb_dir / "pdblist"
        pdblist_val = pdb_dir / "pdblist"
        pin_memory = True
        batch_size = 1
    #
    cg_model = functools.partial(ResidueBasedModel, center_of_mass=True)
    _PDBset = functools.partial(
        PDBset,
        cg_model=cg_model,
        noise_level=0.0,
        get_structure_information=True,
        random_rotation=True,
    )
    #
    _DataLoader = functools.partial(
        torch_geometric.loader.DataLoader,
        batch_size=batch_size,
        num_workers=N_PROC,
        pin_memory=pin_memory,
    )
    train_set = _PDBset(pdb_dir, pdblist_train)
    train_loader = _DataLoader(
        train_set,
        shuffle=True,
    )
    val_set = _PDBset(pdb_dir, pdblist_val)
    val_loader = torch_geometric.loader.DataLoader(
        val_set,
        shuffle=False,
    )
    test_set = _PDBset(pdb_dir, pdblist_test)
    test_loader = torch_geometric.loader.DataLoader(
        test_set,
        shuffle=False,
    )
    #
    config = copy.deepcopy(libmodel.CONFIG)
    config.update_from_flattened_dict(
        {
            "globals.num_recycle": 2,
            "feature_extraction.layer_type": "SE3Transformer",
            "feature_extraction.num_layers": 1,
            "initialization.num_layers": 1,
            "transition.num_layers": 1,
            "backbone.num_layers": 1,
            #
            "globals.loss_weight.rigid_body": 1.0,
            "globals.loss_weight.FAPE_CA": 5.0,
            "globals.loss_weight.bonded_energy": 1.0,
            "globals.loss_weight.rotation_matrix": 1.0,
            "globals.loss_weight.torsion_angle": 1.0,
            #
            # "backbone.loss_weight.rigid_body": 0.5,
            # "backbone.loss_weight.FAPE_CA": 2.5,
            # "backbone.loss_weight.bonded_energy": 0.5,
            # "backbone.loss_weight.rotation_matrix": 0.5,
            # "sidechain.loss_weight.torsion_angle": 0.5,
        }
    )
    #
    if config.globals.num_recycle > 1:
        feature_extraction_in_Irreps = " + ".join(
            [
                config.initialization.out_Irreps,
                config.backbone.out_Irreps,
                # "1x1o",
            ]
        )
        sidechain_in_Irreps = " + ".join(
            [
                config.transition.out_Irreps,
                config.backbone.out_Irreps,
                # "1x1o",
            ]
        )
    else:
        feature_extraction_in_Irreps = config.initialization.out_Irreps
        sidechain_in_Irreps = config.transition.out_Irreps
    #
    config.update_from_flattened_dict(
        {
            "feature_extraction.in_Irreps": feature_extraction_in_Irreps,
            "sidechain.in_Irreps": sidechain_in_Irreps,
        }
    )
    #
    model = Model(config, compute_loss=False, checkpoint=True, memcheck=True)
    if IS_DEVELOP:
        trainer = pl.Trainer(
            max_epochs=100,
            accelerator="auto",
            gradient_clip_val=1.0,
            check_val_every_n_epoch=10,
            #        overfit_batches=5,
        )
        trainer.fit(model, train_loader, val_loader)
    else:
        trainer = pl.Trainer(
            max_epochs=100,
            accelerator="auto",
            gradient_clip_val=1.0,
            check_val_every_n_epoch=1,
        )
        trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()

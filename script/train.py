#!/usr/bin/env python

import os
import sys
import json
import logging
import pathlib
import functools
import subprocess as sp

import numpy as np

import torch
import dgl
import pytorch_lightning as pl

sys.path.insert(0, "lib")
from libdata import PDBset, create_trajectory_from_batch
from libcg import ResidueBasedModel, CalphaBasedModel, Martini
import libmodel

torch.multiprocessing.set_sharing_strategy("file_system")


N_PROC = int(os.getenv("OMP_NUM_THREADS", "8"))
IS_DEVELOP = False
if IS_DEVELOP:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)


class Model(pl.LightningModule):
    def __init__(self, _config, cg_model, compute_loss=False, memcheck=False):
        super().__init__()
        self.save_hyperparameters(_config.to_dict())
        self.model = libmodel.Model(_config, cg_model, compute_loss=compute_loss)
        self.memcheck = memcheck

    @property
    def cg_model(self):
        return self.model.cg_model

    def forward(self, batch: dgl.DGLGraph):
        return self.model.forward(batch)

    def on_fit_start(self):
        self.model.set_rigid_operations(self.device, dtype=self.dtype)

    def on_train_batch_start(self, batch, batch_idx):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            if self.memcheck:
                torch.cuda.reset_peak_memory_stats()
                self.n_residue = batch.num_nodes()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.memcheck and self.device.type == "cuda":
            max_memory = torch.cuda.max_memory_allocated() / 1024**2  # in MBytes
            self.log(
                "memory",
                {"n_residue": float(self.n_residue), "max_memory": max_memory},
                prog_bar=True,
            )

    def configure_optimizers(self):
        if IS_DEVELOP:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.LinearLR(optimizer, 0.1, 1.0, 10),
                torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99),
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
                if isinstance(loss_value, torch.Tensor):
                    loss_s[f"{module_name}-{loss_name}"] = loss_value.item()
                else:
                    loss_s[f"{module_name}-{loss_name}"] = loss_value
        if isinstance(loss_sum, torch.Tensor):
            loss_s["sum"] = loss_sum.item()
        else:
            loss_s["sum"] = loss_sum
        return loss_sum, loss_s

    def training_step(self, batch, batch_idx):
        out, loss, metric = self.forward(batch)
        loss_sum, loss_s = self.get_loss_sum(loss)
        if torch.isnan(loss_sum):
            log_dir = pathlib.Path(self.logger.log_dir)
            torch.save(
                {
                    "model": self.model,
                    "batch": batch,
                    "out": out,
                    "loss_s": loss_s,
                    "metric": metric,
                },
                log_dir / "error.pt",
            )
            raise ValueError(out, loss_s, metric)
        #
        bs = batch.batch_size
        self.log("train_loss", loss_s, batch_size=bs, on_epoch=True, on_step=False)
        self.log(
            "train_metric", metric, batch_size=bs, on_epoch=True, on_step=False, prog_bar=IS_DEVELOP
        )
        return {"loss": loss_sum, "metric": metric, "out": out}

    def test_step(self, batch, batch_idx):
        out, loss, metric = self.forward(batch)
        loss_sum, loss_s = self.get_loss_sum(loss)
        #
        traj_s = create_trajectory_from_batch(batch, out["R"], write_native=True)
        log_dir = pathlib.Path(self.logger.log_dir)
        for i, traj in enumerate(traj_s):
            out_f = log_dir / f"test_{batch_idx}_{i}.pdb"
            try:
                traj.save(out_f)
                # TODO: write_ssbond
            except:
                sys.stderr.write(f"Failed to write {out_f}\n")
        #
        bs = batch.batch_size
        self.log("test_loss", loss_s, batch_size=bs, on_epoch=True, on_step=False)
        self.log("test_metric", metric, prog_bar=True, batch_size=bs, on_epoch=True, on_step=False)
        return {"loss": loss_sum, "metric": metric, "out": out}

    def validation_step(self, batch, batch_idx):
        log_dir = pathlib.Path(self.logger.log_dir)
        #
        if self.current_epoch == 0 and batch_idx == 0:
            if IS_DEVELOP:
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
        bs = batch.batch_size
        self.log("val_loss_sum", loss_sum, batch_size=bs, on_epoch=True, on_step=False)
        self.log("val_loss", loss_s, batch_size=bs, on_epoch=True, on_step=False)
        self.log("val_metric", metric, prog_bar=True, batch_size=bs, on_epoch=True, on_step=False)
        return {"loss": loss_sum, "metric": metric, "out": out}


def main():
    if len(sys.argv) > 1:
        json_fn = pathlib.Path(sys.argv[1])
        with open(json_fn) as f:
            config = json.load(f)
        name = json_fn.stem
    else:
        config = {}
        name = None
    #
    if len(sys.argv) > 2:
        ckpt_fn = pathlib.Path(sys.argv[2])
    else:
        ckpt_fn = None
    #
    if name is None:
        name = "devel"
    #
    pl.seed_everything(25, workers=True)
    #
    # configure
    cg_model = CalphaBasedModel
    config = libmodel.set_model_config(config, cg_model)
    #
    # set file paths
    hostname = os.getenv("HOSTNAME", "local")
    if hostname == "markov.bch.msu.edu" or hostname.startswith("gpu") and (not IS_DEVELOP):
        base_dir = pathlib.Path("./")
        pdb_dir = base_dir / "pdb.pisces"
        pdblist_train = pdb_dir / "targets.train"
        pdblist_test = pdb_dir / "targets.test"
        pdblist_val = pdb_dir / "targets.valid"
        batch_size = 8
    else:
        base_dir = pathlib.Path("./")
        pdb_dir = base_dir / "pdb.processed"
        pdblist_train = pdb_dir / "targets.train"
        pdblist_test = pdb_dir / "targets.test"
        pdblist_val = pdb_dir / "targets.valid"
        batch_size = 4

    _PDBset = functools.partial(
        PDBset,
        cg_model=cg_model,
        radius=config.globals.radius,
        noise_level=0.0,
        get_structure_information=True,
        random_rotation=True,
        cache=IS_DEVELOP,
    )
    _DataLoader = functools.partial(
        dgl.dataloading.GraphDataLoader, batch_size=batch_size, num_workers=N_PROC
    )
    # define train/val/test sets
    train_set = _PDBset(pdb_dir, pdblist_train)
    train_loader = _DataLoader(train_set, shuffle=True)
    val_set = _PDBset(pdb_dir, pdblist_val)
    val_loader = _DataLoader(val_set, shuffle=False)
    test_set = _PDBset(pdb_dir, pdblist_test)
    test_loader = _DataLoader(test_set, shuffle=False)
    #
    # define model
    model = Model(config, cg_model, compute_loss=True, memcheck=True)
    #
    logger = pl.loggers.TensorBoardLogger("lightning_logs", name=name)
    checkpointing = pl.callbacks.ModelCheckpoint(
        dirpath=logger.log_dir,
        monitor="val_loss_sum",
    )
    # early_stopping = pl.callbacks.EarlyStopping(
    #     monitor="val_loss_sum",
    #     min_delta=1e-4,
    # )
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",
        gradient_clip_val=1.0,
        check_val_every_n_epoch=10 if IS_DEVELOP else 1,
        logger=logger,
        callbacks=[checkpointing],  # , early_stopping],
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_fn)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()

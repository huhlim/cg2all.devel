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
import torch_geometric
import pytorch_lightning as pl

sys.path.insert(0, "lib")
from libdata import PDBset, create_trajectory_from_batch
from libcg import ResidueBasedModel, CalphaBasedModel, Martini
import libmodel
from libconfig import USE_EQUIVARIANCE_TEST

torch.multiprocessing.set_sharing_strategy("file_system")


N_PROC = int(os.getenv("OMP_NUM_THREADS", "8"))
IS_DEVELOP = USE_EQUIVARIANCE_TEST | True
if IS_DEVELOP:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)


class Model(pl.LightningModule):
    def __init__(
        self, _config, cg_model, compute_loss=False, gradient_checkpoint=False, memcheck=False
    ):
        super().__init__()
        self.save_hyperparameters(_config.to_dict())
        self.model = libmodel.Model(
            _config, cg_model, compute_loss=compute_loss, gradient_checkpoint=gradient_checkpoint
        )
        self.memcheck = memcheck

    @property
    def cg_model(self):
        return self.model.cg_model

    def forward(self, batch: torch_geometric.data.Batch):
        return self.model.forward(batch)

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
        if IS_DEVELOP:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
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
            if IS_DEVELOP:
                if USE_EQUIVARIANCE_TEST:
                    self.model.test_equivariance(batch)
                    sys.exit()
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
            "val_loss_sum",
            loss_sum,
            batch_size=batch.num_graphs,
            on_epoch=True,
            on_step=False,
        )
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
    if len(sys.argv) > 1:
        json_fn = pathlib.Path(sys.argv[1])
        with open(json_fn) as f:
            config = json.load(f)
        name = json_fn.stem
    else:
        config = {}
        name = None
    if len(sys.argv) > 2:
        ckpt_fn = pathlib.Path(sys.argv[2])
    else:
        ckpt_fn = None
    #
    if IS_DEVELOP:
        name = "devel"
    #
    pl.seed_everything(25, workers=True)
    #
    hostname = os.getenv("HOSTNAME", "local")
    if hostname == "markov.bch.msu.edu" or hostname.startswith("gpu") and (not IS_DEVELOP):
        base_dir = pathlib.Path("./")
        pdb_dir = base_dir / "pdb.pisces"
        pdblist_train = pdb_dir / "targets.train"
        pdblist_test = pdb_dir / "targets.test"
        pdblist_val = pdb_dir / "targets.valid"
        pin_memory = False
        gradient_checkpoint = True
        batch_size = 2
    else:
        base_dir = pathlib.Path("./")
        pdb_dir = base_dir / "pdb.processed"
        pdblist_train = pdb_dir / "pdblist"
        pdblist_test = pdb_dir / "pdblist"
        pdblist_val = pdb_dir / "pdblist"
        pin_memory = True
        gradient_checkpoint = False
        batch_size = 2
    #
    # cg_model = functools.partial(ResidueBasedModel)
    cg_model = CalphaBasedModel
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
    val_loader = _DataLoader(
        val_set,
        shuffle=False,
    )
    test_set = _PDBset(pdb_dir, pdblist_test)
    test_loader = _DataLoader(
        test_set,
        shuffle=False,
    )
    #
    in_Irreps = train_set[0].f_in_Irreps
    config["initialization.in_Irreps"] = str(in_Irreps)
    config = libmodel.set_model_config(config)
    model = Model(
        config, cg_model, compute_loss=True, gradient_checkpoint=gradient_checkpoint, memcheck=True
    )
    #
    logger = pl.loggers.TensorBoardLogger("lightning_logs", name=name)
    checkpointing = pl.callbacks.ModelCheckpoint(
        dirpath=logger.log_dir,
        monitor="val_loss_sum",
    )
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss_sum",
        min_delta=1e-4,
    )
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",
        gradient_clip_val=1.0,
        check_val_every_n_epoch=1,
        logger=logger,
        callbacks=[checkpointing],  # , early_stopping],
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_fn)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import os
import sys
import json
import logging
import pathlib
import functools
import subprocess as sp
import argparse

import numpy as np

import torch
import dgl
import pytorch_lightning as pl

sys.path.insert(0, "lib")
from libdata import PDBset, create_trajectory_from_batch
from libcg import ResidueBasedModel, CalphaBasedModel, Martini
from libpdb import write_SSBOND
import libmodel

import warnings

warnings.filterwarnings("ignore")

torch.multiprocessing.set_sharing_strategy("file_system")


N_PROC = int(os.getenv("OMP_NUM_THREADS", "8"))
IS_DEVELOP = False
if IS_DEVELOP:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)


class Model(pl.LightningModule):
    def __init__(self, _config, cg_model, compute_loss=False, memcheck=False):
        super().__init__()
        self._config = _config
        self.save_hyperparameters(_config.to_dict())
        self.model = libmodel.Model(_config, cg_model, compute_loss=compute_loss)
        self.memcheck = memcheck

    def log(self, *arg, **kwarg):
        super().log(*arg, **kwarg, sync_dist=True)

    @property
    def cg_model(self):
        return self.model.cg_model

    def forward(self, batch: dgl.DGLGraph):
        return self.model.forward(batch)

    def on_fit_start(self):
        self.model.set_constant_tensors(self.device, dtype=self.dtype)

    def on_test_start(self):
        if not hasattr(self.model, "RIGID_OPs"):
            self.model.set_constant_tensors(self.device, dtype=self.dtype)

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
            optimizer = torch.optim.Adam(self.parameters(), lr=self._config.train.lr)
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
        self.write_pdb(batch, out, f"test_{batch_idx}")
        #
        bs = batch.batch_size
        self.log("test_loss", loss_s, batch_size=bs, on_epoch=True, on_step=False)
        self.log("test_metric", metric, prog_bar=True, batch_size=bs, on_epoch=True, on_step=False)
        return {"loss": loss_sum, "metric": metric, "out": out}

    def validation_step(self, batch, batch_idx):
        out, loss, metric = self.forward(batch)
        loss_sum, loss_s = self.get_loss_sum(loss)
        #
        if batch_idx == 0:
            self.write_pdb(batch, out, f"val_{self.current_epoch}")
        #
        bs = batch.batch_size
        self.log("val_loss_sum", loss_sum, batch_size=bs, on_epoch=True, on_step=False)
        self.log("val_loss", loss_s, batch_size=bs, on_epoch=True, on_step=False)
        self.log("val_metric", metric, prog_bar=True, batch_size=bs, on_epoch=True, on_step=False)
        return {"loss": loss_sum, "metric": metric, "out": out}

    @pl.utilities.rank_zero_only
    def write_pdb(self, batch, out, prefix, write_native=True):
        log_dir = pathlib.Path(self.logger.log_dir)
        traj_s, ssbond_s = create_trajectory_from_batch(batch, out["R"], write_native=True)
        for i, (traj, ssbond) in enumerate(zip(traj_s, ssbond_s)):
            out_f = log_dir / f"{prefix}_{i}.pdb"
            try:
                traj.save(out_f)
                if len(ssbond) > 0:
                    write_SSBOND(out_f, traj.top, ssbond)
            except:
                sys.stderr.write(f"Failed to write {out_f}\n")


def main():
    arg = argparse.ArgumentParser(prog="cg2all")
    arg.add_argument("--name", dest="name", default=None)
    arg.add_argument("--config", dest="config_json_fn", default=None)
    arg.add_argument("--ckpt", dest="ckpt_fn", default=None)
    arg.add_argument("--epoch", dest="max_epochs", default=100, type=int)
    arg.add_argument(
        "--cg",
        dest="cg_model",
        default="CalphaBasedModel",
        choices=["CalphaBasedModel", "ResidueBasedModel"],
    )
    arg.add_argument("--requeue", dest="requeue", action="store_true", default=False)
    arg = arg.parse_args()
    #
    if arg.config_json_fn is not None:
        with open(arg.config_json_fn) as fp:
            config = json.load(fp)
    else:
        config = {}
    if arg.name is None:
        if arg.config_json_fn is not None:
            arg.name = pathlib.Path(arg.config_json_fn).stem
        else:
            arg.name = "devel"
    #
    pl.seed_everything(25, workers=True)
    #
    # configure
    config["cg_model"] = config.get("cg_model", arg.cg_model)
    if config["cg_model"] == "CalphaBasedModel":
        cg_model = CalphaBasedModel
    elif config["cg_model"] == "ResidueBasedModel":
        cg_model = ResidueBasedModel
    config = libmodel.set_model_config(config, cg_model)
    #
    # set file paths
    pdb_dir = pathlib.Path(config.train.dataset)
    pdblist_train = pdb_dir / "targets.train"
    pdblist_test = pdb_dir / "targets.test"
    pdblist_val = pdb_dir / "targets.valid"
    #
    _PDBset = functools.partial(
        PDBset,
        cg_model=cg_model,
        radius=config.globals.radius,
        noise_level=0.0,
        get_structure_information=True,
        random_rotation=True,
        use_pt="CA",
    )
    batch_size = config.train.batch_size
    _DataLoader = functools.partial(
        dgl.dataloading.GraphDataLoader, batch_size=batch_size, num_workers=N_PROC
    )
    # define train/val/test sets
    train_set = _PDBset(pdb_dir, pdblist_train, crop=config.train.crop_size)
    train_loader = _DataLoader(train_set, shuffle=True)
    val_set = _PDBset(pdb_dir, pdblist_val)
    val_loader = _DataLoader(val_set, shuffle=False)
    test_set = _PDBset(pdb_dir, pdblist_test)
    test_loader = _DataLoader(test_set, shuffle=False)
    #
    # define model
    model = Model(config, cg_model, compute_loss=True, memcheck=True)
    #
    trainer_kwargs = {}
    trainer_kwargs["max_epochs"] = arg.max_epochs
    trainer_kwargs["gradient_clip_val"] = 1.0
    trainer_kwargs["check_val_every_n_epoch"] = 10 if (IS_DEVELOP or len(train_set) < 20) else 1
    trainer_kwargs["logger"] = pl.loggers.TensorBoardLogger("lightning_logs", name=arg.name)
    trainer_kwargs["callbacks"] = [
        pl.callbacks.ModelCheckpoint(
            dirpath=trainer_kwargs["logger"].log_dir, monitor="val_loss_sum"
        )
    ]
    trainer_kwargs["plugins"] = (
        [pl.plugins.environments.SLURMEnvironment(auto_requeue=True)] if arg.requeue else []
    )
    #
    n_gpu = torch.cuda.device_count()
    trainer_kwargs["accelerator"] = "gpu" if n_gpu > 0 else "cpu"
    if n_gpu >= 2:
        trainer_kwargs["strategy"] = pl.strategies.DDPStrategy(static_graph=True)
        trainer_kwargs["devices"] = n_gpu
    #
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, train_loader, val_loader, ckpt_path=arg.ckpt_fn)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()

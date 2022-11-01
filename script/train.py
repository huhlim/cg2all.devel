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

BASE = pathlib.Path(__file__).parents[1].resolve()
LIB_HOME = str(BASE / "lib")
sys.path.insert(0, LIB_HOME)

from libdata import PDBset, create_trajectory_from_batch
from libcg import ResidueBasedModel, CalphaBasedModel, Martini
from libpdb import write_SSBOND
import libmodel

import warnings

warnings.filterwarnings("ignore")
# torch.autograd.set_detect_anomaly(True)

torch.multiprocessing.set_sharing_strategy("file_system")


N_PROC = int(os.getenv("OMP_NUM_THREADS", "8"))
IS_DEVELOP = False
if IS_DEVELOP:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)


class Model(pl.LightningModule):
    def __init__(self, _config={}, cg_model=CalphaBasedModel, compute_loss=False, memcheck=False):
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

    def on_predict_start(self):
        self.on_test_start()

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
        lr_sc = self._config.train.get("lr_sc", self._config.train.lr)
        if lr_sc == self._config.train.lr:
            lr_sc = None

        if lr_sc is not None:
            parameters = [[], []]
            for name, param in self.model.named_parameters():
                x = name.split(".")
                if x[0] == "structure_module" and x[4] == "0":
                    parameters[1].append(param)
                else:
                    parameters[0].append(param)
        #
        if IS_DEVELOP:
            if lr_sc is None:
                optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
            else:
                optimizer = torch.optim.Adam(
                    [{"params": parameters[0]}, {"params": parameters[1], "lr": lr_sc}], lr=0.01
                )
        else:
            if lr_sc is None:
                optimizer = torch.optim.Adam(self.parameters(), lr=self._config.train.lr)
            else:
                optimizer = torch.optim.Adam(
                    [{"params": parameters[0]}, {"params": parameters[1], "lr": lr_sc}],
                    lr=self._config.train.lr,
                )

        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.LinearLR(optimizer, 0.1, 1.0, 10),
                torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=self._config.train.lr_gamma
                ),
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
                    "model": self.model.state_dict(),
                    "batch": batch,
                    "loss_s": loss_s,
                    "metric": metric,
                },
                log_dir / "error.pt",
            )
            raise ValueError(loss_s, metric)

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

    def predict_step(self, batch, batch_idx):
        out, loss, metric = self.forward(batch)
        self.write_pdb(batch, out, f"test_{batch_idx}", log_dir=self.output_dir)

    def validation_step(self, batch, batch_idx):
        out, loss, metric = self.forward(batch)
        loss_sum, loss_s = self.get_loss_sum(loss)
        #
        if self.current_epoch % 10 == 9 or batch_idx == 0:
            self.write_pdb(batch, out, f"val_{self.current_epoch}_{batch_idx}")
        #
        bs = batch.batch_size
        self.log("val_loss_sum", loss_sum, batch_size=bs, on_epoch=True, on_step=False)
        self.log("val_loss", loss_s, batch_size=bs, on_epoch=True, on_step=False)
        self.log("val_metric", metric, prog_bar=True, batch_size=bs, on_epoch=True, on_step=False)
        return {"loss": loss_sum, "metric": metric, "out": out}

    def write_pdb(self, batch, out, prefix, write_native=True, log_dir=None):
        if log_dir is None:
            log_dir = pathlib.Path(self.logger.log_dir)
        #
        traj_s, ssbond_s, bfac_s = create_trajectory_from_batch(
            batch, out["R"], bfac=out["bfactors"], write_native=True
        )
        #
        for i, (traj, ssbond) in enumerate(zip(traj_s, ssbond_s)):
            try:
                out_f = log_dir / f"{prefix}_{self.global_rank}_{i}.pdb"
            except:
                out_f = log_dir / f"{prefix}_{i}.pdb"
            #
            try:
                traj.save(out_f, bfactors=bfac_s[i])
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
    if config.train.md_frame > 0:
        use_md = True
        n_frame = config.train.md_frame
    else:
        use_md = False
        n_frame = 1
    #
    _PDBset = functools.partial(
        PDBset,
        cg_model=cg_model,
        radius=config.globals.radius,
        use_pt=config.train.get("use_pt", "CA_aug"),
        augment=True,
        use_md=use_md,
        n_frame=n_frame,
    )
    batch_size = config.train.batch_size
    n_runner = max(1, torch.cuda.device_count())
    _DataLoader = functools.partial(
        dgl.dataloading.GraphDataLoader, batch_size=batch_size, num_workers=(N_PROC // n_runner)
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
    model = Model(config, cg_model, compute_loss=True)  # , memcheck=True)
    #
    trainer_kwargs = {}
    trainer_kwargs["max_epochs"] = arg.max_epochs
    trainer_kwargs["gradient_clip_val"] = 1.0
    trainer_kwargs["check_val_every_n_epoch"] = 1
    trainer_kwargs["logger"] = pl.loggers.TensorBoardLogger("lightning_logs", name=arg.name)
    trainer_kwargs["callbacks"] = [
        pl.callbacks.ModelCheckpoint(
            dirpath=trainer_kwargs["logger"].log_dir, monitor="val_loss_sum"
        )
    ]
    #
    n_gpu = torch.cuda.device_count()
    trainer_kwargs["accelerator"] = "gpu" if n_gpu > 0 else "cpu"
    if n_gpu >= 2:
        trainer_kwargs["strategy"] = pl.strategies.DDPStrategy(static_graph=True)
        trainer_kwargs["devices"] = n_gpu
    else:
        trainer_kwargs["plugins"] = (
            [pl.plugins.environments.SLURMEnvironment(auto_requeue=True)] if arg.requeue else []
        )
    #
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, train_loader, val_loader, ckpt_path=arg.ckpt_fn)
    trainer.test(model, test_loader)
    #
    trainer.save_checkpoint(pathlib.Path(trainer_kwargs["logger"].log_dir) / "last.ckpt")


if __name__ == "__main__":
    main()

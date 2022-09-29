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
from train import *

import warnings

warnings.filterwarnings("ignore")

torch.multiprocessing.set_sharing_strategy("file_system")


N_PROC = int(os.getenv("OMP_NUM_THREADS", "8"))


def main():
    arg = argparse.ArgumentParser(prog="cg2all")
    arg.add_argument("--output", dest="output_dir", default="./")
    arg.add_argument("--config", dest="config_json_fn", default=None)
    arg.add_argument("--ckpt", dest="ckpt_fn", default=None, required=True)
    arg.add_argument(
        "--cg",
        dest="cg_model",
        default="CalphaBasedModel",
        choices=["CalphaBasedModel", "ResidueBasedModel"],
    )
    arg = arg.parse_args()
    #
    if arg.config_json_fn is not None:
        with open(arg.config_json_fn) as fp:
            config = json.load(fp)
    else:
        config = {}
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
    pdblist_test = pdb_dir / "targets.test"
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
        noise_level=0.0,
        get_structure_information=True,
        random_rotation=True,
        use_pt="CA",
        use_md=use_md,
        n_frame=n_frame,
    )
    batch_size = config.train.batch_size
    n_runner = max(1, torch.cuda.device_count())
    _DataLoader = functools.partial(
        dgl.dataloading.GraphDataLoader, batch_size=batch_size, num_workers=(N_PROC // n_runner)
    )
    # define train/val/test sets
    test_set = _PDBset(pdb_dir, pdblist_test)
    test_loader = _DataLoader(test_set, shuffle=False)
    #
    # define model
    model = Model.load_from_checkpoint(arg.ckpt_fn, _config=config, cg_model=cg_model)
    model.output_dir = pathlib.Path(arg.output_dir)
    #
    trainer_kwargs = {}
    #
    n_gpu = torch.cuda.device_count()
    trainer_kwargs["accelerator"] = "gpu" if n_gpu > 0 else "cpu"
    if n_gpu >= 2:
        trainer_kwargs["strategy"] = pl.strategies.DDPStrategy(static_graph=True)
        trainer_kwargs["devices"] = n_gpu
    #
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.predict(model, test_loader)


if __name__ == "__main__":
    main()

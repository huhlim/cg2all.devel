#!/usr/bin/env python

import os
import sys
import time
import pathlib
import argparse

import numpy as np

import torch
import dgl

BASE = pathlib.Path(__file__).parents[1].resolve()
LIB_HOME = str(BASE / "lib")
sys.path.insert(0, LIB_HOME)

from libdata import PredictionData, create_trajectory_from_batch
from libcg import ResidueBasedModel, CalphaBasedModel, Martini
from libpdb import write_SSBOND
import libmodel

import warnings

warnings.filterwarnings("ignore")
N_PROC = int(os.getenv("OMP_NUM_THREADS", "8"))


def main():
    arg = argparse.ArgumentParser(prog="cg2all")
    arg.add_argument("--pdb", dest="in_pdb_fn", required=True)
    arg.add_argument("--dcd", dest="in_dcd_fn", default=None)
    arg.add_argument("-o", "--out", "--output", dest="out_fn", required=True)
    arg.add_argument("--ckpt", dest="ckpt_fn", default=None, required=True)
    arg = arg.parse_args()
    #
    timing = {}
    #
    timing["loading_ckpt"] = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(arg.ckpt_fn, map_location=device)
    config = ckpt["hyper_parameters"]
    timing["loading_ckpt"] = time.time() - timing["loading_ckpt"]
    #
    timing["model_configuration"] = time.time()
    if config["cg_model"] == "CalphaBasedModel":
        cg_model = CalphaBasedModel
    elif config["cg_model"] == "ResidueBasedModel":
        cg_model = ResidueBasedModel
    elif config["cg_model"] == "Martini":
        cg_model = Martini
    config = libmodel.set_model_config(config, cg_model)
    model = libmodel.Model(config, cg_model, compute_loss=False)
    #
    state_dict = ckpt["state_dict"]
    for key in list(state_dict):
        state_dict[".".join(key.split(".")[1:])] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.set_constant_tensors(device)
    model.eval()
    timing["model_configuration"] = time.time() - timing["model_configuration"]
    #
    timing["loading_input"] = time.time()
    input_s = PredictionData(
        arg.in_pdb_fn, cg_model, dcd_fn=arg.in_dcd_fn, radius=config.globals.radius
    )
    timing["loading_input"] = time.time() - timing["loading_input"]
    #
    if arg.in_dcd_fn is None:
        t0 = time.time()
        batch = input_s[0].to(device)
        timing["loading_input"] += time.time() - t0
        #
        t0 = time.time()
        with torch.no_grad():
            R = model.forward(batch)[0]["R"]
        timing["forward_pass"] = time.time() - t0
        #
        timing["writing_output"] = time.time()
        traj_s, ssbond_s = create_trajectory_from_batch(batch, R)
        traj_s[0].save(arg.out_fn)
        if len(ssbond_s[0]) > 0:
            write_SSBOND(arg.out_fn, traj_s[0].top, ssbond_s[0])
        timing["writing_output"] = time.time() - timing["writing_output"]
    else:
        xyz = []
        for batch in input_s:
            t0 = time.time()
            batch = batch.to(device)
            timing["loading_input"] += time.time() - t0
            #
            t0 = time.time()
            with torch.no_grad():
                R = model.forward(batch)[0]["R"].cpu().detach().numpy()
                xyz.append(R)
            timing["forward_pass"] = time.time() - t0
        #
        timing["writing_output"] = time.time()
        top = create_topology_from_data(batch)
        traj = mdtraj.Trajectory(xyz=np.array(R), top=top)
        traj.save(arg.out_fn)
        timing["writing_output"] = time.time() - timing["writing_output"]
    #
    print(timing)


if __name__ == "__main__":
    main()
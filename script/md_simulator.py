#!/usr/bin/env python

import os
import sys
import json
import time
import tqdm
import pathlib
import argparse
import numpy as np

import torch
import dgl

os.environ["OPENMM_PLUGIN_DIR"] = "/dev/null"
import mdtraj
from openmm.app import CharmmParameterSet

BASE = pathlib.Path(__file__).parents[1].resolve()
LIB_HOME = str(BASE / "lib")
sys.path.insert(0, LIB_HOME)

from libconfig import MODEL_HOME, DTYPE, DATA_HOME
from libdata import create_topology_from_data
import libcg
from libpdb import write_SSBOND
from libter import patch_termini
import libmodel
from torch_basics import v_norm_safe, inner_product, rotate_vector
from libmd import (
    MDdata,
    DCDReporter,
    MolecularMechanicsForceField,
    LangevinIntegratorTorch,
    Constraint,
    MDsimulator,
)

import warnings

warnings.filterwarnings("ignore")


def main():
    arg = argparse.ArgumentParser(prog="md_simulator")
    arg.add_argument("-p", "--pdb", dest="in_pdb_fn", required=True)
    arg.add_argument("-o", "--out", "--output", dest="output", required=True)
    arg.add_argument("-t", "--temperature", dest="temperature", default=298.15, type=float)
    arg.add_argument("-n", "--step", dest="n_step", default=1000, type=int)
    arg.add_argument("--gamma", "--friction", dest="gamma", default=100.0, type=float)
    arg.add_argument("--time_step", dest="time_step", default=0.01, type=float)
    arg.add_argument("--freq", "--output_freq", dest="output_freq", default=100, type=int)
    arg.add_argument("--ckpt", dest="ckpt_fn")
    arg.add_argument("--restart", dest="restart", default=None)
    arg.add_argument(
        "--cg",
        dest="cg_model",
        default="ResidueBasedModel",
        choices=["CalphaBasedModel", "CA", "ca", "ResidueBasedModel", "RES", "res"],
    )
    arg.add_argument(
        "--exclude_one_four_pair", dest="include_one_four_pair", action="store_false", default=True
    )
    arg.add_argument("--bond_weight", dest="bond_weight", default=1.0, type=float)
    arg.add_argument("--torsion_weight", dest="torsion_weight", default=0.0, type=float)
    arg.add_argument("--cg_weight", dest="cg_weight", default=1.0, type=float)
    arg = arg.parse_args()
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_start = time.time()
    #
    if arg.cg_model in ["CalphaBasedModel", "CA", "ca"]:
        model_type = "CalphaBasedModel"
    elif arg.cg_model in ["ResidueBasedModel", "RES", "res"]:
        model_type = "ResidueBasedModel"
    #
    if arg.ckpt_fn is not None:
        ckpt_fn = arg.ckpt_fn
    else:
        ckpt_fn = MODEL_HOME / f"{model_type}.ckpt"
    ckpt = torch.load(ckpt_fn, map_location=device)
    config = ckpt["hyper_parameters"]
    #
    if model_type == "CalphaBasedModel":
        cg_model = libcg.CalphaBasedModel
    elif model_type == "ResidueBasedModel":
        cg_model = libcg.ResidueBasedModel
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
    #
    if arg.restart is None:
        data = MDdata(arg.in_pdb_fn, cg_model)
        data.set_velocities(arg.temperature)
    else:
        restart = np.load(arg.restart)
        r_cg = restart["r_cg"]
        v_cg = restart["v_cg"]
        data = MDdata(arg.in_pdb_fn, cg_model, r_cg=r_cg, v_cg=v_cg)
    #
    output_dir = pathlib.Path("./")
    #
    minimizer = torch.optim.SGD([data.r_cg], lr=1e-5)
    if model_type == "CalphaBasedModel":
        integrator = LangevinIntegratorTorch(
            arg.time_step, arg.temperature, arg.gamma, constraint=Constraint(data)
        )
    else:
        integrator = LangevinIntegratorTorch(arg.time_step, arg.temperature, arg.gamma)
    #
    batch = data.convert_to_batch(data.r_cg).to(device)
    ret = model.forward(batch)[0]
    #
    out_top, out_atom_index = create_topology_from_data(batch)
    out_mask = batch.ndata["output_atom_mask"].cpu().detach().numpy()
    #
    ssbond = []
    for cys_i, cys_j in enumerate(batch.ndata["ssbond_index"].cpu().detach().numpy()):
        if cys_j != -1:
            ssbond.append((cys_j, cys_i))
    ssbond.sort()
    #
    out_fn = output_dir / f"{arg.output}.init.pdb"
    xyz = ret["R"].cpu().detach().numpy()[out_mask > 0.0][None, out_atom_index]
    output = mdtraj.Trajectory(xyz=xyz, topology=out_top)
    output = patch_termini(output)
    output.save(out_fn)
    if len(ssbond) > 0:
        write_SSBOND(out_fn, output.top, ssbond)
    #
    loss_f = MolecularMechanicsForceField(
        data,
        model,
        model_type,
        device,
        include_one_four_pair=arg.include_one_four_pair,
        bond_weight=arg.bond_weight,
        torsion_weight=arg.torsion_weight,
        cg_weight=arg.cg_weight,
    )
    simulation = MDsimulator(model, loss_f, integrator, device)

    dcd_out = DCDReporter(str(output_dir / f"{arg.output}.dcd"), arg.output_freq)
    for i in range(arg.n_step):
        T, energy, kinetic_energy = simulation.step(data, arg.output_freq)
        n_step = (i + 1) * arg.output_freq
        #
        print(
            f"STEP_MD  {n_step:4d} {n_step*arg.time_step:8.3f} "
            f"{energy:8.2f} {kinetic_energy:8.2f} {T:6.1f} "
            f"{time.time() - time_start:6.1f}"
        )
        #
        with torch.no_grad():
            batch = data.convert_to_batch(data.r_cg).to(device)
            ret = model.forward(batch)[0]
        xyz = ret["R"].cpu().detach().numpy()[out_mask > 0.0][None, out_atom_index]
        output = mdtraj.Trajectory(xyz=xyz, topology=out_top)
        output = patch_termini(output)
        dcd_out.report(output)

        if T > arg.temperature + 10000:
            sys.exit("Too high temperature!")
    #
    r_cg = data.r_cg.detach().cpu().numpy()
    v_cg = data.v_cg.detach().cpu().numpy()
    np.savez(str(output_dir / f"{arg.output}.restart.npz"), r_cg=r_cg, v_cg=v_cg)


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import os
import sys
import json
import time
import tqdm
import pathlib
import argparse

import torch
import dgl

os.environ["OPENMM_PLUGIN_DIR"] = "/dev/null"
import mdtraj

BASE = pathlib.Path(__file__).parents[1].resolve()
LIB_HOME = str(BASE / "lib")
sys.path.insert(0, LIB_HOME)

from libconfig import MODEL_HOME, DTYPE
from libdata import MinimizableData, create_topology_from_data
import libcg
from libpdb import write_SSBOND
from libter import patch_termini
import libmodel
from torch_basics import v_norm_safe, inner_product, rotate_vector
from libcryoem import CryoEMLossFunction

import warnings

warnings.filterwarnings("ignore")


def rotation_matrix_from_6D(v):
    v0 = v[0]
    v1 = v[1]
    e0 = v_norm_safe(v0, index=0)
    u1 = v1 - e0 * inner_product(e0, v1)
    e1 = v_norm_safe(u1, index=1)
    e2 = torch.cross(e0, e1)
    rot = torch.stack([e0, e1, e2], dim=1).mT
    return rot


def rigid_body_move(r, trans, rotation):
    center_of_mass = r.mean(dim=(0, 1))
    rotation_matrix = rotation_matrix_from_6D(rotation)
    #
    r_cg = r - center_of_mass
    r_cg = rotate_vector(rotation_matrix, r_cg) + center_of_mass + trans
    #
    return r_cg


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_start = time.time()

    model_type = "CalphaBasedModel"
    ckpt_fn = MODEL_HOME / f"{model_type}.ckpt"
    ckpt = torch.load(ckpt_fn, map_location=device)
    config = ckpt["hyper_parameters"]
    #
    cg_model = libcg.CalphaBasedModel
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
    data = MinimizableData("test/3iyg.pdb", cg_model)
    loss_f = CryoEMLossFunction("test/emd_5148.map", data, device)
    #
    trans = torch.zeros(3, dtype=DTYPE, requires_grad=True)
    rotation = torch.tensor(
        [[1, 0, 0], [0, 1, 0]], dtype=DTYPE, requires_grad=True
    )
    #
    optimizer = torch.optim.Adam([data.r_cg, trans, rotation], lr=0.001)
    #
    r_cg = rigid_body_move(data.r_cg, trans, rotation)
    batch = data.convert_to_batch(r_cg).to(device)
    R = model.forward(batch)[0]["R"]
    #
    out_top, out_atom_index = create_topology_from_data(batch)
    out_mask = batch.ndata["output_atom_mask"].cpu().detach().numpy()
    #
    ssbond = []
    for cys_i, cys_j in enumerate( batch.ndata["ssbond_index"].cpu().detach().numpy()):
        if cys_j != -1:
            ssbond.append((cys_j, cys_i))
    ssbond.sort()
    #
    out_fn = f"test/min.{0:04d}.pdb"
    xyz = R.cpu().detach().numpy()[out_mask > 0.0][None, out_atom_index]
    output = mdtraj.Trajectory(xyz=xyz, topology=out_top)
    output = patch_termini(output)
    output.save(out_fn)
    if len(ssbond) > 0:
        write_SSBOND(out_fn, output.top, ssbond)
    #
    for i in range(1000):
        loss_sum, loss = loss_f.eval(batch, R)
        print("STEP", i, loss["cryo_em"].detach().cpu().item(), time.time() - time_start)
        print({name: value.detach().cpu().item() for name, value in loss.items()})
        loss_sum.backward()
        optimizer.step()
        optimizer.zero_grad()
        #
        r_cg = rigid_body_move(data.r_cg, trans, rotation)
        batch = data.convert_to_batch(r_cg).to(device)
        R = model.forward(batch)[0]["R"]
        #
        if (i+1)%10 == 0:
            out_fn = f"test/min.{i+1:04d}.pdb"
            xyz = R.cpu().detach().numpy()[out_mask > 0.0][None, out_atom_index]
            output = mdtraj.Trajectory(xyz=xyz, topology=out_top)
            output = patch_termini(output)
            output.save(out_fn)
            if len(ssbond) > 0:
                write_SSBOND(out_fn, output.top, ssbond)


if __name__ == "__main__":
    main()

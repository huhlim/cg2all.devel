#!/usr/bin/env python

import copy
import numpy as np
import pathlib
import mdtraj
from typing import List

import torch
from torch.utils.data import Dataset

import e3nn

import dgl

import libcg
from libconfig import BASE, DTYPE
from torch_basics import v_norm
from residue_constants import AMINO_ACID_s, AMINO_ACID_REV_s, residue_s, ATOM_INDEX_CA


class PDBset(Dataset):
    def __init__(
        self,
        basedir: str,
        pdblist: List[str],
        cg_model,
        radius=0.8,
        noise_level=0.0,
        self_loop=False,
        get_structure_information=False,
        random_rotation=False,
        dtype=DTYPE,
    ):
        super().__init__()
        #
        self.dtype = dtype
        self.basedir = pathlib.Path(basedir)
        self.pdb_s = []
        with open(pdblist) as fp:
            for line in fp:
                if line.startswith("#"):
                    continue
                self.pdb_s.append(line.strip())
        #
        self.n_pdb = len(self.pdb_s)
        self.cg_model = cg_model
        self.radius = radius
        self.self_loop = self_loop
        self.noise_level = noise_level
        self.get_structure_information = get_structure_information
        self.random_rotation = random_rotation

    def __len__(self):
        return self.n_pdb

    def __getitem__(self, index):
        pdb_id = self.pdb_s[index]
        pdb_fn = self.basedir / f"{pdb_id}.pdb"
        #
        cg = self.cg_model(pdb_fn)
        if self.random_rotation:
            cg = self.rotate_randomly(cg)
        frame_index = np.random.randint(cg.n_frame)
        #
        r_cg = torch.as_tensor(cg.R_cg[frame_index], dtype=self.dtype)
        if self.noise_level > 0.0:
            noise_size = torch.randn(1).item() * (self.noise_level / 2.0) + self.noise_level
            if noise_size > 0.0:
                dr = torch.randn(r_cg.size()) * noise_size
                r_cg += dr - dr.mean(axis=0)
            else:
                noise_size = 0.0
        else:
            noise_size = 0.0
        noise_size = torch.full((cg.n_residue,), noise_size)
        #
        geom_s = cg.get_geometry(r_cg, cg.continuous, cg.atom_mask_cg, pca=True)
        node_feat = cg.geom_to_feature(geom_s, noise_size=noise_size, dtype=self.dtype)
        pos = r_cg[cg.atom_mask_cg > 0.0]
        data = dgl.radius_graph(pos, self.radius, self_loop=self.self_loop)
        data.ndata["pos"] = pos
        data.ndata["pos0"] = pos.clone()
        data.ndata["node_feat_0"] = node_feat["0"][..., None]  # shape=(N, 16, 1)
        data.ndata["node_feat_1"] = node_feat["1"]  # shape=(N, 4, 3)
        #
        edge_src, edge_dst = data.edges()
        data.edata["rel_pos"] = pos[edge_dst] - pos[edge_src]
        #
        global_frame = torch.as_tensor(geom_s["pca"], dtype=self.dtype).reshape(-1)
        data.ndata["global_frame"] = global_frame.repeat(cg.n_residue, 1)
        #
        data.ndata["chain_index"] = torch.as_tensor(cg.chain_index, dtype=torch.long)
        data.ndata["residue_type"] = torch.as_tensor(cg.residue_index, dtype=torch.long)
        data.ndata["continuous"] = torch.as_tensor(cg.continuous, dtype=self.dtype)
        #
        ssbond_index = torch.full((data.num_nodes(),), -1, dtype=torch.long)
        for cys_i, cys_j in cg.ssbond_s:
            if cys_i < cys_j:   # because of loss_f_atomic_clash
                ssbond_index[cys_j] = cys_i
            else:
                ssbond_index[cys_i] = cys_j
        data.ndata["ssbond_index"] = ssbond_index
        #
        edge_feat = torch.zeros((data.num_edges(), 3), dtype=self.dtype)  # bonded / ssbond / space
        for i, cont in enumerate(cg.continuous):
            if cont and data.has_edges_between(i - 1, i):  # i-1 and i is connected
                eid = data.edge_ids(i - 1, i)
                edge_feat[eid, 0] = 1.0
                eid = data.edge_ids(i, i - 1)
                edge_feat[eid, 0] = 1.0
        for cys_i, cys_j in cg.ssbond_s:
            if data.has_edges_between(cys_i, cys_j):
                eid = data.edge_ids(cys_i, cys_j)
                edge_feat[eid, 1] = 1.0
                eid = data.edge_ids(cys_j, cys_i)
                edge_feat[eid, 1] = 1.0
        edge_feat[edge_feat.sum(dim=-1) == 0.0, 2] = 1.0
        data.edata["edge_feat_0"] = edge_feat[..., None]
        #
        data.ndata["atomic_radius"] = torch.as_tensor(cg.atomic_radius, dtype=self.dtype)
        data.ndata["atomic_mass"] = torch.as_tensor(cg.atomic_mass, dtype=self.dtype)
        data.ndata["input_atom_mask"] = torch.as_tensor(cg.atom_mask_cg, dtype=self.dtype)
        data.ndata["output_atom_mask"] = torch.as_tensor(cg.atom_mask, dtype=self.dtype)
        data.ndata["pdb_atom_mask"] = torch.as_tensor(cg.atom_mask_pdb, dtype=self.dtype)
        data.ndata["output_xyz"] = torch.as_tensor(cg.R[frame_index], dtype=self.dtype)
        #
        r_cntr = libcg.get_residue_center_of_mass(
            data.ndata["output_xyz"], data.ndata["atomic_mass"]
        )
        v_cntr = r_cntr - data.ndata["output_xyz"][:, ATOM_INDEX_CA]
        data.ndata["v_cntr"] = v_norm(v_cntr)
        #
        if self.get_structure_information:
            cg.get_structure_information()
            data.ndata["correct_bb"] = torch.as_tensor(cg.bb[frame_index], dtype=self.dtype)
            data.ndata["correct_torsion"] = torch.as_tensor(
                cg.torsion[frame_index], dtype=self.dtype
            )
            data.ndata["torsion_mask"] = torch.as_tensor(cg.torsion_mask, dtype=self.dtype)
        #
        return data

    def rotate_randomly(self, cg):
        random_rotation = e3nn.o3.rand_matrix().numpy()
        #
        out = copy.deepcopy(cg)
        out.R_cg = out.R_cg @ random_rotation.T
        out.R = out.R @ random_rotation.T
        return out


def create_topology_from_data(data: dgl.DGLGraph, write_native: bool = False) -> mdtraj.Topology:
    top = mdtraj.Topology()
    #
    chain_prev = -1
    resSeq = 0
    for i_res in range(data.ndata["residue_type"].size(0)):
        chain_index = data.ndata["chain_index"][i_res]
        if chain_index != chain_prev:
            chain_prev = chain_index
            resSeq = 0
            top_chain = top.add_chain()
        #
        resSeq += 1
        residue_type_index = int(data.ndata["residue_type"][i_res])
        residue_name = AMINO_ACID_s[residue_type_index]
        residue_name_std = AMINO_ACID_REV_s.get(residue_name, residue_name)
        if residue_name == "UNK":
            continue
        ref_res = residue_s[residue_name]
        top_residue = top.add_residue(residue_name_std, top_chain, resSeq)
        #
        if write_native:
            mask = data.ndata["pdb_atom_mask"][i_res]
        else:
            mask = data.ndata["output_atom_mask"][i_res]
        #
        for i_atm, atom_name in enumerate(ref_res.atom_s):
            if mask[i_atm] > 0.0:
                element = mdtraj.core.element.Element.getBySymbol(atom_name[0])
                top.add_atom(atom_name, element, top_residue)
    return top


def create_trajectory_from_batch(
    batch: dgl.DGLGraph,
    output: torch.Tensor = None,
    write_native: bool = False,
) -> List[mdtraj.Trajectory]:
    #
    if output is not None:
        R = output.cpu().detach().numpy()
    #
    write_native = write_native or output is None
    #
    traj_s = []
    start = 0
    for idx, data in enumerate(dgl.unbatch(batch)):
        top = create_topology_from_data(data, write_native=write_native)
        #
        xyz = []
        if write_native:
            mask = data.ndata["pdb_atom_mask"].cpu().detach().numpy()
            xyz.append(data.ndata["output_xyz"].cpu().detach().numpy()[mask > 0.0])
        else:
            mask = data.ndata["output_atom_mask"].cpu().detach().numpy()
            mask = np.ones_like(mask)
        #
        if output is not None:
            end = start + data.num_nodes()
            xyz.append(R[start:end][mask > 0.0])
            start = end
        xyz = np.array(xyz)
        #
        traj = mdtraj.Trajectory(xyz=xyz, topology=top)
        traj_s.append(traj)
    return traj_s


def test():
    base_dir = BASE / "pdb.processed"
    pdblist = base_dir / "targets.train"
    cg_model = libcg.ResidueBasedModel
    #
    train_set = PDBset(
        base_dir,
        pdblist,
        cg_model,
        noise_level=0.0,
        random_rotation=True,
        get_structure_information=True,
    )
    train_loader = dgl.dataloading.GraphDataLoader(
        train_set, batch_size=2, shuffle=False, num_workers=1
    )
    batch = next(iter(train_loader))
    # traj_s = create_trajectory_from_batch(batch, batch.ndata["output_xyz"], write_native=True)
    # for i,traj in enumerate(traj_s):
    #     traj.save(f"test_{i}.pdb")
    # # for batch in train_loader:
    #     traj_s = create_trajectory_from_batch(
    #         batch, batch.output_xyz, write_native=True
    #     )


if __name__ == "__main__":
    test()

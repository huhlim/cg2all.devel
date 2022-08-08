#!/usr/bin/env python

import copy
import numpy as np
import pathlib
import mdtraj
from typing import List

import torch
import torch_geometric
import torch_cluster
import e3nn

import libcg
from libconfig import BASE, DTYPE, EQUIVARIANT_TOLERANCE
from residue_constants import AMINO_ACID_s, AMINO_ACID_REV_s, residue_s


class PDBset(torch_geometric.data.Dataset):
    def __init__(
        self,
        basedir: str,
        pdblist: List[str],
        cg_model,
        noise_level=0.0,
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
        self.noise_level = noise_level
        self.get_structure_information = get_structure_information
        self.random_rotation = random_rotation

    def __len__(self):
        return self.n_pdb

    def __getitem__(self, index) -> torch_geometric.data.Data:
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
            if noise_size >= 0.0:
                r_cg += torch.randn(r_cg.size()) * noise_size
            else:
                noise_size = 0.0
        else:
            noise_size = 0.0
        noise_size = torch.full((cg.n_residue,), noise_size)
        #
        geom_s = cg.get_geometry(r_cg, cg.continuous, cg.atom_mask_cg, pca=True)
        f_in, n_scalar, n_vector = cg.geom_to_feature(
            geom_s, noise_size=noise_size, dtype=self.dtype
        )
        #
        data = torch_geometric.data.Data(
            pos=torch.as_tensor(r_cg[cg.atom_mask_cg > 0.0], dtype=self.dtype)
        )
        data.pos0 = data.pos.clone()
        #
        data.f_in = f_in
        data.f_in_Irreps = f"{n_scalar}x0e + {n_vector}x1o"
        #
        global_frame = torch.as_tensor(geom_s["pca"], dtype=self.dtype).reshape(-1)
        data.global_frame = global_frame.repeat(cg.n_residue, 1)
        #
        data.chain_index = torch.as_tensor(cg.chain_index, dtype=int)
        data.residue_type = torch.as_tensor(cg.residue_index, dtype=torch.long)
        data.continuous = torch.as_tensor(cg.continuous, dtype=self.dtype)
        #
        data.atomic_radius = torch.as_tensor(cg.atomic_radius, dtype=self.dtype)
        data.atomic_mass = torch.as_tensor(cg.atomic_mass, dtype=self.dtype)
        data.input_atom_mask = torch.as_tensor(cg.atom_mask_cg, dtype=self.dtype)
        data.output_atom_mask = torch.as_tensor(cg.atom_mask, dtype=self.dtype)
        data.output_xyz = torch.as_tensor(cg.R[frame_index], dtype=self.dtype)
        #
        if self.get_structure_information:
            cg.get_structure_information()
            data.correct_bb = torch.as_tensor(cg.bb[frame_index], dtype=self.dtype)
            data.correct_torsion = torch.as_tensor(cg.torsion[frame_index], dtype=self.dtype)
            data.torsion_mask = torch.as_tensor(cg.torsion_mask, dtype=self.dtype)
        #
        return data

    def rotate_randomly(self, cg):
        random_rotation = e3nn.o3.rand_matrix().numpy()
        #
        out = copy.deepcopy(cg)
        out.R_cg = out.R_cg @ random_rotation.T
        out.R = out.R @ random_rotation.T
        return out


def create_topology_from_data(data: torch_geometric.data.Data) -> mdtraj.Topology:
    top = mdtraj.Topology()
    #
    chain_prev = -1
    resSeq = 0
    for i_res in range(data.residue_type.size(0)):
        chain_index = data.chain_index[i_res]
        if chain_index != chain_prev:
            chain_prev = chain_index
            resSeq = 0
            top_chain = top.add_chain()
        #
        resSeq += 1
        residue_type_index = int(data.residue_type[i_res])
        residue_name = AMINO_ACID_s[residue_type_index]
        residue_name_std = AMINO_ACID_REV_s.get(residue_name, residue_name)
        if residue_name == "UNK":
            continue
        ref_res = residue_s[residue_name]
        top_residue = top.add_residue(residue_name_std, top_chain, resSeq)
        #
        for i_atm, atom_name in enumerate(ref_res.atom_s):
            if data.output_atom_mask[i_res, i_atm] > 0.0:
                element = mdtraj.core.element.Element.getBySymbol(atom_name[0])
                top.add_atom(atom_name, element, top_residue)
    return top


def create_topology_from_batch(
    batch: torch_geometric.data.Batch,
) -> List[mdtraj.Topology]:
    top_s = list(map(create_topology_from_data, batch.to_data_list()))
    return top_s


def create_trajectory_from_batch(
    batch: torch_geometric.data.Batch,
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
    for idx, data in enumerate(batch.to_data_list()):
        top = create_topology_from_data(data)
        mask = data.output_atom_mask.cpu().detach().numpy()
        #
        xyz = []
        if write_native:
            xyz.append(data.output_xyz.cpu().detach().numpy()[mask > 0.0])
        else:
            mask = np.ones_like(mask)
        #
        if output is not None:
            start = int(batch._slice_dict["output_atom_mask"][idx])
            end = int(batch._slice_dict["output_atom_mask"][idx + 1])
            xyz.append(R[start:end][mask > 0.0])
        xyz = np.array(xyz)
        #
        traj = mdtraj.Trajectory(xyz=xyz, topology=top)
        traj_s.append(traj)
    return traj_s


def test():
    base_dir = BASE / "pdb.processed"
    pdblist = BASE / "pdb/pdblist"
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
    train_loader = torch_geometric.loader.DataLoader(
        train_set, batch_size=5, shuffle=True, num_workers=1
    )
    batch = next(iter(train_loader))
    # for batch in train_loader:
    #     traj_s = create_trajectory_from_batch(
    #         batch, batch.output_xyz, write_native=True
    #     )
    r = batch.output_xyz
    mass = batch.atomic_mass
    _pos_new = cg_model.convert_to_cg_tensor(r, mass)
    print(
        torch.allclose(
            batch.pos,
            _pos_new[batch.input_atom_mask == 1.0],
            rtol=EQUIVARIANT_TOLERANCE,
            atol=EQUIVARIANT_TOLERANCE,
        )
    )
    #
    for k in range(batch.num_graphs):
        selected = batch.batch == k
        pos_new = _pos_new[selected]
        #
        geom_s = cg_model.get_geometry(
            pos_new, batch.continuous[selected], batch.input_atom_mask[selected]
        )
        f_in, _, _ = cg_model.geom_to_feature(
            geom_s, batch.f_in[selected, 15], dtype=batch.f_in.dtype
        )
        f_in = f_in.to(f_in.device)
        #
        print(
            torch.allclose(
                f_in[:, 1:],
                batch.f_in[selected, 1:],
                rtol=0.005,
                atol=0.005,
            )
        )
        delta = torch.abs(f_in[:, 1:] - batch.f_in[selected, 1:])
        print(delta.max())
        print(delta.max(axis=0))


if __name__ == "__main__":
    test()

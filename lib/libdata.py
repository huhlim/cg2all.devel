#!/usr/bin/env python

# %%
import numpy as np
import pathlib
import functools
import itertools
import mdtraj
from typing import List

import torch
import torch_geometric

import libcg
from libconfig import BASE, DTYPE
from residue_constants import AMINO_ACID_s, residue_s

# %%
class PDBset(torch_geometric.data.Dataset):
    def __init__(
        self,
        basedir: str,
        pdblist: List[str],
        cg_model,
        noise_level=0.0,
        get_structure_information=False,
    ):
        super().__init__()
        #
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

    def __len__(self):
        return self.n_pdb

    def __getitem__(self, index) -> torch_geometric.data.Data:
        pdb_id = self.pdb_s[index]
        pdb_fn = self.basedir / f"{pdb_id}.pdb"
        #
        cg = self.cg_model(pdb_fn)
        frame_index = np.random.randint(cg.n_frame)
        #
        r = cg.R_cg[frame_index]
        dr = np.zeros((r.shape[0] + 1, 3))  # shape=(Nres+1, 3)
        dr[1:-1] = r[:-1, 0, :] - r[1:, 0, :]
        dr[1:-1] /= np.linalg.norm(dr[1:-1], axis=1)[:, None]
        dr[:-1][cg.continuous == 0.0] = 0.0
        if self.noise_level > 0.0:
            noise_size = np.random.normal(loc=self.noise_level, scale=self.noise_level / 2.0)
            if noise_size >= 0.0:
                r += np.random.normal(scale=noise_size, size=r.shape)
            else:
                noise_size = 0.0
        else:
            noise_size = 0.0
        #
        data = torch_geometric.data.Data(
            pos=torch.tensor(r[cg.atom_mask_cg == 1.0], dtype=DTYPE)
        )
        #
        # features for each residue
        f_in = [[], []]  # 0d, 1d
        # 0d
        # one-hot encoding of residue type
        index = cg.bead_index[cg.atom_mask_cg == 1.0]
        f_in[0].append(np.eye(cg.max_bead_type)[index])
        # noise-level
        f_in[0].append(np.full((r.shape[0], 1), noise_size))
        f_in[0] = torch.tensor(np.concatenate(f_in[0], axis=1), dtype=DTYPE)
        #
        # 1d: unit vectors from adjacent residues to the current residue
        f_in[1].append(dr[:-1, None])
        f_in[1].append(-dr[1:, None])
        f_in[1] = torch.tensor(np.concatenate(f_in[1], axis=1), dtype=DTYPE)
        f_in = torch.cat(
            [
                f_in[0],
                f_in[1].reshape(f_in[1].shape[0], -1),
            ],
            dim=1,
        )
        data.f_in = f_in
        #
        data.chain_index = torch.tensor(cg.chain_index, dtype=int)
        data.residue_type = torch.tensor(cg.residue_index, dtype=int)
        data.continuous = torch.tensor(cg.continuous, dtype=DTYPE)
        data.output_atom_mask = torch.tensor(cg.atom_mask, dtype=DTYPE)
        data.output_xyz = torch.tensor(cg.R[frame_index], dtype=DTYPE)
        #
        if self.get_structure_information:
            cg.get_structure_information()
            data.correct_bb = torch.tensor(cg.bb[frame_index], dtype=DTYPE)
            data.correct_torsion = torch.tensor(cg.torsion[frame_index], dtype=DTYPE)
            data.torsion_mask = torch.tensor(cg.torsion_mask, dtype=DTYPE)
        return data


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
        if residue_name == "UNK":
            continue
        ref_res = residue_s[residue_name]
        top_residue = top.add_residue(residue_name, top_chain, resSeq)
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


# %%
def test():
    base_dir = BASE / "pdb.processed"
    pdblist = BASE / "pdb/pdblist"
    cg_model = functools.partial(libcg.ResidueBasedModel, center_of_mass=True)
    #
    train_set = PDBset(base_dir, pdblist, cg_model, noise_level=0.5)
    train_loader = torch_geometric.loader.DataLoader(
        train_set, batch_size=5, shuffle=True, num_workers=1
    )
    for batch in train_loader:
        traj_s = create_trajectory_from_batch(
            batch, batch.output_xyz, write_native=True
        )


if __name__ == "__main__":
    test()

# %%

#!/usr/bin/env python

import torch

from residue_constants import ATOM_INDEX_N, ATOM_INDEX_CA, ATOM_INDEX_C, BOND_LENGTH0
from libloss import v_size


def rmsd_CA(R, R_ref):
    return torch.sqrt(
        torch.mean(torch.pow(R[:, ATOM_INDEX_CA, :] - R_ref[:, ATOM_INDEX_CA, :], 2))
    )


def rmsd_rigid(R, R_ref):
    return torch.sqrt(torch.mean(torch.pow(R[:, :3] - R_ref[:, :3], 2)))


def rmsd_all(R, R_ref, mask):
    dr_sq = torch.sum(torch.pow(R - R_ref, 2) * mask[..., None])
    return torch.sqrt(dr_sq / mask.sum())


def bonded_energy(R, is_continuous):
    bonded = is_continuous[1:]
    n_bonded = torch.sum(bonded)
    # vector: -C -> N
    v1 = R[1:, ATOM_INDEX_N, :] - R[:-1, ATOM_INDEX_C, :]
    #
    # bond lengths
    d1 = v_size(v1)
    bond_energy = torch.sum(torch.pow(d1 - BOND_LENGTH0, 2) * bonded) / n_bonded
    return bond_energy

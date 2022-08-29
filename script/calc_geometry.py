#!/usr/bin/env python

import os
import sys
import pathlib
from string import ascii_uppercase as CHAIN_IDs

import numpy as np
import mdtraj

v_size = lambda v: np.linalg.norm(v, axis=-1)

BOND_LENGTH0 = 1.345
BOND_ANGLE0 = (120.0, 116.5)
BOND_LENGTH_TOL = 0.1
BOND_ANGLE_TOL = 10.0
OMEGA_TOL = 15.0


def run(_pdb_fn):
    pdb_fn = pathlib.Path(_pdb_fn)
    out_fn = pdb_fn.parent / f"{pdb_fn.stem}.geom.dat"
    #
    traj = mdtraj.load(_pdb_fn, standard_names=False)
    #
    n_frames = traj.n_frames
    n_residues = traj.top.n_residues
    #
    atom_N = traj.top.select("name N")
    atom_CA = traj.top.select("name CA")
    atom_C = traj.top.select("name C")
    #
    data = {}
    #
    data["b_len"] = np.zeros((n_frames, n_residues, 1))
    data["b_len"][:, :-1, 0] = (
        v_size(traj.xyz[:, atom_C][:, :-1] - traj.xyz[:, atom_N][:, 1:]) * 10.0
    )
    #
    data["b_ang"] = np.zeros((n_frames, n_residues, 2))
    index_s = np.array([atom_CA[:-1], atom_C[:-1], atom_N[1:]]).T
    data["b_ang"][:, :-1, 0] = np.rad2deg(mdtraj.compute_angles(traj, index_s))
    index_s = np.array([atom_C[:-1], atom_N[1:], atom_CA[1:]]).T
    data["b_ang"][:, :-1, 1] = np.rad2deg(mdtraj.compute_angles(traj, index_s))
    #
    data["t_ang"] = np.zeros((n_frames, n_residues, 7))  # phi/psi/omg/chi_s
    atom_s, phi_s = mdtraj.compute_phi(traj)
    for atom, phi in zip(atom_s[:, 1], phi_s.T):
        i = traj.top.atom(atom).residue.index
        data["t_ang"][:, i, 0] = np.rad2deg(phi)
    atom_s, psi_s = mdtraj.compute_psi(traj)
    for atom, psi in zip(atom_s[:, 1], psi_s.T):
        i = traj.top.atom(atom).residue.index
        data["t_ang"][:, i, 1] = np.rad2deg(psi)
    atom_s, omg_s = mdtraj.compute_omega(traj)
    for atom, omg in zip(atom_s[:, 1], omg_s.T):
        i = traj.top.atom(atom).residue.index
        data["t_ang"][:, i, 2] = np.rad2deg(omg)
    atom_s, chi_s = mdtraj.compute_chi1(traj)
    for atom, chi in zip(atom_s[:, 1], chi_s.T):
        i = traj.top.atom(atom).residue.index
        data["t_ang"][:, i, 3] = np.rad2deg(chi)
    atom_s, chi_s = mdtraj.compute_chi2(traj)
    for atom, chi in zip(atom_s[:, 1], chi_s.T):
        i = traj.top.atom(atom).residue.index
        data["t_ang"][:, i, 4] = np.rad2deg(chi)
    atom_s, chi_s = mdtraj.compute_chi3(traj)
    for atom, chi in zip(atom_s[:, 1], chi_s.T):
        i = traj.top.atom(atom).residue.index
        data["t_ang"][:, i, 5] = np.rad2deg(chi)
    atom_s, chi_s = mdtraj.compute_chi4(traj)
    for atom, chi in zip(atom_s[:, 1], chi_s.T):
        i = traj.top.atom(atom).residue.index
        data["t_ang"][:, i, 6] = np.rad2deg(chi)
    #
    chain_prev = None
    chain_break = np.zeros(n_residues, dtype=bool)
    residue_s = []
    for residue in traj.top.residues:
        residue_index = residue.index
        chain_curr = residue.chain.index
        if residue_index > 0 and (chain_curr != chain_prev):  # new chain
            chain_break[residue_index - 1] = True
        chain_prev = chain_curr
        #
        residue_s.append(
            f"CHAIN {CHAIN_IDs[chain_curr]} : RESIDUE {residue.resSeq:4d} {residue.name}"
        )
    chain_break[-1] = True
    #
    data["b_len"][:, chain_break] = 0.0
    data["b_ang"][:, chain_break] = 0.0
    #
    fout = open(out_fn, "wt")
    sys.stdout.write(f"# PDB {pdb_fn}\n")
    fout.write(f"# PDB {pdb_fn}\n")
    for k_frame in range(n_frames):
        sys.stdout.write(f"# MODEL {k_frame}\n")
        fout.write(f"# MODEL {k_frame}\n")
        #
        for i, residue in enumerate(residue_s):
            out = [[], []]
            if n_frames > 1:
                out[0].append(f"MODEL {k_frame}")
                out[1].append(f"MODEL {k_frame}")
                out[0].append(":")
                out[1].append(":")
            #
            out[0].append(residue)
            out[1].append(residue)
            out[0].append(":")
            out[1].append(":")
            #
            b_len = data["b_len"][k_frame, i, 0]
            if chain_break[i] or np.abs(b_len - BOND_LENGTH0) < BOND_LENGTH_TOL:
                out[0].append(f"{b_len:6.3f}")
            else:
                out[0].append("\033[1;31m%6.3f\033[0m" % b_len)
            out[1].append(f"{b_len:6.3f}")
            out[0].append(":")
            out[1].append(":")
            #
            for k, b_ang in enumerate(data["b_ang"][k_frame, i]):
                if chain_break[i] or np.abs(b_ang - BOND_ANGLE0[k]) < BOND_ANGLE_TOL:
                    out[0].append(f"{b_ang:6.1f}")
                else:
                    out[0].append("\033[1;31m%6.1f\033[0m" % b_ang)
                out[1].append(f"{b_ang:6.1f}")
            out[0].append(":")
            out[1].append(":")
            #
            out[0].append(" ".join([f"{x:6.1f}" for x in data["t_ang"][k_frame, i, :2]]))
            out[1].append(" ".join([f"{x:6.1f}" for x in data["t_ang"][k_frame, i, :2]]))
            omg = data["t_ang"][k_frame, i, 2]
            d_omg = np.min(np.abs([omg + 180.0, omg - 180.0, omg]))
            d_omg = np.min([d_omg, 360.0 - d_omg])
            if d_omg < OMEGA_TOL:
                out[0].append(f"{omg:6.1f}")
            else:
                out[0].append("\033[1;31m%6.1f\033[0m" % omg)
            out[1].append(f"{omg:6.1f}")
            out[0].append(":")
            out[1].append(":")
            #
            out[0].append(" ".join([f"{x:6.1f}" for x in data["t_ang"][k_frame, i, 3:]]))
            out[1].append(" ".join([f"{x:6.1f}" for x in data["t_ang"][k_frame, i, 3:]]))
            sys.stdout.write(" ".join(out[0]) + "\n")
            fout.write(" ".join(out[1]) + "\n")
            if chain_break[i]:
                sys.stdout.write("TER\n")
                fout.write("TER\n")
        sys.stdout.write("END\n\n")
        fout.write("END\n")


def main():
    for pdb_fn in sys.argv[1:]:
        run(pdb_fn)


if __name__ == "__main__":
    main()

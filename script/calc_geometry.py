#!/usr/bin/env python

import os
import sys
import pathlib
from string import ascii_uppercase as CHAIN_IDs
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import mdtraj

try:
    from tqdm.contrib.concurrent import process_map
except:
    process_map = None
    import multiprocessing

N_PROC = int(os.getenv("OMP_NUM_THREADS", "8"))
v_size = lambda v: np.linalg.norm(v, axis=-1)


def run(_pdb_fn):
    try:
        _run(_pdb_fn)
    except:
        sys.stderr.write(f"ERROR: {_pdb_fn}\n")


def _run(_pdb_fn):
    pdb_fn = pathlib.Path(_pdb_fn)
    out_fn = pdb_fn.parent / f"{pdb_fn.stem}.geom.dat"
    #
    traj = mdtraj.load(_pdb_fn, standard_names=True)
    load_index = traj.top.select("protein or (resname HSD or resname HSE or resname MSE)")
    traj = traj.atom_slice(load_index)
    #
    atom_N = traj.top.select("name N")
    atom_CA = traj.top.select("name CA")
    atom_C = traj.top.select("name C")
    #
    occ = np.zeros((3, traj.top.n_residues))
    for i, atom_s in enumerate([atom_N, atom_CA, atom_C]):
        for atom in atom_s:
            i_res = traj.top.atom(atom).residue.index
            occ[i, i_res] += 1
    res_exclude = " or ".join([f"resid {r}" for r in np.where(occ.sum(0) != 3)[0]])
    if len(res_exclude) > 0:
        load_index = traj.top.select(f"not ({res_exclude})")
        traj = traj.atom_slice(load_index)
        atom_N = traj.top.select("name N")
        atom_CA = traj.top.select("name CA")
        atom_C = traj.top.select("name C")
    #
    n_frames = traj.n_frames
    n_residues = traj.top.n_residues
    #
    data = {}
    #
    data["ss"] = mdtraj.compute_dssp(traj, simplified=True)
    data["asa"] = np.zeros((n_frames, n_residues), dtype=int)
    for k_frame in range(n_frames):
        d_ij = v_size(traj.xyz[k_frame, atom_CA][None, :] - traj.xyz[k_frame, atom_CA][:, None])
        residue_index, n_count = np.unique(np.where(d_ij < 1.0)[0], return_counts=True)
        data["asa"][k_frame, residue_index] = n_count - 1
    #
    data["b_len"] = np.zeros((n_frames, n_residues, 3))
    data["b_len"][:, :-1, 0] = (
        v_size(traj.xyz[:, atom_C][:, :-1] - traj.xyz[:, atom_N][:, 1:]) * 10.0
    )
    data["b_len"][:, :, 1] = v_size(traj.xyz[:, atom_N] - traj.xyz[:, atom_CA]) * 10.0
    data["b_len"][:, :, 2] = v_size(traj.xyz[:, atom_CA] - traj.xyz[:, atom_C]) * 10.0
    #
    data["b_ang"] = np.zeros((n_frames, n_residues, 3))
    index_s = np.array([atom_CA[:-1], atom_C[:-1], atom_N[1:]]).T
    data["b_ang"][:, :-1, 0] = np.rad2deg(mdtraj.compute_angles(traj, index_s))
    index_s = np.array([atom_C[:-1], atom_N[1:], atom_CA[1:]]).T
    data["b_ang"][:, :-1, 1] = np.rad2deg(mdtraj.compute_angles(traj, index_s))
    index_s = np.array([atom_N, atom_CA, atom_C]).T
    data["b_ang"][:, :, 2] = np.rad2deg(mdtraj.compute_angles(traj, index_s))
    #
    data["t_ang"] = np.full((n_frames, n_residues, 7), 1e3)  # phi/psi/omg/chi_s
    for k, func in enumerate(
        [
            mdtraj.compute_phi,
            mdtraj.compute_psi,
            mdtraj.compute_omega,
            mdtraj.compute_chi1,
            mdtraj.compute_chi2,
            mdtraj.compute_chi3,
            mdtraj.compute_chi4,
        ]
    ):
        atom_s, angle_s = func(traj)
        for atom, angle in zip(atom_s[:, 1], angle_s.T):
            i = traj.top.atom(atom).residue.index
            data["t_ang"][:, i, k] = np.rad2deg(angle)
    #
    chain_prev = None
    res_prev = None
    chain_break = np.zeros(n_residues, dtype=bool)
    resName_s = []
    residue_s = []
    for residue in traj.top.residues:
        chain_curr = residue.chain.index
        res_curr = residue.resSeq
        residue_index = residue.index
        if residue_index > 0 and (chain_curr != chain_prev or res_curr - res_prev > 1):
            chain_break[residue_index - 1] = True
        chain_prev = chain_curr
        res_prev = res_curr
        resName_s.append(residue.name)
        #
        chain_id = CHAIN_IDs[chain_curr % len(CHAIN_IDs)]
        residue_s.append(f"CHAIN {chain_id} : RESIDUE {residue.resSeq:4d} {residue.name}")
    chain_break[-1] = True
    #
    data["b_len"][:, chain_break, 0] = 0.0
    data["b_ang"][:, chain_break, :2] = 0.0
    #
    fout = open(out_fn, "wt")
    fout.write(f"# PDB {pdb_fn}\n")
    #
    for k_frame in range(n_frames):
        fout.write(f"# MODEL {k_frame}\n")
        #
        for i, (resName, residue) in enumerate(zip(resName_s, residue_s)):
            out = []
            out.append(f"MODEL {k_frame}")
            out.append(":")
            #
            out.append(residue)
            out.append(":")
            #
            out.append(
                {"H": "HELIX", "E": "SHEET", "C": "COIL ", "NA": "COIL "}[data["ss"][k_frame, i]]
            )
            out.append(f"{data['asa'][k_frame, i]:2d}")
            out.append(":")
            #
            out.append(" ".join([f"{x:6.3f}" for x in data["b_len"][k_frame, i]]))
            out.append(":")
            #
            out.append(" ".join([f"{x:6.1f}" for x in data["b_ang"][k_frame, i]]))
            out.append(":")
            #
            out.append(" ".join([f"{x:6.1f}" for x in data["t_ang"][k_frame, i, :3]]))
            out.append(":")
            #
            out.append(" ".join([f"{x:6.1f}" for x in data["t_ang"][k_frame, i, 3:]]))
            #
            fout.write(" ".join(out) + "\n")
            if chain_break[i]:
                fout.write("TER\n")
        #
        fout.write("END\n")
    fout.close()


def main():
    pdb_fn_s = [fn for fn in sys.argv[1:] if fn.endswith(".pdb")]
    if len(pdb_fn_s) == 0:
        return
    #
    n_proc = min(N_PROC, len(pdb_fn_s))
    if process_map is None:
        with multiprocessing.Pool(n_proc) as pool:
            pool.map(run, pdb_fn_s)
    else:
        process_map(run, pdb_fn_s, max_workers=n_proc)


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import os
import sys
import pathlib
import tempfile

import mdtraj
import numpy as np
import subprocess as sp
from string import ascii_uppercase as CHAIN_IDs
from string import ascii_uppercase as INSCODEs

CHAIN_BREAKs = 0.5  # nm

os.environ["OPENMM_CPU_THREADS"] = "1"


def call(cmd, stdout=sp.DEVNULL):
    # print(" ".join(cmd))
    sp.call(cmd, stdout=stdout, stderr=sp.DEVNULL)


def detect_ssbond(pdb_fn, pdb):
    chain_s = []
    ssbond_from_pdb = []
    with open(pdb_fn) as fp:
        for line in fp:
            if line.startswith("SSBOND"):
                cys_0 = (line[15], line[17:22].strip())
                cys_1 = (line[29], line[31:36].strip())
                if cys_0 == cys_1:
                    sys.stderr.write(f"WARNING: invalid SSBOND found {pdb_fn}\n")
                    continue
                ssbond_from_pdb.append((cys_0, cys_1))
            elif line.startswith("ATOM"):
                chain_id = line[21]
                if chain_id not in chain_s:
                    chain_s.append(chain_id)
    #
    # find residue.index
    ssbond_s = []
    for cys_s in ssbond_from_pdb:
        residue_index = []
        for chain_id, resSeq in cys_s:
            chain_index = chain_s.index(chain_id)
            if resSeq[-1] in INSCODEs:
                index = pdb.top.select(f"chainid {chain_index} and resSeq '{resSeq}' and name SG")
            else:
                index = pdb.top.select(f"chainid {chain_index} and resSeq {resSeq} and name SG")
            if index.shape[0] == 1:
                residue_index.append(pdb.top.atom(index[0]).residue.index)
        residue_index = sorted(residue_index)
        if len(residue_index) == 2 and residue_index not in ssbond_s:
            ssbond_s.append(residue_index)
    return ssbond_s


def write_SSBOND(pdb_fn, top, ssbond_s):
    SSBOND = "SSBOND  %2d CYS %s %5s   CYS %s %5s\n"
    wrt = []
    for disu_no, ssbond in enumerate(ssbond_s):
        cys_0 = top.residue(ssbond[0])
        cys_1 = top.residue(ssbond[1])
        #
        if isinstance(cys_0.resSeq, int):
            cys_0_resSeq = f"{cys_0.resSeq:4d} "
        else:
            cys_0_resSeq = f"{cys_0.resSeq:>5s}"
        if isinstance(cys_1.resSeq, int):
            cys_1_resSeq = f"{cys_1.resSeq:4d} "
        else:
            cys_1_resSeq = f"{cys_1.resSeq:>5s}"
        #
        wrt.append(
            SSBOND
            % (
                disu_no + 1,
                CHAIN_IDs[cys_0.chain.index],
                cys_0_resSeq,
                CHAIN_IDs[cys_1.chain.index],
                cys_1_resSeq,
            )
        )
    #
    n_atoms = [0]
    model_s = [[]]
    has_model = False
    with open(pdb_fn) as fp:
        for line in fp:
            if line.startswith("MODEL"):
                has_model = True
                if len(model_s[-1]) > 0:
                    model_s.append([])
                    n_atoms.append(0)
                continue
            elif line.startswith("CONECT"):
                continue
            model_s[-1].append(line)
            if line.startswith("ATOM"):
                n_atoms[-1] += 1

    with open(pdb_fn, "wt") as fout:
        model_no = -1
        for n_atom, model in zip(n_atoms, model_s):
            if n_atom == 0:
                continue
            model_no += 1
            if has_model:
                fout.write(f"MODEL   {model_no:5d}\n")
            fout.writelines(wrt)
            fout.writelines(model)


def get_new_residue_numbers(pdb):
    calphaIndex = pdb.top.select("name CA")
    #
    seg_no = 0
    chain_index = -1
    resNo_s = [{}, {}]
    segNo_s = {}
    for i in calphaIndex:
        residue = pdb.top.atom(i).residue
        if residue.chain.index != chain_index:
            i_res = 0
            xyz_prev = None
            seg_no += 1
        #
        xyz = pdb.xyz[0, i]
        if xyz_prev is not None:
            d = np.linalg.norm(xyz - xyz_prev)
            if d > CHAIN_BREAKs:
                i_res += 1
                seg_no += 1
        #
        i_res += 1
        chain_index = residue.chain.index
        xyz_prev = xyz
        #
        resNo_s[0][(chain_index, residue.resSeq)] = i_res
        resNo_s[1][(chain_index, i_res)] = residue.resSeq
        segNo_s[residue.index] = seg_no

    return resNo_s, segNo_s


def update_residue_number(in_pdb, out_pdb, resNo_s=None, segNo_s=None, ssbond_s=None):
    pdb = mdtraj.load(in_pdb)

    for residue in pdb.top.residues:
        if segNo_s is not None:
            residue.segment_id = "P%03d" % (segNo_s[residue.index])
        if resNo_s is not None:
            chain_index = residue.chain.index
            resSeq_new = resNo_s[(chain_index, residue.resSeq)]
            residue.resSeq = resSeq_new
    pdb.save(out_pdb)
    #
    if ssbond_s is not None and len(ssbond_s) > 0:
        write_SSBOND(out_pdb, pdb.top, ssbond_s)


def run_scwrl(in_pdb, out_pdb):
    if not os.path.exists(out_pdb):
        call(["scwrl4", "-i", str(in_pdb), "-o", str(out_pdb)])


def run_reduce(in_pdb, out_pdb):
    if not os.path.exists(out_pdb):
        with open(out_pdb, "wt") as fout:
            call(["reduce.sh", "-Quiet", "-BUILD", in_pdb], stdout=fout)
    with open(out_pdb) as fp:
        n_atoms = 0
        for line in fp:
            if line.startswith("ATOM"):
                n_atoms += 1
    return n_atoms > 100


def run_process_pdb(in_pdb, out_pdb):
    if not os.path.exists(out_pdb):
        BASE = f"{os.getenv('work')}/ml/cg2all"
        EXEC = f"{BASE}/script/process_pdb.py"
        call([EXEC, "-i", in_pdb, "-o", out_pdb])


def run_minimize(in_pdb, out_pdb):
    if not os.path.exists(out_pdb):
        BASE = f"{os.getenv('work')}/ml/cg2all"
        EXEC = f"{BASE}/script/minimize_structure.py"
        FFs = [f"{BASE}/data/toppar/par_all36m_prot.prm", f"{BASE}/data/toppar/top_all36_prot.rtf"]
        call([EXEC, out_pdb[:-4], in_pdb, "--toppar"] + FFs, stdout=sys.stdout)


def main():
    in_pdb = pathlib.Path(sys.argv[1]).resolve()
    out_pdb = pathlib.Path("augment").resolve() / in_pdb.name
    min_pdb = pathlib.Path("augment_min").resolve() / in_pdb.name
    #
    if len(sys.argv) > 2:
        run_dir = pathlib.Path(sys.argv[2])
        run_dir.mkdir(exist_ok=True)
        os.chdir(run_dir)
    else:
        run_dir = tempfile.TemporaryDirectory(prefix="augment.")
        os.chdir(run_dir.name)
    #
    pdb = mdtraj.load(str(in_pdb))
    ssbond_s = detect_ssbond(in_pdb, pdb)
    resNo_s, segNo_s = get_new_residue_numbers(pdb)
    #
    update_residue_number(str(in_pdb), "input.pdb", resNo_s=resNo_s[0], ssbond_s=ssbond_s)
    #
    run_scwrl("input.pdb", "scwrl.pdb")
    if run_reduce("scwrl.pdb", "reduce.pdb"):
        out_fn = "reduce.pdb"
    else:
        out_fn = "scwrl.pdb"
    #
    run_process_pdb(out_fn, "output.pdb")
    #
    update_residue_number("output.pdb", out_pdb, resNo_s=resNo_s[1], ssbond_s=ssbond_s)
    #
    update_residue_number("output.pdb", "md_input.pdb", segNo_s=segNo_s, ssbond_s=ssbond_s)
    run_minimize("md_input.pdb", "md_output.pdb")
    run_process_pdb("md_output.pdb", "minimized.pdb")
    #
    update_residue_number("minimized.pdb", min_pdb, resNo_s=resNo_s[1], ssbond_s=ssbond_s)


if __name__ == "__main__":
    main()

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

from augment import *

os.environ["OPENMM_CPU_THREADS"] = "1"


def run_reduce(in_pdb, out_pdb, trim=False):
    if trim:
        with open("reduce_trim.pdb", "wt") as fout:
            call(["reduce.sh", "-Quiet", "-TRIM", in_pdb], stdout=fout)
        in_pdb = "reduce_trim.pdb"

    if not os.path.exists(out_pdb):
        with open(out_pdb, "wt") as fout:
            call(["reduce.sh", "-Quiet", "-BUILD", in_pdb], stdout=fout)
    with open(out_pdb) as fp:
        n_atoms = 0
        for line in fp:
            if line.startswith("ATOM"):
                n_atoms += 1
    return n_atoms > 100


def main():
    in_pdb = pathlib.Path(sys.argv[1]).resolve()
    out_pdb = pathlib.Path("augment+FLIP").resolve() / in_pdb.name
    min_pdb = pathlib.Path("augment_min+FLIP").resolve() / in_pdb.name
    if out_pdb.exists() and min_pdb.exists():
        return
    #
    if len(sys.argv) > 2:
        run_dir = pathlib.Path(sys.argv[2])
        run_dir.mkdir(exist_ok=True)
        os.chdir(run_dir)
    else:
        run_dir = tempfile.TemporaryDirectory(prefix="augment.")
        os.chdir(run_dir.name)
    #
    trim_reduce = False
    if len(sys.argv) > 3:
        if sys.argv[3].strip() in ["yes", "true", "t", "y"]:
            trim_reduce = True
    #
    pdb = mdtraj.load(str(in_pdb))
    ssbond_s = detect_ssbond(in_pdb, pdb)
    resNo_s, segNo_s = get_new_residue_numbers(pdb)
    #
    update_residue_number(
        str(in_pdb), "input.pdb", resNo_s=resNo_s[0], ssbond_s=ssbond_s
    )
    #
    run_scwrl("input.pdb", "scwrl.pdb")
    if run_reduce("scwrl.pdb", "reduce.pdb", trim=trim_reduce):
        out_fn = "reduce.pdb"
    else:
        out_fn = "scwrl.pdb"
    #
    run_process_pdb(out_fn, "output.pdb")
    #
    update_residue_number("output.pdb", out_pdb, resNo_s=resNo_s[1], ssbond_s=ssbond_s)
    #
    if not min_pdb.exists():
        update_residue_number(
            "output.pdb", "md_input.pdb", segNo_s=segNo_s, ssbond_s=ssbond_s
        )
        run_minimize("md_input.pdb", "md_output.pdb")
        run_process_pdb("md_output.pdb", "minimized.pdb")
        #
        update_residue_number(
            "minimized.pdb", min_pdb, resNo_s=resNo_s[1], ssbond_s=ssbond_s
        )


if __name__ == "__main__":
    main()

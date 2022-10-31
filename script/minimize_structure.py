#!/usr/bin/env python

import os
import sys
import json
import argparse
import pickle
import numpy as np

import warnings

warnings.filterwarnings("ignore")

import mdtraj

from openmm.unit import *
from openmm.openmm import *
from openmm.app import *

WORK_HOME = os.getenv("PIPE_HOME")
assert WORK_HOME is not None
sys.path.insert(0, "%s/bin" % WORK_HOME)
sys.path.insert(0, "%s/bin/exec" % WORK_HOME)

import path
from libcommon import *

from libcustom import *
from libmd import (
    generate_PSF,
    update_residue_name,
)


def construct_restraint(psf, crd, force_const, atom_s=["CA"]):
    rsr = CustomExternalForce("k0*d^2 ; d=periodicdistance(x,y,z, x0,y0,z0)")
    rsr.addPerParticleParameter("x0")
    rsr.addPerParticleParameter("y0")
    rsr.addPerParticleParameter("z0")
    rsr.addPerParticleParameter("k0")
    #
    for i, (atom, r) in enumerate(
        zip(psf.topology.atoms(), crd.positions.value_in_unit(nanometers))
    ):
        if atom.name not in atom_s:
            continue
        #
        mass = atom.element.mass
        param = [r.x, r.y, r.z]
        param.append(force_const * mass * kilocalories_per_mole / angstroms**2)
        rsr.addParticle(i, param)
    return rsr


def minimize(output_prefix, solv_fn, psf_fn, crd_fn, options):
    psf = CharmmPsfFile(psf_fn.short())
    crd = CharmmCrdFile(crd_fn.short())
    #
    pdb = mdtraj.load(solv_fn.short(), standard_names=False)
    #
    box = np.array(crd.positions.value_in_unit(nanometers), dtype=float)
    boxsize = np.max(box, 0) - np.min(box, 0) + 2.0
    psf.setBox(*boxsize)
    #
    translate = Quantity(boxsize / 2.0 - box.mean(axis=0), unit=nanometers)
    for i, pos in enumerate(crd.positions):
        crd.positions[i] = pos + translate
    #
    ff_file_s = options["ff"]["toppar"]
    ff = CharmmParameterSet(*ff_file_s)
    #
    sys = psf.createSystem(
        ff,
        nonbondedMethod=PME,
        switchDistance=0.8 * nanometers,
        nonbondedCutoff=1.0 * nanometers,
        constraints=HBonds,
    )
    #
    sys.addForce(construct_restraint(psf, crd, 1.0, ["N", "CA", "C", "O", "CB"]))
    #
    integrator = LangevinIntegrator(298.15 * kelvin, 1.0 / picosecond, 0.002 * picosecond)
    #
    simulation = Simulation(psf.topology, sys, integrator)
    simulation.context.setPositions(crd.positions)

    simulation.minimizeEnergy(maxIterations=1000)
    state = simulation.context.getState(
        getPositions=True,
        getVelocities=True,
        getForces=True,
        getEnergy=True,
        enforcePeriodicBox=True,
    )
    #
    boxinfo = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(nanometer)
    #
    out_pdb_fn = path.Path("%s.pdb" % output_prefix)
    xyz = state.getPositions() - translate
    with out_pdb_fn.open("wt") as fout:
        PDBFile.writeFile(simulation.topology, xyz, fout)


def run(init_pdb, output_prefix, options):
    pdb = mdtraj.load(init_pdb.short())
    update_residue_name(init_pdb, pdb)
    #
    psf_fn, crd_fn = generate_PSF(output_prefix, init_pdb, options, False)
    #
    minimize(output_prefix, init_pdb, psf_fn, crd_fn, options)


def main():
    arg = argparse.ArgumentParser(prog="equil")
    arg.add_argument(dest="output_prefix")
    arg.add_argument(dest="input_pdb")
    arg.add_argument("--input", dest="input_json", default=None)
    arg.add_argument("--toppar", dest="toppar", nargs="*", default=None)
    #
    if len(sys.argv) == 1:
        arg.print_help()
        return
    arg = arg.parse_args()
    #
    input_pdb = path.Path(arg.input_pdb)
    #
    if arg.input_json is None:
        options = {}
        options["ff"] = {}
    else:
        with open(arg.input_json) as fp:
            options = json.load(fp)
            for key in options:
                options[key] = JSONdeserialize(options[key])
    #
    if arg.toppar is not None:
        options["ff"]["toppar"] = arg.toppar
    #
    if "toppar" not in options["ff"]:
        raise KeyError("toppar is missing")
    #
    run(input_pdb, arg.output_prefix, options)


if __name__ == "__main__":
    main()

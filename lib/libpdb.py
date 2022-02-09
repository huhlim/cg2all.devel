#!/usr/bin/env python

#%%
# load modules
import sys
import numpy as np
import mdtraj
from numpy_basics import *
from residue_constants import *

# %%
class PDB(object):
    def __init__(self, pdb_fn, dcd_fn=None):
        pdb = mdtraj.load(pdb_fn, standard_names=False)
        load_index = pdb.top.select("protein")
        if dcd_fn is None:
            self.is_dcd = False
            self.traj = pdb.atom_slice(load_index)
        else:
            self.is_dcd = True
            self.traj = mdtraj.load(dcd_fn, top=pdb.top, atom_indicds=load_index)
        self.top = self.traj.top
        #
        self.n_frame = self.traj.n_frames
        self.n_residue = self.top.n_residues
        self.residue_name = []
        self.residue_index = np.zeros(self.n_residue, dtype=np.int16)
    def to_atom(self):
        self.R = np.zeros((self.n_frame, self.n_residue, MAX_ATOM, 3))
        self.atom_mask = np.zeros((self.n_residue, MAX_ATOM), dtype=np.float16)
        #
        for residue in self.top.residues:
            i_res = residue.index
            residue_name = AMINO_ACID_ALT_s.get(residue.name, residue.name)
            self.residue_name.append(residue_name)
            self.residue_index[i_res] = AMINO_ACID_s.index(residue_name)
            ref_res = residue_s[residue_name]
            #
            for atom in residue.atoms:
                atom_name = ATOM_NAME_ALT_s.get((residue_name, atom.name), atom.name)
                if atom_name in ['OXT', "H1", "H2", "H3"]:
                    continue
                if atom_name.startswith("D"):
                    continue
                if atom_name not in ref_res.atom_s:
                    sys.stderr.write(f"Unrecognized atom_name: {residue_name} {atom_name}\n")
                    continue
                i_atm = ref_res.atom_s.index(atom_name)
                self.R[:, i_res, i_atm, :] = self.traj.xyz[:, atom.index, :]
                self.atom_mask[i_res, i_atm] = 1.

    # create a new topology based on the standard residue definitions
    def create_new_topology(self):
        top = mdtraj.Topology()
        #
        i_res = -1
        for chain in self.top.chains:
            top_chain = top.add_chain()
            #
            for residue in chain.residues:
                i_res += 1
                top_residue = top.add_residue(residue.name, top_chain, residue.resSeq)
                #
                for i_atm, atom_name in enumerate(residue_s[residue.name].atom_s):
                    if self.atom_mask[i_res, i_atm] == 0.:
                        continue
                    element = mdtraj.core.element.Element.getBySymbol(atom_name[0])
                    top.add_atom(atom_name, element, top_residue)
        return top
    
    # get backbone orientations by placing rigid body atoms (N, CA, C)
    def get_backbone_orientation(self, i_res, residue_name):
        # find BB orientations
        opr_s = [[], []]
        for k in range(self.n_frame):
            r = self.R[k, i_res, :3, :]
            frame = rigid_from_3points(r)
            opr_s[0].append(frame[0].T)
            opr_s[1].append(frame[1])
        opr_s = [np.array(opr_s[0]), np.array(opr_s[1])]
        #
        # place BB atoms
        t_ang0, atom_s, rigid = get_rigid_group_by_torsion(residue_name, "BB")
        rigid_s = [translate_and_rotate(rigid, opr_s[0][i], opr_s[1][i]) for i in range(self.n_frame)]
        return opr_s, atom_s, np.array(rigid_s)

    def write(self, pdb_fn, dcd_fn=None):
        top = self.create_new_topology()
        mask = np.where(self.atom_mask)
        xyz = self.R[:,mask[0],mask[1],:]
        #
        traj = mdtraj.Trajectory(xyz, top)
        traj.save(pdb_fn)
        #
        if dcd_fn is not None:
            traj = mdtraj.Trajectory(xyz[:1], top)
            traj.save(dcd_fn)
#!/usr/bin/env python

#%%
# load modules
import numpy as np
import mdtraj
from numpy_basics import *
from residue_constants import *
from libpdbname import *

# %%
class PDB(object):
    def __init__(self, pdb_fn):
        pdb = mdtraj.load(pdb_fn, standard_names=False)
        pdb = pdb.atom_slice(pdb.top.select("protein"))
        self.pdb = pdb
        self.top = pdb.top
        #
        self.n_frame = self.pdb.n_frames
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
                    print (residue_name, atom_name)
                    continue
                i_atm = ref_res.atom_s.index(atom_name)
                self.R[:, i_res, i_atm, :] = self.pdb.xyz[:, atom.index, :]
                self.atom_mask[i_res, i_atm] = 1.
    
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

    def make_atom_names_consistent(self):
        self.R_ideal = self.R.copy()
        for i_res in range(self.n_residue):
            residue_name = self.residue_name[i_res]
            ref_res = residue_s[residue_name]
            tor_s = torsion_s[residue_name]
            #
            opr_s = {}
            #
            # find BB orientations
            opr_bb, atom_s, rigid_s = self.get_backbone_orientation(i_res, residue_name)
            opr_s[("BB", 0)] = opr_bb
            for atom in atom_s:
                self.R_ideal[:,i_res,ref_res.atom_s.index(atom),:] = rigid_s[:,atom_s.index(atom),:]
            #
            amb = get_ambiguous_atom_list(residue_name, "BB")
            if residue_name == 'GLY':
                update_by_glycine_backbone_method(self.R, i_res, ref_res, amb, atom_s, rigid_s)
            #
            # update side chain atom names
            for tor in tor_s:
                if tor is None:
                    continue
                if tor.name in ['BB', 'PHI', 'PSI']:
                    continue
                #
                amb = get_ambiguous_atom_list(residue_name, tor.name, tor.index)
                if amb is None or amb.method == 'closest':
                    opr_sc, atom_s, rigid_s = update_by_closest_method(self.R, i_res, ref_res, tor, amb, opr_s)
                elif amb.method == 'permute':
                    opr_sc, atom_s, rigid_s = update_by_permute_method(self.R, i_res, ref_res, tor, amb, opr_s)
                elif amb.method == 'periodic':
                    opr_sc, atom_s, rigid_s = update_by_periodic_method(self.R, i_res, ref_res, tor, amb, opr_s)
                else:
                    raise ValueError("Unknown ambiguous method: %s" % amb.method)
                opr_s[(tor.name, tor.index)] = opr_sc
                #
                for atom in atom_s:
                    self.R_ideal[:,i_res,ref_res.atom_s.index(atom),:] = rigid_s[:,atom_s.index(atom),:]
            #
            # special torsion angles, only for Asn, Gln, Arg
            if residue_name in ['ASN', 'GLN']:
                update_by_special_method(self.R, i_res, ref_res)
            elif residue_name == 'ARG':
                update_by_guanidium_method(self.R, i_res, ref_res)


    # print out as a PDB file
    def to_pdb(self, fn):
        wrt = []
        i_atm = 0
        for i_res in range(self.n_residue):
            residue_number = self.top.residue(i_res).resSeq
            residue_name = self.residue_name[i_res]
            ref_res = residue_s[residue_name]
            for i,atom_name in enumerate(ref_res.atom_s):
                if not self.atom_mask[i_res,i]: continue
                i_atm += 1
                if len(atom_name) < 4:
                    atom_name = ' %-3s' % atom_name
                #R = self.R_ideal[0,i_res,i,:] * 10.
                R = self.R[0,i_res,i,:] * 10.
                wrt.append('ATOM  %5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s  \n' % \
                    (i_atm, atom_name, residue_name, 'A', residue_number, \
                    R[0], R[1], R[2], 1.0, 0.0, ' '))
        with open(fn, 'w') as f:
            f.write(''.join(wrt))

pdb = PDB("../pdb/1VII.pdb")
pdb.to_atom()
pdb.make_atom_names_consistent()
pdb.to_pdb('renamed.pdb')
# %%

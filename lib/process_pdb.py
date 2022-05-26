#!/usr/bin/env python
'''
This application renames atom names to make a consistent orientations
for permutable atoms
'''

#%%
# load modules
import sys
from numpy_basics import *
from residue_constants import *
from libpdb import PDB
from libpdbname import *
import argparse
import warnings
warnings.filterwarnings("ignore")

# %%
class ProcessPDB(PDB):
    def __init__(self, *arg, **kwarg):
        super().__init__(*arg, **kwarg)
    def make_atom_names_consistent(self):
        self.R_ideal = self.R.copy()
        for i_res in range(self.n_residue):
            residue_name = self.residue_name[i_res]
            if residue_name == 'UNK':
                continue
            ref_res = residue_s[residue_name]
            tor_s = torsion_s[residue_name]
            #
            opr_s = {}
            #
            # find BB orientations
            mask, opr_bb = self.get_backbone_orientation(i_res)
            if not mask: continue
            opr_s[("BB", 0)] = opr_bb
            #
            # place BB atoms
            t_ang0, atom_s, rigid = get_rigid_group_by_torsion(self.residue_name[i_res], "BB")
            rigid_s = translate_and_rotate(rigid, opr_bb[0], opr_bb[1])
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
                if tor.name in ['BB']:#, 'PHI', 'PSI']:
                    continue
                #
                amb = get_ambiguous_atom_list(residue_name, tor.name, tor.index)
                if amb is None or amb.method == 'closest':
                    opr_sc, atom_s, rigid_s = update_by_closest_method(self.R, self.atom_mask, i_res, ref_res, tor, amb, opr_s)
                elif amb.method == 'permute':
                    opr_sc, atom_s, rigid_s = update_by_permute_method(self.R, self.atom_mask, i_res, ref_res, tor, amb, opr_s)
                elif amb.method == 'periodic':
                    opr_sc, atom_s, rigid_s = update_by_periodic_method(self.R, self.atom_mask, i_res, ref_res, tor, amb, opr_s)
                else:
                    raise ValueError("Unknown ambiguous method: %s" % amb.method)
                if atom_s is None:
                    continue
                #
                opr_s[(tor.name, tor.index)] = opr_sc
                for atom in atom_s:
                    self.R_ideal[:,i_res,ref_res.atom_s.index(atom),:] = rigid_s[:,atom_s.index(atom),:]
            #
            # special torsion angles, only for Asn, Gln, Arg
            if residue_name in ['ASN', 'GLN']:
                update_by_special_method(self.R, self.atom_mask, i_res, ref_res)
            elif residue_name == 'ARG':
                update_by_guanidium_method(self.R, self.atom_mask, i_res, ref_res)

def main():
    arg = argparse.ArgumentParser(prog="process_pdb")
    arg.add_argument('-i', '--input', dest="in_pdb", help="input PDB file/topology file", required=True)
    arg.add_argument('-o', '--output', dest='output_fn', help="output file", required=True)
    arg.add_argument('-d', '--dcd', dest='dcd_fn', help="input trajectory file", default=None)
    if len(sys.argv) == 1:
        arg.print_help()
        return
    arg = arg.parse_args()
    #
    pdb = ProcessPDB(arg.in_pdb, dcd_fn=arg.dcd_fn)
    pdb.make_atom_names_consistent()
    pdb.write(pdb.R, arg.output_fn)

if __name__ == '__main__':
    main()

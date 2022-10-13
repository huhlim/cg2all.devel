#!/usr/bin/env python
"""
This application renames atom names to make a consistent orientations
for permutable atoms
"""

import sys
import argparse
import warnings
import pathlib

BASE = pathlib.Path(__file__).parents[1].resolve()
LIB_HOME = str(BASE / "lib")
sys.path.insert(0, LIB_HOME)

from numpy_basics import *
from residue_constants import *
from libpdb import PDB
from libpdbname import *

warnings.filterwarnings("ignore")


class ProcessPDB(PDB):
    def __init__(self, *arg, **kwarg):
        super().__init__(*arg, **kwarg)

    def check_amb_valid(self, i_res, amb, ref_res):
        atom_index_s = [
            ref_res.atom_s.index(atom) for atom in amb.atom_s if not atom.startswith("H")
        ]
        if len(atom_index_s) == 0:
            return True
        #
        mask = self.atom_mask_pdb[i_res, atom_index_s]
        if np.all(mask):
            return True
        else:
            self.atom_mask_pdb[i_res, atom_index_s] = 0.0
            return False

    def make_atom_names_consistent(self):
        self.R_ideal = self.R.copy()
        for i_res in range(self.n_residue):
            residue_name = self.residue_name[i_res]
            if residue_name == "UNK":
                continue
            ref_res = residue_s[residue_name]
            tor_s = torsion_s[residue_name]
            #
            opr_s = {}
            #
            # find BB orientations
            mask, opr_bb = self.get_backbone_orientation(i_res)
            if not mask:
                continue
            opr_s[("BB", 0)] = opr_bb
            #
            # place BB atoms
            t_ang0, atom_s, rigid = get_rigid_group_by_torsion(self.residue_name[i_res], "BB")
            rigid_s = translate_and_rotate(rigid, opr_bb[0], opr_bb[1])
            for atom in atom_s:
                self.R_ideal[:, i_res, ref_res.atom_s.index(atom), :] = rigid_s[
                    :, atom_s.index(atom), :
                ]
            #
            amb = get_ambiguous_atom_list(residue_name, "BB")
            if residue_name == "GLY" and self.check_amb_valid(i_res, amb, ref_res):
                update_by_glycine_backbone_method(self.R, i_res, ref_res, amb, atom_s, rigid_s)
            #
            # update side chain atom names
            for tor in tor_s:
                if tor is None:
                    continue
                if tor.name in ["BB"]:  # , 'PHI', 'PSI']:
                    continue
                #
                if tor.name == "XI":
                    amb = get_ambiguous_atom_list(residue_name, tor.name, tor.index, tor.sub_index)
                else:
                    amb = get_ambiguous_atom_list(residue_name, tor.name, tor.index)
                #
                # check if all ambiguous atoms are present
                if amb is not None:
                    if not self.check_amb_valid(i_res, amb, ref_res):
                        continue
                #
                if amb is None or amb.method in ["closest", "xi"]:
                    opr_sc, atom_s, rigid_s = update_by_closest_method(
                        self.R, self.atom_mask_pdb, i_res, ref_res, tor, amb, opr_s
                    )
                elif amb.method == "permute":
                    opr_sc, atom_s, rigid_s = update_by_permute_method(
                        self.R, self.atom_mask_pdb, i_res, ref_res, tor, amb, opr_s
                    )
                elif amb.method == "periodic":
                    opr_sc, atom_s, rigid_s = update_by_periodic_method(
                        self.R, self.atom_mask_pdb, i_res, ref_res, tor, amb, opr_s
                    )
                elif amb.method in ["amide", "guanidium"]:
                    continue
                else:
                    raise ValueError("Unknown ambiguous method: %s" % amb.method)
                #
                if atom_s is None:
                    continue
                #
                opr_s[(tor.name, tor.index)] = opr_sc
                for atom in atom_s:
                    atom_index = ref_res.atom_s.index(atom)
                    self.R_ideal[:, i_res, atom_index, :] = rigid_s[:, atom_s.index(atom), :]
            #
            # amide torsion angles, only for Asn, Gln, Arg
            if residue_name in ["ASN", "GLN"]:
                amb = get_ambiguous_atom_list(ref_res.residue_name, "amide")
                if self.check_amb_valid(i_res, amb, ref_res):
                    update_by_amide_method(self.R, self.atom_mask_pdb, i_res, ref_res, amb)

            elif residue_name == "ARG":
                update_by_guanidium_method(self.R, self.atom_mask_pdb, i_res, ref_res)


def main():
    arg = argparse.ArgumentParser(prog="process_pdb")
    arg.add_argument(
        "-i",
        "--input",
        dest="in_pdb",
        help="input PDB file/topology file",
        required=True,
    )
    arg.add_argument("-o", "--output", dest="output_fn", help="output file", required=True)
    arg.add_argument(
        "--indcd", dest="in_dcd_fn", help="input trajectory file", default=None, nargs="*"
    )
    arg.add_argument("--outdcd", dest="out_dcd_fn", help="input trajectory file", default=None)
    arg.add_argument("--stride", dest="stride", default=1, type=int)
    if len(sys.argv) == 1:
        arg.print_help()
        return
    arg = arg.parse_args()
    #
    pdb = ProcessPDB(arg.in_pdb, dcd_fn=arg.in_dcd_fn, stride=arg.stride)
    pdb.make_atom_names_consistent()
    pdb.write(pdb.R, arg.output_fn, dcd_fn=arg.out_dcd_fn)


if __name__ == "__main__":
    main()

#!/usr/bin/env python

# %%
# load modules
import sys
import numpy as np
import mdtraj
from string import ascii_uppercase as CHAIN_IDs
from numpy_basics import *
from residue_constants import *

# %%
class PDB(object):
    def __init__(self, pdb_fn, dcd_fn=None):
        # read protein
        pdb = mdtraj.load(pdb_fn, standard_names=False)
        load_index = pdb.top.select("protein or (resname HSD or resname HSE)")
        if dcd_fn is None:
            self.is_dcd = False
            self.traj = pdb.atom_slice(load_index)
        else:
            self.is_dcd = True
            self.traj = mdtraj.load(dcd_fn, top=pdb.top, atom_indices=load_index)
        self.top = self.traj.top
        #
        self.n_frame = self.traj.n_frames
        self.n_chain = self.top.n_chains
        self.n_residue = self.top.n_residues
        self.chain_index = np.array([r.chain.index for r in self.top.residues], dtype=np.int16)
        self.residue_name = []
        self.residue_index = np.zeros(self.n_residue, dtype=np.int16)
        #
        self.detect_ssbond(pdb_fn)
        #
        self.to_atom()
        self.get_continuity()

    def detect_ssbond(self, pdb_fn):
        chain_s = []
        ssbond_from_pdb = []
        with open(pdb_fn) as fp:
            for line in fp:
                if line.startswith("SSBOND"):
                    cys_0 = (line[15], line[17:21])
                    cys_1 = (line[29], line[31:35])
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
                index = self.top.select(f"chainid {chain_index} and resSeq {resSeq} and name SG")
                if index.shape[0] == 1:
                    residue_index.append(self.top.atom(index[0]).residue.index)
            if len(residue_index) == 2:
                ssbond_s.append(residue_index)
        self.ssbond_s = ssbond_s

    def to_atom(self, verbose=False):
        # set up
        #   - R
        #   - atom_mask
        #   - residue_name
        #   - residue_index

        self.R = np.zeros((self.n_frame, self.n_residue, MAX_ATOM, 3))
        self.atom_mask = np.zeros((self.n_residue, MAX_ATOM), dtype=np.float16)
        self.atom_mask_pdb = np.zeros((self.n_residue, MAX_ATOM), dtype=np.float16)
        self.atomic_radius = np.zeros((self.n_residue, MAX_ATOM, 2, 2), dtype=np.float16)
        self.atomic_mass = np.zeros((self.n_residue, MAX_ATOM), dtype=np.float16)
        #
        if len(self.ssbond_s) > 0:
            ssbond_s = np.concatenate(self.ssbond_s, dtype=int)
        else:
            ssbond_s = []
        #
        for residue in self.top.residues:
            i_res = residue.index
            residue_name = AMINO_ACID_ALT_s.get(residue.name, residue.name)
            if residue_name not in AMINO_ACID_s:
                residue_name = "UNK"
            #
            self.residue_name.append(residue_name)
            self.residue_index[i_res] = AMINO_ACID_s.index(residue_name)
            if residue_name == "UNK":
                continue
            ref_res = residue_s[residue_name]
            #
            for atom in residue.atoms:
                atom_name = ATOM_NAME_ALT_s.get((residue_name, atom.name), atom.name)
                if atom_name in [
                    "OXT",
                    "H2",
                    "H3",
                ]:  # at this moment, I ignore N/C-terminal specific atoms
                    continue
                if atom_name.startswith("D"):
                    continue
                if atom_name not in ref_res.atom_s:
                    if verbose:
                        sys.stderr.write(f"Unrecognized atom_name: {residue_name} {atom_name}\n")
                    continue
                i_atm = ref_res.atom_s.index(atom_name)
                self.R[:, i_res, i_atm, :] = self.traj.xyz[:, atom.index, :]
                self.atom_mask_pdb[i_res, i_atm] = 1.0
                self.atomic_mass[i_res, i_atm] = atom.element.mass
            #
            n_atom = len(ref_res.atom_s)
            self.atomic_radius[i_res, :n_atom] = ref_res.atomic_radius[:n_atom]
            self.atom_mask[i_res, :n_atom] = 1.0
            if i_res in ssbond_s:
                self.atom_mask[i_res, ref_res.atom_s.index("HG1")] = 0.0

    # get continuity information, whether it has a previous residue
    def get_continuity(self):
        # different chains
        self.continuous = self.chain_index[1:] == self.chain_index[:-1]

        # chain breaks
        dr = self.R[:, 1:, ATOM_INDEX_N] - self.R[:, :-1, ATOM_INDEX_C]
        d = v_size(dr).mean(axis=0)
        self.continuous[d > BOND_LENGTH0 * 2.0] = 0.0
        self.continuous = np.concatenate([[0], self.continuous])

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
                residue_name = AMINO_ACID_ALT_s.get(residue.name, residue.name)
                if residue_name not in AMINO_ACID_s:
                    residue_name = "UNK"
                    continue
                top_residue = top.add_residue(residue_name, top_chain, residue.resSeq)
                #
                for i_atm, atom_name in enumerate(residue_s[residue_name].atom_s):
                    if self.atom_mask_pdb[i_res, i_atm] == 0.0:
                        continue
                    element = mdtraj.core.element.Element.getBySymbol(atom_name[0])
                    top.add_atom(atom_name, element, top_residue)
        return top

    # get backbone orientations by placing rigid body atoms (N, CA, C)
    def get_backbone_orientation(self, i_res):
        mask = np.all(self.atom_mask_pdb[i_res, :3])
        if not mask:
            return mask, [np.eye(3), np.zeros(3)]

        # find BB orientations
        opr_s = [[], []]
        for k in range(self.n_frame):
            r = self.R[k, i_res, :3, :]
            frame = rigid_from_3points(r)
            opr_s[0].append(frame[0].T)
            opr_s[1].append(frame[1])
        opr_s = [np.array(opr_s[0]), np.array(opr_s[1])]
        return mask, opr_s

    # get torsion angles for a residue
    def get_torsion_angles(self, i_res):
        residue_name = self.residue_name[i_res]
        ref_res = residue_s[residue_name]
        #
        torsion_mask = np.zeros(MAX_TORSION, dtype=np.float16)
        torsion_angle_s = np.zeros(
            (self.n_frame, MAX_TORSION), dtype=np.float16
        )  # this is a list of torsion angles to rotate rigid bodies
        for tor in torsion_s[self.residue_name[i_res]]:
            if tor is None or tor.name in ["BB"]:
                continue
            #
            t_ang0, atom_s, rigid = get_rigid_group_by_torsion(residue_name, tor.name, tor.index)
            index = [ref_res.atom_s.index(atom) for atom in tor.atom_s[:4]]
            mask = self.atom_mask_pdb[i_res, index]
            if not np.all(mask):  # if any of the atoms are missing, skip this torsion
                continue
            #
            t_ang = torsion_angle(self.R[:, i_res, index, :]) - t_ang0
            torsion_mask[tor.i - 1] = 1.0
            torsion_angle_s[:, tor.i - 1] = t_ang
        return torsion_mask, torsion_angle_s

    def get_structure_information(self):
        # get rigid body operations, backbone_orientations and torsion angles
        self.bb_mask = np.zeros(self.n_residue, dtype=float)
        self.bb = np.zeros((self.n_frame, self.n_residue, 4, 3), dtype=float)
        self.torsion_mask = np.zeros((self.n_residue, MAX_TORSION), dtype=float)
        self.torsion = np.zeros((self.n_frame, self.n_residue, MAX_TORSION), dtype=float)
        for i_res in range(self.n_residue):
            mask, opr_s = self.get_backbone_orientation(i_res)
            self.bb_mask[i_res] = mask
            self.bb[:, i_res, :3, :] = opr_s[0]  # rotation matrix
            self.bb[:, i_res, 3, :] = opr_s[1]  # translation vector
            #
            mask, tor_s = self.get_torsion_angles(i_res)
            self.torsion_mask[i_res, :] = mask
            self.torsion[:, i_res, :] = tor_s

    def write_ssbond(self, pdb_fn):
        SSBOND = "SSBOND  %2d CYS %s %4d    CYS %s %4d\n"
        wrt = []
        for disu_no, ssbond in enumerate(self.ssbond_s):
            cys_0 = self.top.residue(ssbond[0])
            cys_1 = self.top.residue(ssbond[1])
            wrt.append(
                SSBOND
                % (
                    disu_no + 1,
                    CHAIN_IDs[cys_0.chain.index],
                    cys_0.resSeq,
                    CHAIN_IDs[cys_1.chain.index],
                    cys_1.resSeq,
                )
            )
        with open(pdb_fn) as fp:
            wrt.extend(fp.readlines())
        with open(pdb_fn, "wt") as fout:
            fout.writelines(wrt)

    def write(self, R, pdb_fn, dcd_fn=None):
        top = self.create_new_topology()
        mask = np.where(self.atom_mask_pdb)
        xyz = R[:, mask[0], mask[1], :]
        #
        traj = mdtraj.Trajectory(xyz[:1], top)
        traj.save(pdb_fn)
        if len(self.ssbond_s) > 0:
            self.write_ssbond(pdb_fn)
        #
        if dcd_fn is not None:
            traj = mdtraj.Trajectory(xyz, top)
            traj.save(dcd_fn)


def generate_structure_from_bb_and_torsion(residue_index, bb, torsion):
    # convert from rigid body operations to coordinates

    n_frame = bb.shape[0]
    n_residue = bb.shape[1]
    R = np.zeros((n_frame, n_residue, MAX_ATOM, 3), dtype=np.float16)
    #
    def rotate_matrix(R, X):
        return np.einsum("...ij,...jk->...ik", R, X)

    def rotate_vector(R, X):
        return np.einsum("...ij,...j", R, X)

    def combine_operations(X, Y):
        Y[..., :3, :] = rotate_matrix(X[..., :3, :], Y[..., :3, :])
        Y[..., 3, :] = rotate_vector(X[..., :3, :], Y[..., 3, :]) + X[..., 3, :]
        return Y

    #
    transforms = rigid_transforms_tensor[residue_index]
    transforms_dep = rigid_transforms_dep[residue_index]
    #
    rigids = rigid_groups_tensor[residue_index]
    rigids_dep = rigid_groups_dep[residue_index]
    #
    for frame in range(n_frame):
        opr = np.zeros_like(transforms)
        opr[:, 0] = bb
        opr[:, 1:, 0, 0] = 1.0
        sine = np.sin(torsion[frame])
        cosine = np.cos(torsion[frame])
        opr[:, 1:, 1, 1] = cosine
        opr[:, 1:, 1, 2] = -sine
        opr[:, 1:, 2, 1] = sine
        opr[:, 1:, 2, 2] = cosine
        #
        opr = combine_operations(transforms, opr)
        #
        for i_tor in range(1, MAX_RIGID):
            prev = np.take_along_axis(opr, transforms_dep[:, i_tor][:, None, None, None], axis=1)
            opr[:, i_tor] = combine_operations(prev[:, 0], opr[:, i_tor])

        # np.take_along_axis -> torch.gather
        opr = np.take_along_axis(opr, rigids_dep[..., None, None], axis=1)
        R[frame] = rotate_vector(opr[:, :, :3], rigids) + opr[:, :, 3]
    return R


if __name__ == "__main__":
    pdb = PDB("../db/pisces/pdb1/1ab1_A.pdb")
    # pdb = PDB("pdb.processed/3PBL.pdb")
    # pdb.get_structure_information()
    # R = generate_structure_from_bb_and_torsion(pdb.residue_index, pdb.bb, pdb.torsion)
    # pdb.write(R, "test.pdb")

# %%

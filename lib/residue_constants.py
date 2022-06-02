# %%
# load modules
import os
import json
import numpy as np
from collections import namedtuple

from libconfig import DATA_HOME

# %%
AMINO_ACID_s = (
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HSD",
    "HSE",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
    "UNK",
)
BACKBONE_ATOM_s = ("N", "CA", "C", "O")
ATOM_INDEX_N = BACKBONE_ATOM_s.index("N")
ATOM_INDEX_CA = BACKBONE_ATOM_s.index("CA")
ATOM_INDEX_C = BACKBONE_ATOM_s.index("C")
ATOM_INDEX_O = BACKBONE_ATOM_s.index("O")

BOND_LENGTH0 = 0.1345
BOND_ANGLE0 = (np.deg2rad(120.0), np.deg2rad(116.5))
TORSION_ANGLE0 = (np.deg2rad(0.0), np.deg2rad(180.0))

AMINO_ACID_ALT_s = {"HIS": "HSD"}
ATOM_NAME_ALT_s = {}
with open(DATA_HOME / "rename_atoms.dat") as fp:
    for line in fp:
        x = line.strip().split()
        if x[0] == "*":
            for residue_name in AMINO_ACID_s:
                ATOM_NAME_ALT_s[(residue_name, x[1])] = x[2]
        else:
            ATOM_NAME_ALT_s[(x[0], x[1])] = x[2]

MAX_RESIDUE_TYPE = len(AMINO_ACID_s)
MAX_ATOM = 24
MAX_TORSION_CHI = 4
MAX_TORSION_XI = 2
MAX_TORSION = MAX_TORSION_CHI + MAX_TORSION_XI + 2  # 2 for bb/phi/psi
MAX_RIGID = MAX_TORSION + 1
# %%
class Torsion(object):
    def __init__(self, i, name, index, sub_index, index_prev, atom_s, periodic=1):
        self.i = i
        self.name = name
        self.index = index
        self.index_prev = index_prev
        self.sub_index = sub_index
        self.atom_s = [atom.replace("*", "") for atom in atom_s]
        self.periodic = periodic
        if self.periodic > 1:
            self.atom_alt_s = self.generate_atom_alt_s(atom_s, periodic)
        else:
            self.atom_alt_s = [atom_s]

    def __repr__(self):
        return f"{self.name} {self.index} {'-'.join(self.atom_s)}"

    def generate_atom_alt_s(self, atom_s, periodic):
        alt_s = [[] for _ in range(periodic)]
        for atom in atom_s[:3]:
            for i in range(periodic):
                alt_s[i].append(atom)
        i = 0
        for atom in atom_s[3:]:
            if "*" in atom:
                atom_name = atom.replace("*", "")
                for k in range(periodic):
                    alt_s[k].append(atom_name)
            else:
                k = i % periodic
                alt_s[k].append(atom)
                i += 1
        return alt_s


# read TORSION.dat file
def read_torsion(fn):
    backbone_rigid = ["N", "CA", "C"]
    tor_s = {}
    with open(fn) as fp:
        for line in fp:
            if line.startswith("RESI"):
                residue_name = line.strip().split()[1]
                xi_index = -1
                tor_s[residue_name] = []
                if residue_name not in ["GLY"]:
                    atom_s = backbone_rigid + ["N", "CB", "HA"]
                    tor_s[residue_name].append(Torsion(0, "BB", 0, -1, -1, atom_s, 1))
                else:
                    atom_s = backbone_rigid + ["N", "HA1", "HA2"]
                    tor_s[residue_name].append(Torsion(0, "BB", 0, -1, -1, atom_s, 1))
                if residue_name not in ["PRO"]:
                    atom_s = backbone_rigid[::-1] + ["HN"]
                    tor_s[residue_name].append(Torsion(1, "PHI", 1, -1, 0, atom_s, 1))
                else:
                    tor_s[residue_name].append(None)
                atom_s = backbone_rigid + ["O"]
                tor_s[residue_name].append(Torsion(2, "PSI", 1, -1, 0, atom_s, 1))
            elif line.startswith("CHI"):
                x = line.strip().split()
                tor_no = int(x[1])
                periodic = int(x[2])
                atom_s = x[3:]
                tor_s[residue_name].append(
                    Torsion(tor_no + 2, "CHI", tor_no, -1, tor_no - 1, atom_s, periodic)
                )
            elif line.startswith("XI"):
                xi_index += 1
                x = line.strip().split()
                tor_no, sub_index = x[1].split(".")
                periodic = int(x[2])
                tor_no = int(tor_no)
                sub_index = int(sub_index)
                atom_s = x[3:]
                tor_s[residue_name].append(
                    Torsion(
                        xi_index + 7,
                        "XI",
                        tor_no,
                        sub_index,
                        tor_no - 1,
                        atom_s,
                        periodic,
                    )
                )
            elif line.startswith("#"):
                continue
    return tor_s


torsion_s = read_torsion(DATA_HOME / "torsion.dat")
# %%
class Residue(object):
    def __init__(self, residue_name: str) -> None:
        self.residue_name = residue_name
        self.residue_index = AMINO_ACID_s.index(residue_name)

        self.atom_s = [atom for atom in BACKBONE_ATOM_s]
        self.ic_s = [{}, {}, {}]  # bond, angle, torsion
        self.build_ic = []

        self.torsion_bb_atom = []
        self.torsion_chi_atom = []
        self.torsion_xi_atom = []
        self.torsion_chi_mask = np.zeros(MAX_TORSION_CHI, dtype=np.float16)
        self.torsion_xi_mask = np.zeros(MAX_TORSION_XI, dtype=np.float16)
        self.torsion_chi_periodic = np.zeros((MAX_TORSION_CHI, 3), dtype=np.float16)
        self.torsion_xi_periodic = np.zeros((MAX_TORSION_XI, 3), dtype=np.float16)

    def __str__(self):
        return self.residue_name

    def __eq__(self, other):
        if isinstance(other, Residue):
            return self.residue_name == other.residue_name
        else:
            return self.residue_name == other

    def append_atom(self, atom_name):
        if atom_name not in BACKBONE_ATOM_s:
            self.atom_s.append(atom_name)

    def append_ic(self, atom_s, param_s):
        is_improper = atom_s[2][0] == "*"
        param_s[1:4] = np.deg2rad(param_s[1:4])
        if is_improper:
            atom_s[2] = atom_s[2][1:]
            self.ic_s[0][(atom_s[0], atom_s[2])] = param_s[0]
            self.ic_s[0][(atom_s[2], atom_s[0])] = param_s[0]
            self.ic_s[1][(atom_s[0], atom_s[2], atom_s[1])] = param_s[1]
            self.ic_s[1][(atom_s[1], atom_s[2], atom_s[0])] = param_s[1]
        else:
            self.ic_s[0][(atom_s[0], atom_s[1])] = param_s[0]
            self.ic_s[0][(atom_s[1], atom_s[0])] = param_s[0]
            self.ic_s[1][(atom_s[0], atom_s[1], atom_s[2])] = param_s[1]
            self.ic_s[1][(atom_s[2], atom_s[1], atom_s[0])] = param_s[1]
        self.ic_s[2][(atom_s[0], atom_s[1], atom_s[2], atom_s[3])] = param_s[2]
        self.ic_s[1][(atom_s[1], atom_s[2], atom_s[3])] = param_s[3]
        self.ic_s[1][(atom_s[3], atom_s[2], atom_s[1])] = param_s[3]
        self.ic_s[0][(atom_s[2], atom_s[3])] = param_s[4]
        self.ic_s[0][(atom_s[3], atom_s[2])] = param_s[4]
        self.build_ic.append(tuple(atom_s))

    def get_bond_parameter(self, atom_name_s) -> float:
        b0 = self.ic_s[0].get(atom_name_s, None)
        if b0 is None:
            raise ValueError("bond parameter not found", atom_name_s)
        return b0

    def get_angle_parameter(self, atom_name_s) -> float:
        a0 = self.ic_s[1].get(atom_name_s, None)
        if a0 is None:
            raise ValueError("angle parameter not found", atom_name_s)
        return a0

    def get_torsion_parameter(self, atom_name_s) -> float:
        t0 = self.ic_s[2].get(atom_name_s, None)
        if t0 is None:
            raise ValueError("torsion parameter not found", atom_name_s)
        return t0

    def add_torsion_info(self, tor_s):
        for tor in tor_s:
            if tor is None:
                continue
            if tor.name == "CHI":
                index = tor.index - 1
                periodic = tor.periodic - 1
                self.torsion_chi_atom.append(tor.atom_s[:4])
                self.torsion_chi_mask[index] = 1.0
                self.torsion_chi_periodic[index, periodic] = 1.0
            elif tor.name == "XI":
                index = tor.i - 7
                periodic = tor.periodic - 1
                self.torsion_xi_atom.append(tor.atom_s[:4])
                self.torsion_xi_mask[index] = 1.0
                self.torsion_xi_periodic[index, periodic] = 1.0
            else:
                self.torsion_bb_atom.append(tor.atom_s[:4])

    def add_rigid_group_info(self, rigid_group, transform):
        self.rigid_group = []
        for info in rigid_group:
            self.rigid_group.append(info[:-1] + [np.array(info[-1])])
        self.transform = []
        for info in transform:
            self.transform.append(
                info[:-1] + [(np.array(info[-1][0]), np.array(info[-1][1]))]
            )


# %%
# read CHARMM topology file
def read_CHARMM_rtf(fn):
    residue_s = {}
    with open(fn) as fp:
        read = False
        for line in fp:
            if line.startswith("RESI"):
                residue_name = line.split()[1]
                if residue_name not in AMINO_ACID_s:
                    continue
                residue = Residue(residue_name)
                residue_s[residue_name] = residue
                read = True
            elif not read:
                continue
            elif line.strip() == "":
                read = False
            elif line.startswith("ATOM"):
                x = line.strip().split()
                atom_name = x[1]
                residue.append_atom(atom_name)
            elif line.startswith("IC"):
                x = line.strip().split()
                atom_s = x[1:5]
                param_s = np.array(x[5:10], dtype=float)
                residue.append_ic(atom_s, param_s)
    return residue_s


residue_s = read_CHARMM_rtf(DATA_HOME / "toppar/top_all36_prot.rtf")
for residue_name, residue in residue_s.items():
    torsion = torsion_s[residue_name]
    residue.add_torsion_info(torsion)

# %%
if os.path.exists(DATA_HOME / "rigid_groups.json") and os.path.exists(
    DATA_HOME / "rigid_body_transformation_between_frames.json"
):
    with open(DATA_HOME / "rigid_groups.json") as fp:
        rigid_groups = json.load(fp)
    with open(DATA_HOME / "rigid_body_transformation_between_frames.json") as fp:
        rigid_group_transformations = json.load(fp)
for residue_name, residue in residue_s.items():
    if (residue_name not in rigid_groups) or (
        residue_name not in rigid_group_transformations
    ):
        continue
    residue.add_rigid_group_info(
        rigid_groups[residue_name], rigid_group_transformations[residue_name]
    )
# %%
def get_rigid_group_by_torsion(residue_name, tor_name, index=-1, sub_index=-1):
    rigid_group = [[], []]  # atom_name, coord
    for X in rigid_groups[residue_name]:
        if X[1] == tor_name:
            if (index < 0 or X[2] == index) and (sub_index < 0 or X[3] == sub_index):
                t_ang = X[5]
                rigid_group[0].append(X[0])
                rigid_group[1].append(X[6])
    rigid_group[1] = np.array(rigid_group[1]) / 10.0  # in nm
    if len(rigid_group[0]) == 0:
        raise ValueError(
            "Cannot find rigid group for"
            f" {residue_name} {tor_name} {index} {sub_index}\n"
        )
    return t_ang, rigid_group[0], rigid_group[1]


# %%
def get_rigid_transform_by_torsion(residue_name, tor_name, index, sub_index=-1):
    rigid_transform = None
    for X, Y, tR in rigid_group_transformations[residue_name]:
        if (X[0] == tor_name and X[1] == index) and (
            sub_index < 0 or X[2] == sub_index
        ):
            rigid_transform = (np.array(tR[1]), np.array(tR[0]) / 10.0)
            break
    return Y, rigid_transform


# %%
rigid_transforms_tensor = np.zeros(
    (MAX_RESIDUE_TYPE, MAX_RIGID, 4, 3), dtype=np.float32
)
rigid_transforms_tensor[:, :, :3, :3] = np.eye(3)
rigid_transforms_dep = np.full((MAX_RESIDUE_TYPE, MAX_RIGID), -1, dtype=np.int16)
for i, residue_name in enumerate(AMINO_ACID_s):
    if residue_name == "UNK":
        continue
    #
    for tor in torsion_s[residue_name]:
        if tor is None or tor.name == "BB":
            continue
        if tor.name != "XI":
            prev, (R, t) = get_rigid_transform_by_torsion(
                residue_name, tor.name, tor.index
            )
        else:
            prev, (R, t) = get_rigid_transform_by_torsion(
                residue_name, tor.name, tor.index, tor.sub_index
            )
        rigid_transforms_tensor[i, tor.i, :3] = R
        rigid_transforms_tensor[i, tor.i, 3] = t
        if prev[0] == "CHI":
            dep = prev[1] + 2
        elif prev[0] == "BB":
            dep = 0
        rigid_transforms_dep[i, tor.i] = dep

# %%
rigid_groups_tensor = np.zeros((MAX_RESIDUE_TYPE, MAX_ATOM, 3), dtype=np.float32)
rigid_groups_dep = np.full((MAX_RESIDUE_TYPE, MAX_ATOM), -1, dtype=np.int16)
for i, residue_name in enumerate(AMINO_ACID_s):
    if residue_name == "UNK":
        continue
    #
    residue_atom_s = residue_s[residue_name].atom_s
    for tor in torsion_s[residue_name]:
        if tor is None:
            continue
        if tor.name != "XI":
            _, atom_names, coords = get_rigid_group_by_torsion(
                residue_name, tor.name, tor.index
            )
        else:
            _, atom_names, coords = get_rigid_group_by_torsion(
                residue_name, tor.name, tor.index, tor.sub_index
            )
        index = tuple([residue_atom_s.index(x) for x in atom_names])
        rigid_groups_tensor[i, index] = coords
        rigid_groups_dep[i, index] = tor.i
# %%

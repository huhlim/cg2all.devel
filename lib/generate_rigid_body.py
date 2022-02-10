
#%%
# load modules
import sys
import numpy as np
from typing import List
from collections import namedtuple
from libquat import Quaternion
from numpy_basics import *
from residue_constants import *
import json

from libconfig import DATA_HOME, SMALL_NUMBER

# %%
def build_structure_from_ic(residue):
    def rotate(v, axis=None, angle=0.):
        # make sure v is normalized
        v = v_norm(v)
        if axis is None:
            axis_tmp = np.cross(v, np.array([0., 0., 1.]))
            axis = np.cross(v_norm(axis_tmp), v)
        axis = v_norm(axis)
        q = Quaternion.from_axis_and_angle(axis, angle)
        return q.rotate().dot(v)
    
    R = {}
    R['-C'] = np.zeros(3)
    #
    # build N, index=0
    atom_name = 'N'
    b0 = residue.get_bond_parameter(("-C", atom_name))
    R[atom_name] = R['-C'] + b0 * np.array([1, 0, 0], dtype=np.float32)
    #
    # build CA, index=1
    atom_name = 'CA'
    b0 = residue.get_bond_parameter(("N", atom_name))
    a0 = residue.get_angle_parameter(("-C", "N", atom_name))
    v = R['N'] - R['-C']
    v = rotate(v, angle=np.pi-a0)
    R[atom_name] = R['N'] + b0 * v
    #
    # build the rest
    for atom_s in residue.build_ic:
        atom_name = atom_s[-1]
        b0 = residue.get_bond_parameter(atom_s[-2:])
        a0 = residue.get_angle_parameter(atom_s[-3:])
        t0 = residue.get_torsion_parameter(atom_s)
        r = [R.get(atom_name, None) for atom_name in atom_s[:-1]]
        if True in [np.any(np.isnan(ri)) for ri in r]:
            raise ValueError("Cannot get coordinates for atom", atom_s, r)
        r = np.array(r, dtype=np.float32)
        #
        v21 = v_norm(r[2] - r[1])
        v01 = v_norm(r[0] - r[1])
        axis = v_norm(np.cross(v21, v01))
        v = rotate(v21, axis=axis, angle=np.pi-a0)
        v = rotate(v, axis=v21, angle=t0)
        R[atom_name] = r[-1] + b0 * v
    return R

for residue in residue_s.values():
    residue.R = build_structure_from_ic(residue)

# %%
# define rigid bodies
#  - second atom at the origin
#  - align the rotation axis to the x-axis
#  - last atom on the xy-plane
def get_rigid_groups(residue_s, tor_s):
    X_axis = np.array([1., 0., 0.])
    Y_axis = np.array([0., 1., 0.])
    Z_axis = np.array([0., 0., 1.])
    #
    rigid_groups = {}
    to_json = {}
    for residue_name, residue in residue_s.items():
        rigid_groups[residue_name] = []
        #
        data = {}
        for tor in tor_s[residue_name]:
            if tor is None:
                continue
            tor_type = tor.name
            atom_s = tor.atom_s

            R = np.array([residue.R.get(atom_name) for atom_name in atom_s], dtype=np.float32)
            t_ang = torsion_angle(R[:4])
            
            # move the second atom to the origin
            R -= R[2]
            
            # align the rotation axis to the x-axis
            v = v_norm(R[2] - R[1])
            angle = np.arccos(v.dot(X_axis))
            axis = v_norm(np.cross(v, X_axis))
            q = Quaternion.from_axis_and_angle(axis, angle)
            R = q.rotate().dot(R.T).T

            # last atom on the xy-plane
            v = v_norm(R[3] - R[2])
            n = v_norm(np.cross(v, X_axis))
            angle = np.arccos(n.dot(Z_axis)) * angle_sign(n[1])
            axis = X_axis
            q = Quaternion.from_axis_and_angle(axis, angle)
            R = q.rotate().dot(R.T).T
            if R[3][1] < 0.:
                q = Quaternion.from_axis_and_angle(axis, np.pi)
                R = q.rotate().dot(R.T).T
            if tor_type == 'BB':
                R -= R[1]
            for k,atom_name in enumerate(atom_s):
                if atom_name not in data:
                    data[atom_name] = [tor, t_ang, R[k]]
            #
            # save rigid frames to evaluate rigid body transformation between frames
            rigid_groups[residue_name].append((tor, t_ang, R))

        to_json[residue_name] = []
        for atom_name in residue.atom_s:
            tor, t_ang, R = data[atom_name]
            to_json[residue_name].append(\
                [atom_name, tor.name, tor.index, tor.sub_index, tor.index_prev, t_ang, tuple(R.tolist())])
    with open(DATA_HOME / "rigid_groups.json", 'wt') as fout:
        fout.write(json.dumps(to_json, indent=2))
    return rigid_groups
                
rigid_group_s = get_rigid_groups(residue_s, torsion_s)
# %%
# define rigid body transformations between frames
# this function evaluates T_{i->j}^{lit} in the Algorithm 24 in the AF2 paper.
#  - OMEGA/PHI/PSI -> BB
#  - CHI1 -> BB
#  - CHI[2-4] -> CHI[1-3]
#  - XI[i] -> CHI[i-1]
def get_rigid_body_transformation_between_frames(rigid_group_s):
    def get_prev_frame(tor_name, tor_index, rigid_group):
        for tor in rigid_group:
            if tor[0].name == tor_name and tor[0].index == tor_index:
                return tor
        raise Exception(f"{tor_name} {tor_index} not found")
    def get_common_atoms(tor, tor_prev):
        indices = []
        for i,atom_name in enumerate(tor.atom_s[:3]):
            indices.append(tor_prev.atom_s.index(atom_name))
        return tuple(indices)

    to_json = {}
    for residue_name, rigid_group in rigid_group_s.items():
        to_json[residue_name] = []
        for tor, _, R in rigid_group:
            if tor.index_prev < 0:    # backbone do not have a previous frame
                continue
            elif tor.index_prev == 0: # backbone
                tor_prev, _, R_prev = get_prev_frame('BB', tor.index_prev, rigid_group)
            else:
                tor_prev, _, R_prev = get_prev_frame('CHI', tor.index_prev, rigid_group)
            #
            index = get_common_atoms(tor, tor_prev)
            P = R_prev[index,:].copy()
            Q = R[:3].copy()
            #
            # align the second atoms at the origin
            P0 = P[1].copy()
            Q0 = Q[1].copy()
            P -= P0
            Q -= Q0

            # align the torsion axis
            v21 = v_norm(Q[2] - Q[1])
            torsion_axis = v_norm(P[2] - P[1])
            angle = np.arccos(np.clip(v21.dot(torsion_axis), -1., 1.))
            axis = np.cross(v21, torsion_axis)
            if v_size(axis) > 0.:
                axis = v_norm(axis)
                q = Quaternion.from_axis_and_angle(axis, angle)
                rotation_1 = q.rotate()
            else:
                rotation_1 = np.eye(3)
            Q = rotation_1.dot(Q.T).T

            # align the dihedral angle
            v01 = v_norm(Q[0] - Q[1])
            u01 = v_norm(P[0] - P[1])
            n0 = v_norm(np.cross(v01, torsion_axis))
            n1 = v_norm(np.cross(u01, torsion_axis))
            angle = np.arccos(n0.dot(n1)) * angle_sign(v01.dot(n1))
            if angle != 0.:
                q = Quaternion.from_axis_and_angle(torsion_axis, angle)
                rotation_2 = q.rotate()
            else:
                rotation_2 = np.eye(3)
            Q = rotation_2.dot(Q.T).T
            #
            rotation = rotation_2.dot(rotation_1)
            translation = P0 - rotation.dot(Q0)
            #
            R = translate_and_rotate(R, rotation, translation)
            delta = R[:3] - R_prev[index,:]
            delta = np.sqrt(np.mean(np.power(delta, 2).sum(-1)))
            if delta > 1e-5:
                raise ValueError (residue_name, tor, tor_prev, R.round(3), R_prev[index,:].round(3), delta)
            #
            to_json[residue_name].append(\
                [(tor.name, tor.index, tor.sub_index), (tor_prev.name, tor_prev.index), \
                    (translation.tolist(), rotation.tolist())])
    with open(DATA_HOME / "rigid_body_transformation_between_frames.json", 'wt') as fout:
        fout.write(json.dumps(to_json, indent=2))

get_rigid_body_transformation_between_frames(rigid_group_s)
# %%

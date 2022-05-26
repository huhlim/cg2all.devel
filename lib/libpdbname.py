#!/usr/bin/env python

import numpy as np
from numpy_basics import *
from residue_constants import *
from collections import namedtuple
import itertools

from libconfig import DATA_HOME

def combine_opr_s(opr_s):
    R, t = opr_s[0]
    for opr in opr_s[1:]:
        R = opr[0].dot(R)
        t = opr[0].dot(t) + opr[1]
    return R, t

# read ambiguous atom list
def read_ambiguous_atom_list():
    Ambiguous = namedtuple("AmbiguousAtomName", \
        ("residue_name", "atom_s", "method", "torsion_name", "torsion_index", "torsion_atom_s"))
    
    amb_s = []
    with open(DATA_HOME / "ambiguous_names.dat", 'r') as fp:
        for line in fp:
            if line.startswith("#"):
                continue
            x = line.strip().split()
            residue_name = x[0]
            atom_s = x[3:]
            if '-' not in x[1]:
                torsion_name, torsion_index = x[1].split("_")
            else:
                torsion_name = 'special'
                torsion_definition, torsion_index = x[1].split("_")
                torsion_definition = torsion_definition.split("-")
            torsion_index = int(torsion_index)
            for tor in torsion_s[residue_name]:
                if tor is None: continue
                if tor.name == torsion_name and tor.index == torsion_index:
                    if atom_s[0] == 'ALL': 
                        atom_s = tor.atom_s[3:]
                    torsion_definition = tor.atom_s[:3]
                    break
            method = x[2]
            amb_s.append(Ambiguous(residue_name, atom_s, method, \
                torsion_name, torsion_index, torsion_definition))

    for residue_name in AMINO_ACID_s:
        if residue_name == 'UNK': continue
        for tor in torsion_s[residue_name]:
            if tor is None: continue
            if tor.name == 'XI' and len(tor.atom_s) > 1:
                amb_s.append(Ambiguous(residue_name, tor.atom_s[3:], "periodic", \
                    "XI", tor.index, tor.atom_s[:3]))
    return amb_s
ambiguous_atom_list = read_ambiguous_atom_list()

def get_ambiguous_atom_list(residue_name, torsion_name, torsion_index=-1):
    for amb in ambiguous_atom_list:
        if amb.residue_name != residue_name:
            continue
        elif amb.torsion_name != torsion_name:
            continue
        elif torsion_index >= 0 and amb.torsion_index != torsion_index:
            continue
        return amb
    return None

# %%
def apply_swapping_rule(ref_res, amb, R):
    greek = amb.atom_s[0][1]
    group_number_s = [atom_name[2] for atom_name in amb.atom_s]
    group_s = []
    for i,group_number in enumerate(group_number_s):
        prefix = f"H{greek}{group_number}"
        group = [amb.atom_s[i]] + [atom for atom in ref_res.atom_s if atom.startswith(prefix)]
        group_s.append([ref_res.atom_s.index(atom) for atom in group])
    before = group_s[0] + group_s[1]
    after = group_s[1] + group_s[0]
    R[before,:] = R[after,:]

def apply_closest_rule(ref_res, amb, periodic_s, atom_s, rigid, R):
    index_s = []
    index_tgt = []
    for periodic in periodic_s:
        index_s.append([ref_res.atom_s.index(atom) for atom in periodic])  # index for R
        index_tgt += [atom_s.index(atom) for atom in periodic] # rigid is properly oriented
    #
    R_tgt = rigid[index_tgt,:]
    #
    d_min = np.inf
    index_orig = None
    n_periodic = len(periodic_s)
    for permutation in itertools.permutations(range(n_periodic)):
        index = []
        for i in permutation:
            index += index_s[i]
        R_p = R[index,:]
        d = v_size(R_tgt - R_p).sum()
        if d < d_min:
            d_min = d
            index_min = index
        if index_orig is None:
            index_orig = index
    R[index_orig,:] = R[index_min,:]

# %%
def update_by_closest_method(R, atom_mask, i_res, ref_res, tor, amb, opr_dict):
    # get rigid-body transformation
    prev, rigid_tR = get_rigid_transform_by_torsion(ref_res.residue_name, tor.name, tor.index)
    t_ang0, atom_s, rigid = get_rigid_group_by_torsion(ref_res.residue_name, tor.name, tor.index)
    #
    # calculate the torsion angle
    torsion_angle_atom_s = tor.atom_s[:4]
    index = [ref_res.atom_s.index(atom) for atom in torsion_angle_atom_s]
    mask = np.all(atom_mask[i_res, index])
    if not mask:
        return None, None, None
    #
    r = R[:, i_res, index, :]
    t_ang = torsion_angle(r)
    t_delta = t_ang - t_ang0
    #
    opr_s = [[], []]
    rigid_s = []
    opr_prev = opr_dict[tuple(prev)]
    if opr_prev is None:
        return None, None, None
    #
    for k in range(r.shape[0]):
        opr = rotate_x(t_delta[k])
        opr = combine_opr_s([opr, rigid_tR, (opr_prev[0][k], opr_prev[1][k])])
        opr_s[0].append(opr[0])
        opr_s[1].append(opr[1])
    rigid_s = translate_and_rotate(rigid, np.array(opr_s[0]), np.array(opr_s[1]))

    if amb is not None:
        periodic_s = [[atom] for atom in amb.atom_s]
        for k in range(R.shape[0]):
            apply_closest_rule(ref_res, amb, periodic_s, atom_s, rigid_s[k], R[k, i_res, :, :])
    opr_s = [np.array(opr_s[0]), np.array(opr_s[1])]
    return opr_s, atom_s, np.array(rigid_s)

def update_by_permute_method(R, atom_mask, i_res, ref_res, tor, amb, opr_dict):
    # get rigid-body transformation
    prev, rigid_tR = get_rigid_transform_by_torsion(ref_res.residue_name, tor.name, tor.index)
    t_ang0, atom_s, rigid = get_rigid_group_by_torsion(ref_res.residue_name, tor.name, tor.index)
    #
    index_amb0 = [ref_res.atom_s.index(atom) for atom in amb.atom_s]
    index_amb1 = [atom_s.index(atom) for atom in amb.atom_s]
    #
    opr_s = [[], []]
    rigid_s = []
    #
    for k in range(R.shape[0]):
        R_amb = R[k, i_res, index_amb0, :]
        d_min = None
        #
        for swap, atom in enumerate(amb.atom_s):
            torsion_angle_atom_s = amb.torsion_atom_s + [atom]
            index = [ref_res.atom_s.index(atom) for atom in torsion_angle_atom_s]
            mask = np.all(atom_mask[i_res, index])
            if not mask:
                return None, None, None
            #
            r = R[k, i_res, index, :]
            t_ang = torsion_angle(r)
            t_delta = t_ang - t_ang0
            #
            opr = rotate_x(t_delta)
            opr_prev = opr_dict[tuple(prev)]
            if opr_prev is None:
                return None, None, None
            opr = combine_opr_s([opr, rigid_tR, (opr_prev[0][k], opr_prev[1][k])])

            rigid_try = translate_and_rotate(rigid, *opr)
            R_try = rigid_try[index_amb1,:]
            d_try = v_size(R_try - R_amb).sum()
            if d_min is None or d_try < d_min:
                d_min = d_try
                opr_min = opr
                rigid_min = rigid_try
                swap_min = swap
        #
        opr_s[0].append(opr[0])
        opr_s[1].append(opr[1])
        rigid_s.append(rigid_min)
        #
        if swap_min == 1: 
            apply_swapping_rule(ref_res, amb, R[k, i_res, :, :])

    opr_s = [np.array(opr_s[0]), np.array(opr_s[1])]
    return opr_s, atom_s, np.array(rigid_s)

def update_by_periodic_method(R, atom_mask, i_res, ref_res, tor, amb, opr_dict):
    # get rigid-body transformation
    if tor.name == 'XI':
        prev, rigid_tR = get_rigid_transform_by_torsion(ref_res.residue_name, tor.name, tor.index, tor.sub_index)
        t_ang0, atom_s, rigid = get_rigid_group_by_torsion(ref_res.residue_name, tor.name, tor.index, tor.sub_index)
    else:
        prev, rigid_tR = get_rigid_transform_by_torsion(ref_res.residue_name, tor.name, tor.index)
        t_ang0, atom_s, rigid = get_rigid_group_by_torsion(ref_res.residue_name, tor.name, tor.index)
    #
    # find 
    t_ang_min = np.full(R.shape[0], np.inf, dtype=np.float32)
    for torsion_angle_atom_s in tor.atom_alt_s:
        index = [ref_res.atom_s.index(atom) for atom in torsion_angle_atom_s]
        r = R[:, i_res, index, :]
        mask = np.all(atom_mask[i_res, index])
        if not mask:
            return None, None, None
        t_ang = torsion_angle(r)
        selected = (np.abs(t_ang) < np.abs(t_ang_min))
        t_ang_min[selected] = t_ang[selected]
    t_delta = t_ang_min - t_ang0
    #
    # rigid body operation
    opr_s = [[], []]
    rigid_s = []
    periodic_s = [alt[3:] for alt in tor.atom_alt_s]
    for k in range(R.shape[0]):
        opr = rotate_x(t_delta[k])
        opr_prev = opr_dict[tuple(prev)]
        if opr_prev is None:
            return None, None, None
        opr = combine_opr_s([opr, rigid_tR, (opr_prev[0][k], opr_prev[1][k])])
        opr_s[0].append(opr[0])
        opr_s[1].append(opr[1])
    opr_s = [np.array(opr_s[0]), np.array(opr_s[1])]
    rigid_s = translate_and_rotate(rigid, opr_s[0], opr_s[1])
    for k in range(R.shape[0]):
        apply_closest_rule(ref_res, amb, periodic_s, atom_s, rigid_s[k], R[k, i_res, :, :])
        #
    return opr_s, atom_s, np.array(rigid_s)

def update_by_glycine_backbone_method(R, i_res, ref_res, amb, atom_s, rigid_s):
    periodic_s = [[atom] for atom in amb.atom_s]
    for k in range(R.shape[0]):
        apply_closest_rule(ref_res, amb, periodic_s, atom_s, rigid_s[k], R[k, i_res, :, :])

def update_by_special_method(R, atom_mask, i_res, ref_res):
    amb = get_ambiguous_atom_list(ref_res.residue_name, 'special')
    index_amb = [ref_res.atom_s.index(atom) for atom in amb.atom_s]
    index_tor = [ref_res.atom_s.index(atom) for atom in amb.torsion_atom_s]
    if not np.all(atom_mask[i_res,index_amb]):
        return
    if not np.all(atom_mask[i_res,index_tor]):
        return

    # calculate the torsion angle
    index = index_tor + [index_amb[0]]
    t_ang0 = torsion_angle(R[:,i_res,index,:])
    index = index_tor + [index_amb[1]]
    t_ang1 = torsion_angle(R[:,i_res,index,:])
    #
    # let's swap
    swap = (np.abs(t_ang0) > np.abs(t_ang1))
    if np.any(swap):
        R[swap,i_res,(index_amb[1], index_amb[0]),:] = R[swap,i_res,index_amb,:]

def update_by_guanidium_method(R, atom_mask, i_res, ref_res):
    for i in range(3):
        amb = get_ambiguous_atom_list(ref_res.residue_name, 'special', i)
        #
        index_amb = [ref_res.atom_s.index(atom) for atom in amb.atom_s]
        index_tor = [ref_res.atom_s.index(atom) for atom in amb.torsion_atom_s]
        if not np.all(atom_mask[i_res,index_amb]):
            return
        if not np.all(atom_mask[i_res,index_tor]):
            return

        # calculate the torsion angle
        index = index_tor + [index_amb[0]]
        t_ang0 = torsion_angle(R[:,i_res,index,:])
        index = index_tor + [index_amb[1]]
        t_ang1 = torsion_angle(R[:,i_res,index,:])
        #
        if i == 0:
            dep_s = []
            for index, atom in zip(index_amb, amb.atom_s):
                group_number = atom[2]
                prefix= f"HH{group_number}"
                dep_s.append([index] + [i for i,atom_name in enumerate(ref_res.atom_s) \
                        if atom_name.startswith(prefix)])
        else:
            dep_s = [[x] for x in index_amb]
        #
        # let's swap
        swap = (np.abs(t_ang0) > np.abs(t_ang1))
        if np.any(swap):
            before = tuple(dep_s[0] + dep_s[1])
            after = tuple(dep_s[1] + dep_s[0])
            R[swap,i_res,before,:] = R[swap,i_res,after,:]


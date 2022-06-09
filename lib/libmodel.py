#!/usr/bin/env python

import sys
import copy
import functools

import torch
import torch.nn as nn
from ml_collections import ConfigDict

import e3nn
import e3nn.nn
from e3nn import o3

import liblayer
from libconfig import DTYPE, EPS
from residue_constants import (
    MAX_ATOM,
    MAX_TORSION,
    MAX_RIGID,
    rigid_transforms_tensor,
    rigid_transforms_dep,
    rigid_groups_tensor,
    rigid_groups_dep,
)
from libloss import (
    v_norm,
    v_size,
    loss_f_rigid_body,
    loss_f_mse_R,
    loss_f_distance_matrix,
    loss_f_torsion_angle,
    loss_f_bonded_energy,
)
from libmetric import rmsd_CA, rmsd_rigid, rmsd_all, bonded_energy

RIGID_TRANSFORMS_TENSOR = torch.tensor(rigid_transforms_tensor)
RIGID_TRANSFORMS_DEP = torch.tensor(rigid_transforms_dep, dtype=torch.long)
RIGID_TRANSFORMS_DEP[RIGID_TRANSFORMS_DEP == -1] = MAX_RIGID - 1
RIGID_GROUPS_TENSOR = torch.tensor(rigid_groups_tensor)
RIGID_GROUPS_DEP = torch.tensor(rigid_groups_dep, dtype=torch.long)
RIGID_GROUPS_DEP[RIGID_GROUPS_DEP == -1] = MAX_RIGID - 1


CONFIG = ConfigDict()

CONFIG_BASE = ConfigDict()
CONFIG_BASE = {}
CONFIG_BASE["num_recycle"] = 2
CONFIG_BASE["num_layers"] = 3
CONFIG_BASE["layer_type"] = "ConvLayer"
CONFIG_BASE["in_Irreps"] = "23x0e + 2x1o"
CONFIG_BASE["out_Irreps"] = "20x0e + 10x1o"
CONFIG_BASE["mid_Irreps"] = "40x0e + 20x1o"
CONFIG_BASE["attn_Irreps"] = "40x0e + 20x1o"
CONFIG_BASE["l_max"] = 2
CONFIG_BASE["mlp_num_neurons"] = [20, 20]
CONFIG_BASE["activation"] = ["relu", None]
CONFIG_BASE["norm"] = True
CONFIG_BASE["radius"] = 1.0
CONFIG_BASE["skip_connection"] = False
CONFIG_BASE["loss_weight"] = ConfigDict()

CONFIG["feature_extraction"] = copy.deepcopy(CONFIG_BASE)
CONFIG["backbone"] = copy.deepcopy(CONFIG_BASE)
CONFIG["sidechain"] = copy.deepcopy(CONFIG_BASE)

CONFIG["backbone"].update(
    {
        "num_layers": 2,
        "in_Irreps": "20x0e + 10x1o",
        # "out_Irreps": "3x0e + 1x1o",  # scalars for quaternions and a vector for translation
        "out_Irreps": "1x0e + 2x1o",  # scalars for quaternions and a vector for translation
        "mid_Irreps": "20x0e + 10x1o",
        "attn_Irreps": "20x0e + 10x1o",
        "loss_weight": {
            "rigid_body": 0.0,
            "bonded_energy": 0.0,
            "distance_matrix": 0.0,
        },
    }
)

CONFIG["sidechain"].update(
    {
        "num_layers": 2,
        "in_Irreps": "20x0e + 10x1o",
        "out_Irreps": f"{MAX_TORSION*2:d}x0e",
        "mid_Irreps": "20x0e + 10x1o",
        "attn_Irreps": "20x0e + 10x1o",
        "loss_weight": {"torsion_angle": 0.0},
    }
)

CONFIG["loss_weight"] = ConfigDict()
CONFIG["loss_weight"].update(
    {
        "rigid_body": 0.0,
        "mse_R": 0.0,
        "bonded_energy": 0.0,
        "distance_matrix": 0.0,
        "torsion_angle": 0.0,
    }
)


class BaseModule(nn.Module):
    def __init__(self, config, compute_loss=False):
        super().__init__()
        #
        self.compute_loss = compute_loss
        self.loss_weight = config.loss_weight
        #
        self.in_Irreps = o3.Irreps(config.in_Irreps)
        self.mid_Irreps = o3.Irreps(config.mid_Irreps)
        self.out_Irreps = o3.Irreps(config.out_Irreps)
        if config.activation[0] is None:
            activation = None
        elif config.activation[0] == "relu":
            activation = torch.relu
        elif config.activation[0] == "elu":
            activation = torch.elu
        else:
            raise NotImplementedError
        if config.activation[1] is None:
            activation_final = None
        elif config.activation[1] == "relu":
            activation_final = torch.relu
        elif config.activation[1] == "elu":
            activation_final = torch.elu
        else:
            raise NotImplementedError
        self.skip_connection = config.skip_connection
        self.num_recycle = max(1, config.num_recycle)
        #
        if config.layer_type == "ConvLayer":
            layer_partial = functools.partial(
                liblayer.ConvLayer,
                radius=config.radius,
                l_max=config.l_max,
                mlp_num_neurons=config.mlp_num_neurons,
                activation=activation,
                norm=config.norm,
            )

        elif config.layer_type == "SE3Transformer":
            layer_partial = functools.partial(
                liblayer.SE3Transformer,
                attn_Irreps=config.attn_Irreps,
                radius=config.radius,
                l_max=config.l_max,
                mlp_num_neurons=config.mlp_num_neurons,
                activation=activation,
                norm=config.norm,
            )
        #
        self.layer_0 = layer_partial(self.in_Irreps, self.mid_Irreps)
        self.layer_1 = layer_partial(self.mid_Irreps, self.out_Irreps, norm=False)
        if activation_final is None:
            self.activation_final = None
        else:
            self.activation_final = e3nn.nn.NormActivation(
                self.out_Irreps, activation_final
            )

        layer = layer_partial(self.mid_Irreps, self.mid_Irreps)
        self.layer_s = nn.ModuleList([layer for _ in range(config.num_layers)])

    def forward(self, batch, feat):
        loss_out = {}
        #
        feat, graph = self.layer_0(batch, feat)
        radius_prev = self.layer_0.radius
        #
        for k in range(self.num_recycle):
            for i, layer in enumerate(self.layer_s):
                if self.skip_connection:
                    feat0 = feat.clone()
                if layer.radius == radius_prev:
                    feat, graph = layer(batch, feat, graph)
                else:
                    feat, graph = layer(batch, feat)
                if self.skip_connection:
                    feat = feat + feat0
                radius_prev = layer.radius

            if self.layer_1.radius == radius_prev:
                feat_out, graph = self.layer_1(batch, feat, graph)
            else:
                feat_out, graph = self.layer_1(batch, feat)
            if self.activation_final is not None:
                feat_out = self.activation_final(feat_out)
            radius_prev = self.layer_1.radius
            #
            if (self.compute_loss or self.training) and (k + 1 != self.num_recycle):
                for loss_name, loss_value in self.loss_f(feat_out, batch).items():
                    if loss_name in loss_out:
                        loss_out[loss_name] += loss_value
                    else:
                        loss_out[loss_name] = loss_value

        return feat_out, loss_out

    def loss_f(self, f_out, batch):
        return {}


class FeatureExtractionModule(BaseModule):
    def __init__(self, config, compute_loss=False):
        super().__init__(config, compute_loss=compute_loss)


class BackboneModule(BaseModule):
    def __init__(self, config, compute_loss=False):
        super().__init__(config, compute_loss=compute_loss)

    def loss_f(self, f_out, batch):
        R = build_structure(batch, f_out, sc=None)
        #
        loss = {}
        if self.loss_weight.get("rigid_body", 0.0) > 0.0:
            loss["rigid_body"] = (
                loss_f_rigid_body(R, batch.output_xyz) * self.loss_weight.rigid_body
            )
        if self.loss_weight.get("bonded_energy", 0.0) > 0.0:
            loss["bonded_energy"] = (
                loss_f_bonded_energy(R, batch.continuous)
                * self.loss_weight.bonded_energy
            )
        if self.loss_weight.get("distance_matrix", 0.0) > 0.0:
            loss["distance_matrix"] = (
                loss_f_distance_matrix(R, batch) * self.loss_weight.distance_matrix
            )
        return loss


class SidechainModule(BaseModule):
    def __init__(self, config, compute_loss=False):
        super().__init__(config, compute_loss=compute_loss)

    def loss_f(self, f_out, batch):
        loss = {}
        if self.loss_weight.get("torsion_angle", 0.0) > 0.0:
            loss["torsion_angle"] = (
                loss_f_torsion_angle(f_out, batch.correct_torsion, batch.torsion_mask)
                * self.loss_weight.torsion_angle
            )
        return loss


class Model(nn.Module):
    def __init__(self, _config, compute_loss=False):
        super().__init__()
        #
        self.compute_loss = compute_loss
        self.loss_weight = _config.loss_weight
        #
        self.feature_extraction_module = FeatureExtractionModule(
            _config.feature_extraction, compute_loss=False
        )
        self.backbone_module = BackboneModule(
            _config.backbone, compute_loss=compute_loss
        )
        self.sidechain_module = SidechainModule(
            _config.sidechain, compute_loss=compute_loss
        )

    def forward(self, batch):
        f_out, _ = self.feature_extraction_module(batch, batch.f_in)
        #
        ret = {}
        loss = {}
        ret["bb"], loss["bb"] = self.backbone_module(batch, f_out)
        ret["sc"], loss["sc"] = self.sidechain_module(batch, f_out)
        ret["R"] = build_structure(batch, ret["bb"], ret["sc"])
        if self.compute_loss or self.training:
            loss["R"] = self.loss_f(ret, batch)
        metrics = self.calc_metrics(ret, batch)
        return ret, loss, metrics

    def loss_f(self, ret, batch):
        R = ret["R"]
        #
        loss = {}
        if self.loss_weight.get("rigid_body", 0.0) > 0.0:
            loss["rigid_body"] = (
                loss_f_rigid_body(R, batch.output_xyz) * self.loss_weight.rigid_body
            )
        if self.loss_weight.get("mse_R", 0.0) > 0.0:
            loss["mse_R"] = (
                loss_f_mse_R(R, batch.output_xyz, batch.output_atom_mask)
                * self.loss_weight.mse_R
            )
        if self.loss_weight.get("bonded_energy", 0.0) > 0.0:
            loss["bonded_energy"] = (
                loss_f_bonded_energy(R, batch.continuous)
                * self.loss_weight.bonded_energy
            )
        if self.loss_weight.get("distance_matrix", 0.0) > 0.0:
            loss["distance_matrix"] = (
                loss_f_distance_matrix(R, batch) * self.loss_weight.distance_matrix
            )
        if self.loss_weight.get("torsion_angle", 0.0) > 0.0:
            loss["torsion_angle"] = (
                loss_f_torsion_angle(
                    ret["sc"], batch.correct_torsion, batch.torsion_mask
                )
                * self.loss_weight.torsion_angle
            )
        return loss

    def calc_metrics(self, ret, batch):
        R = ret["R"]
        R_ref = batch.output_xyz
        #
        metric_s = {}
        metric_s["rmsd_CA"] = rmsd_CA(R, R_ref)
        metric_s["rmsd_rigid"] = rmsd_rigid(R, R_ref)
        metric_s["rmsd_all"] = rmsd_all(R, R_ref, batch.output_atom_mask)
        metric_s["bond_energy"] = bonded_energy(R, batch.continuous)
        return metric_s


def build_structure(batch, bb, sc=None):
    def rotate_matrix(R, X):
        return torch.einsum("...ij,...jk->...ik", R, X)

    def rotate_vector(R, X):
        return torch.einsum("...ij,...j", R, X)

    def combine_operations(X, Y):
        y = Y.clone()
        Y[..., :3, :] = rotate_matrix(X[..., :3, :], y[..., :3, :])
        Y[..., 3, :] = rotate_vector(X[..., :3, :], y[..., 3, :]) + X[..., 3, :]
        return Y

    device = bb.device
    residue_type = batch.residue_type
    #
    transforms = RIGID_TRANSFORMS_TENSOR[residue_type].to(device)
    transforms_dep = RIGID_TRANSFORMS_DEP[residue_type].to(device)
    #
    rigids = RIGID_GROUPS_TENSOR[residue_type].to(device)
    rigids_dep = RIGID_GROUPS_DEP[residue_type].to(device)
    #
    opr = torch.zeros_like(transforms, device=device)
    #
    n_residue = batch.residue_type.size(0)
    #
    # backbone operations
    angle = bb[:, :1] / 2.0
    v = v_norm(bb[:, 1:4])
    q = torch.cat([torch.cos(angle), v * torch.sin(angle)], dim=1)
    # v = 2.0 * (torch.sigmoid(bb[:, :3].clone()) - 0.5)
    # _q = torch.cat([torch.ones((n_residue, 1), dtype=DTYPE, device=device), v], dim=1)
    # q = _q / torch.linalg.norm(_q, dim=1)[:, None]
    #
    R = torch.zeros((n_residue, 3, 3), dtype=DTYPE, device=device)
    R[:, 0, 0] = q[:, 0] ** 2 + q[:, 1] ** 2 - q[:, 2] ** 2 - q[:, 3] ** 2
    R[:, 1, 1] = q[:, 0] ** 2 - q[:, 1] ** 2 + q[:, 2] ** 2 - q[:, 3] ** 2
    R[:, 2, 2] = q[:, 0] ** 2 - q[:, 1] ** 2 - q[:, 2] ** 2 + q[:, 3] ** 2
    R[:, 0, 1] = 2 * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3])
    R[:, 0, 2] = 2 * (q[:, 1] * q[:, 3] + q[:, 0] * q[:, 2])
    R[:, 1, 0] = 2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
    R[:, 1, 2] = 2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1])
    R[:, 2, 0] = 2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
    R[:, 2, 1] = 2 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1])
    opr[:, 0, :3] = R
    opr[:, 0, 3, :] = 0.1 * bb[:, 4:] + batch.pos

    # sidechain operations
    if sc is not None:
        opr[:, 1:, 0, 0] = 1.0
        torsion = sc.reshape(n_residue, -1, 2).clone()
        torsion[:, :, 1] = torsion[:, :, 1] + EPS
        norm = torch.linalg.norm(torsion, dim=2)
        sine = torsion[:, :, 0] / norm
        cosine = torsion[:, :, 1] / norm
        opr[:, 1:, 1, 1] = cosine
        opr[:, 1:, 1, 2] = -sine
        opr[:, 1:, 2, 1] = sine
        opr[:, 1:, 2, 2] = cosine
    #
    opr = combine_operations(transforms, opr)
    #
    if sc is not None:
        for i_tor in range(1, MAX_RIGID):
            prev = torch.take_along_dim(
                opr.clone(), transforms_dep[:, i_tor][:, None, None, None], 1
            )
            opr[:, i_tor] = combine_operations(prev[:, 0], opr[:, i_tor])

    opr = torch.take_along_dim(opr, rigids_dep[..., None, None], axis=1)
    R = rotate_vector(opr[:, :, :3], rigids) + opr[:, :, 3]
    return R

#!/usr/bin/env python

import sys
import copy
import functools

import torch
import torch.nn as nn
import torch.utils.checkpoint
from ml_collections import ConfigDict

import e3nn
import e3nn.nn
from e3nn import o3

import liblayer
import libnorm
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
    v_norm_safe,
    loss_f_rigid_body,
    loss_f_FAPE_CA,
    loss_f_quaternion,
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

CONFIG["globals"] = ConfigDict()
CONFIG["globals"]["num_recycle"] = 2
CONFIG["globals"]["loss_weight"] = ConfigDict()
CONFIG["globals"]["loss_weight"].update(
    {
        "rigid_body": 0.0,
        "mse_R": 0.0,
        "bonded_energy": 0.0,
        "distance_matrix": 0.0,
        "torsion_angle": 0.0,
        "quaternion": 0.0,
    }
)

# the base config for using ConvLayer or SE3Transformer
CONFIG_BASE = ConfigDict()
CONFIG_BASE = {}
CONFIG_BASE["layer_type"] = "ConvLayer"
CONFIG_BASE["num_layers"] = 4
CONFIG_BASE["in_Irreps"] = "38x0e + 4x1o"
CONFIG_BASE["out_Irreps"] = "80x0e + 20x1o"
CONFIG_BASE["mid_Irreps"] = "80x0e + 20x1o"
CONFIG_BASE["attn_Irreps"] = "40x0e + 20x1o"
CONFIG_BASE["l_max"] = 2
CONFIG_BASE["mlp_num_neurons"] = [20, 20]
CONFIG_BASE["activation"] = "relu"
CONFIG_BASE["radius"] = 1.0
CONFIG_BASE["norm"] = True
CONFIG_BASE["skip_connection"] = True
CONFIG_BASE["loss_weight"] = ConfigDict()

CONFIG["feature_extraction"] = copy.deepcopy(CONFIG_BASE)
CONFIG["transition"] = copy.deepcopy(CONFIG_BASE)
CONFIG["backbone"] = copy.deepcopy(CONFIG_BASE)
CONFIG["sidechain"] = copy.deepcopy(CONFIG_BASE)

CONFIG["transition"].update(
    {
        "layer_type": "Linear",
        "num_layers": 2,
        "in_Irreps": "80x0e + 20x1o",
        "out_Irreps": "80x0e + 20x1o",
        "mid_Irreps": "80x0e + 20x1o",
        "skip_connection": True,
        "norm": True,
    }
)
CONFIG["backbone"].update(
    {
        "layer_type": "Linear",
        "num_layers": 1,
        "in_Irreps": "80x0e + 20x1o",
        "out_Irreps": "4x0e + 1x1o",  # quaternion + translation
        "mid_Irreps": "20x0e + 4x1o",
        "skip_connection": True,
        "norm": False,
        "loss_weight": {
            "rigid_body": 0.0,
            "quaternion": 0.0,
            "bonded_energy": 0.0,
            "distance_matrix": 0.0,
        },
    }
)

CONFIG["sidechain"].update(
    {
        "layer_type": "Linear",
        "num_layers": 2,
        "in_Irreps": "80x0e + 20x1o",
        "out_Irreps": f"{MAX_TORSION*2:d}x0e",
        "mid_Irreps": "20x0e + 4x1o",
        "skip_connection": True,
        "norm": False,
        "loss_weight": {"torsion_angle": 0.0},
    }
)


def _get_gpu_mem():
    return torch.cuda.memory_allocated()/1024/1024, torch.cuda.memory_allocated()/1024/1024


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
        if config.activation is None:
            self.activation = None
        elif config.activation == "relu":
            self.activation = torch.relu
        elif config.activation == "elu":
            self.activation = torch.elu
        elif config.activation == "sigmoid":
            self.activation = torch.sigmoid
        else:
            raise NotImplementedError
        self.skip_connection = config.skip_connection
        if config.norm:
            self.norm = libnorm.LayerNorm(self.out_Irreps)
        else:
            self.norm = None
        #
        if config.layer_type == "ConvLayer":
            self.use_graph = True
            layer_partial = functools.partial(
                liblayer.ConvLayer,
                radius=config.radius,
                l_max=config.l_max,
                mlp_num_neurons=config.mlp_num_neurons,
                activation=self.activation,
            )

        elif config.layer_type == "SE3Transformer":
            self.use_graph = True
            layer_partial = functools.partial(
                liblayer.SE3Transformer,
                attn_Irreps=config.attn_Irreps,
                radius=config.radius,
                l_max=config.l_max,
                mlp_num_neurons=config.mlp_num_neurons,
                activation=self.activation,
            )

        elif config.layer_type == "Linear":
            self.use_graph = False
            layer_partial = functools.partial(o3.Linear, biases=True)
        #
        self.layer_0 = layer_partial(self.in_Irreps, self.mid_Irreps)
        layer = layer_partial(self.mid_Irreps, self.mid_Irreps)
        self.layer_s = nn.ModuleList([layer for _ in range(config.num_layers)])
        self.layer_1 = layer_partial(self.mid_Irreps, self.out_Irreps)

    def forward(self, batch, feat):
        if self.use_graph:
            out = self.forward_graph(batch, feat)
        else:
            out = self.forward_linear(feat)
        #
        if self.norm is not None:
            out = self.norm(out)
        return out

    def forward_graph(self, batch, feat):
        feat, graph = self.layer_0(batch, feat)
        radius_prev = self.layer_0.radius
        #
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
        #
        feat_out, graph = self.layer_1(batch, feat)
        return feat_out

    def forward_linear(self, feat):
        feat = self.layer_0(feat)

        for i, layer in enumerate(self.layer_s):
            if self.skip_connection:
                feat0 = feat.clone()
                #
            feat = layer(feat)
            #
            if self.activation is not None:
                feat = self.activation(feat)
            if self.skip_connection:
                feat = feat + feat0
        #
        feat_out = self.layer_1(feat)
        return feat_out

    def loss_f(self, batch, f_out):
        raise NotImplementedError


class FeatureExtractionModule(BaseModule):
    def __init__(self, config, compute_loss=False):
        super().__init__(config, compute_loss=compute_loss)


class BackboneModule(BaseModule):
    def __init__(self, config, compute_loss=False):
        super().__init__(config, compute_loss=compute_loss)

    def loss_f(self, batch, bb, bb1, loss_prev):
        R, opr_bb = build_structure(batch, bb, sc=None)
        #
        loss = {}
        if self.loss_weight.get("rigid_body", 0.0) > 0.0:
            loss["rigid_body"] = (
                loss_f_rigid_body(R, batch.output_xyz) * self.loss_weight.rigid_body
            )
        if self.loss_weight.get("FAPE_CA", 0.0) > 0.0:
            loss["FAPE_CA"] = (
                loss_f_FAPE_CA(batch, R, opr_bb) * self.loss_weight.FAPE_CA
            )

        if self.loss_weight.get("quaternion", 0.0) > 0.0:
            loss["quaternion"] = (
                loss_f_quaternion(bb, bb1, batch.correct_quat)
                * self.loss_weight.quaternion
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
        #
        for k, v in loss_prev.items():
            if k in loss:
                loss[k] += v
            else:
                loss[k] = v
        return loss

    @staticmethod
    def compose(bb0, bb1):
        # translation
        bb1[:, 4:] = bb1[:, 4:] + bb0[:, 4:]
        #
        # quaternion
        # q = q1 * q0
        q0 = bb0[:, :4]  # assume q0 is normalized
        q1 = bb1[:, :4].clone()
        q1[:, 0] = q1[:, 0] + EPS
        q1 = v_norm(q1)
        q = q1.clone()
        q[:, 0] = q0[:, 0] * q1[:, 0] - torch.sum(q0[:, 1:] * q1[:, 1:], dim=-1)
        q[:, 1:] = (
            q0[:, :1] * q1[:, 1:]
            + q1[:, :1] * q0[:, 1:]
            + torch.cross(q1[:, 1:], q0[:, 1:])
        )
        bb1[:, :4] = q
        return bb1

    def init_value(self, batch):
        out = torch.zeros((batch.pos.size(0), self.out_Irreps.dim), dtype=DTYPE)
        out[:, 0] = 1.0
        return out


class SidechainModule(BaseModule):
    def __init__(self, config, compute_loss=False):
        super().__init__(config, compute_loss=compute_loss)

    def forward(self, batch, feat):
        out = super().forward(batch, feat)
        out = out.reshape(-1, MAX_TORSION, 2)
        return out

    def loss_f(self, batch, sc, sc1, loss_prev):
        loss = {}
        if self.loss_weight.get("torsion_angle", 0.0) > 0.0:
            loss["torsion_angle"] = (
                loss_f_torsion_angle(sc, sc1, batch.correct_torsion, batch.torsion_mask)
                * self.loss_weight.torsion_angle
            )
        for k, v in loss_prev.items():
            if k in loss:
                loss[k] += v
            else:
                loss[k] = v
        return loss

    @staticmethod
    def compose(sc0, sc):
        sc = v_norm_safe(sc + sc0)
        return sc

    def init_value(self, batch):
        out = torch.zeros((batch.pos.size(0), MAX_TORSION, 2), dtype=DTYPE)
        out[:, :, 0] = 1.0
        return out


class Model(nn.Module):
    def __init__(self, _config, compute_loss=False, checkpoint=False):
        super().__init__()
        #
        self.compute_loss = compute_loss
        self.checkpoint = checkpoint
        self.num_recycle = _config.globals.num_recycle
        self.loss_weight = _config.globals.loss_weight
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
        n_residue = batch.pos.size(0)
        device = batch.pos.device
        #
        loss = {}
        #
        ret = {}
        ret["bb"] = self.backbone_module.init_value(batch).to(device)
        ret["sc"] = self.sidechain_module.init_value(batch).to(device)
        #
        for _ in range(self.num_recycle):
            if self.training and self.checkpoint:
                f_out = torch.utils.checkpoint.checkpoint(self.feature_extraction_module, batch, batch.f_in)
            else:
                f_out = self.feature_extraction_module(batch, batch.f_in)
            #
            bb = self.backbone_module(batch, f_out)
            ret["bb"] = self.backbone_module.compose(ret["bb"], bb)
            #
            sc = self.sidechain_module(batch, f_out)
            ret["sc"] = self.sidechain_module.compose(ret["sc"], sc)
            #
            self.update_graph(batch, ret["bb"])
            #
            if self.compute_loss or self.training:
                loss["bb"] = self.backbone_module.loss_f(
                    batch, ret["bb"], bb, loss.get("bb", {})
                )
                loss["sc"] = self.sidechain_module.loss_f(
                    batch, ret["sc"], sc, loss.get("sc", {})
                )

        ret["R"], ret["opr_bb"] = build_structure(batch, ret["bb"], sc=ret["sc"])
        if self.compute_loss or self.training:
            loss["R"] = self.loss_f(batch, ret)
            for k, v in loss["bb"].items():
                loss["bb"][k] = v / self.num_recycle
            for k, v in loss["sc"].items():
                loss["sc"][k] = v / self.num_recycle

        metrics = self.calc_metrics(batch, ret)

        return ret, loss, metrics

    def update_graph(self, batch, bb):
        batch.pos = batch.pos0 + 0.1 * bb[:, 4:]

    def loss_f(self, batch, ret):
        R = ret["R"]
        #
        loss = {}
        if self.loss_weight.get("rigid_body", 0.0) > 0.0:
            loss["rigid_body"] = (
                loss_f_rigid_body(R, batch.output_xyz) * self.loss_weight.rigid_body
            )
        if self.loss_weight.get("FAPE_CA", 0.0) > 0.0:
            loss["FAPE_CA"] = (
                loss_f_FAPE_CA(batch, R, ret["opr_bb"]) * self.loss_weight.FAPE_CA
            )
        if self.loss_weight.get("quaternion", 0.0) > 0.0:
            loss["quaternion"] = (
                loss_f_quaternion(ret["bb"], None, batch.correct_quat)
                * self.loss_weight.quaternion
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
                    ret["sc"], None, batch.correct_torsion, batch.torsion_mask
                )
                * self.loss_weight.torsion_angle
            )
        return loss

    def calc_metrics(self, batch, ret):
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
    n_residue = batch.residue_type.size(0)
    #
    transforms = RIGID_TRANSFORMS_TENSOR[residue_type].to(device)
    transforms_dep = RIGID_TRANSFORMS_DEP[residue_type].to(device)
    #
    rigids = RIGID_GROUPS_TENSOR[residue_type].to(device)
    rigids_dep = RIGID_GROUPS_DEP[residue_type].to(device)
    #
    opr = torch.zeros_like(transforms, device=device)
    #
    # backbone operations
    # angle = bb[:, :2]
    # angle[:, 0] = angle[:, 0] + EPS
    # norm = torch.linalg.norm(angle, dim=-1)
    # cosine = angle[:, 0] / norm
    # sine = angle[:, 1] / norm
    # v = v_norm(bb[:, 2:5])
    # q = torch.cat([cosine[:, None], v * sine[:, None]], dim=1)
    q = bb[:, :4]
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
    opr[:, 0, 3, :] = 0.1 * bb[:, 4:] + batch.pos0

    # sidechain operations
    if sc is not None:
        # assume that sc is v_norm_safed
        opr[:, 1:, 0, 0] = 1.0
        opr[:, 1:, 1, 1] = sc[:, :, 0]
        opr[:, 1:, 1, 2] = -sc[:, :, 1]
        opr[:, 1:, 2, 1] = sc[:, :, 1]
        opr[:, 1:, 2, 2] = sc[:, :, 0]
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
    return R, opr[:, 0]

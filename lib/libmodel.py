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
from libconfig import DTYPE
from residue_constants import (
    MAX_ATOM,
    MAX_TORSION,
    MAX_RIGID,
    rigid_transforms_tensor,
    rigid_transforms_dep,
    rigid_groups_tensor,
    rigid_groups_dep,
)

RIGID_TRANSFORMS_TENSOR = torch.tensor(rigid_transforms_tensor)
RIGID_TRANSFORMS_DEP = torch.tensor(rigid_transforms_dep, dtype=torch.long)
RIGID_TRANSFORMS_DEP[RIGID_TRANSFORMS_DEP == -1] = MAX_RIGID - 1
RIGID_GROUPS_TENSOR = torch.tensor(rigid_groups_tensor)
RIGID_GROUPS_DEP = torch.tensor(rigid_groups_dep, dtype=torch.long)
RIGID_GROUPS_DEP[RIGID_GROUPS_DEP == -1] = MAX_RIGID - 1


CONFIG = ConfigDict()

CONFIG_BASE = ConfigDict()
CONFIG_BASE = {}
CONFIG_BASE["num_layers"] = 3
CONFIG_BASE["layer_type"] = "ConvLayer"
CONFIG_BASE["in_Irreps"] = "23x0e + 2x1o"
CONFIG_BASE["out_Irreps"] = "20x0e + 10x1o"
CONFIG_BASE["mid_Irreps"] = "40x0e + 20x1o"
CONFIG_BASE["attn_Irreps"] = "40x0e + 20x1o"
CONFIG_BASE["l_max"] = 2
CONFIG_BASE["mlp_num_neurons"] = [20, 20]
CONFIG_BASE["activation"] = "relu"
CONFIG_BASE["norm"] = True
CONFIG_BASE["radius"] = 1.0
CONFIG_BASE["skip_connection"] = True

CONFIG["feature_extraction"] = copy.deepcopy(CONFIG_BASE)
CONFIG["backbone"] = copy.deepcopy(CONFIG_BASE)
CONFIG["sidechain"] = copy.deepcopy(CONFIG_BASE)

CONFIG["backbone"].update(
    {
        "num_layers": 2,
        "in_Irreps": "20x0e + 10x1o",
        "out_Irreps": "3x0e + 1x1o",  # scalars for quaternions and a vector for translation
        "mid_Irreps": "10x0e + 4x1o",
        "attn_Irreps": "10x0e + 4x1o",
    }
)

CONFIG["sidechain"].update(
    {
        "num_layers": 2,
        "in_Irreps": "20x0e + 10x1o",
        "out_Irreps": f"{MAX_TORSION*2:d}x0e",
        "mid_Irreps": "10x0e + 4x1o",
        "attn_Irreps": "10x0e + 4x1o",
    }
)


class BaseModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        #
        self.in_Irreps = o3.Irreps(config.in_Irreps)
        self.mid_Irreps = o3.Irreps(config.mid_Irreps)
        self.out_Irreps = o3.Irreps(config.out_Irreps)
        if config.activation == "relu":
            activation = torch.relu
        elif config.activation == "elu":
            activation = torch.elu
        else:
            raise NotImplementedError
        self.skip_connection = config.skip_connection
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
        self.layer_0 = layer_partial(self.in_Irreps, self.mid_Irreps)
        self.layer_1 = layer_partial(self.mid_Irreps, self.out_Irreps)

        layer = layer_partial(self.mid_Irreps, self.mid_Irreps)
        self.layer_s = nn.ModuleList([layer for _ in range(config.num_layers - 2)])

    def forward(self, batch, feat):
        feat = self.layer_0(batch, feat)
        for i, layer in enumerate(self.layer_s):
            if self.skip_connection:
                feat = layer(batch, feat) + feat
            else:
                feat = layer(batch, feat)
        feat = self.layer_1(batch, feat)
        return feat


class Model(nn.Module):
    def __init__(self, _config):
        super().__init__()
        #
        self.feature_extraction_module = BaseModule(_config.feature_extraction)
        self.backbone_module = BaseModule(_config.backbone)
        self.sidechain_module = BaseModule(_config.sidechain)

    def forward(self, batch):
        f_in = torch.cat(
            [
                batch.f_in[0],
                batch.f_in[1].reshape(batch.f_in[1].shape[0], -1),
            ],
            dim=1,
        )
        #
        f_out = self.feature_extraction_module(batch, f_in)
        #
        ret = {}
        ret["bb"] = self.backbone_module(batch, f_out)
        ret["sc"] = self.sidechain_module(batch, f_out)
        ret["R"] = self.build_structure(batch, ret)
        return ret

    def build_structure(self, batch, ret):
        def rotate_matrix(R, X):
            return torch.einsum("...ij,...jk->...ik", R, X)

        def rotate_vector(R, X):
            return torch.einsum("...ij,...j", R, X)

        def combine_operations(X, Y):
            y = Y.clone()
            Y[..., :3, :] = rotate_matrix(X[..., :3, :], y[..., :3, :])
            Y[..., 3, :] = rotate_vector(X[..., :3, :], y[..., 3, :]) + X[..., 3, :]
            return Y

        #
        residue_type = batch.residue_type
        #
        transforms = RIGID_TRANSFORMS_TENSOR[residue_type]
        transforms_dep = RIGID_TRANSFORMS_DEP[residue_type]
        #
        rigids = RIGID_GROUPS_TENSOR[residue_type]
        rigids_dep = RIGID_GROUPS_DEP[residue_type]
        #
        opr = torch.zeros_like(transforms)
        #
        n_residue = batch.residue_type.size(0)
        #
        # backbone operations
        _q = torch.cat(
            [torch.ones((n_residue, 1), dtype=DTYPE), ret["bb"][:, :3]], dim=1
        )
        q = _q / torch.linalg.norm(_q, dim=1)[:, None]
        #
        R = torch.zeros((n_residue, 3, 3), dtype=DTYPE)
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
        opr[:, 0, 3, :] = ret["bb"][:, 3:] + batch.pos

        # sidechain operations
        opr[:, 1:, 0, 0] = 1.0
        torsion = ret["sc"].reshape(n_residue, -1, 2)
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
        for i_tor in range(1, MAX_RIGID):
            prev = torch.take_along_dim(
                opr.clone(), transforms_dep[:, i_tor][:, None, None, None], 1
            )
            opr[:, i_tor] = combine_operations(prev[:, 0], opr[:, i_tor])

        opr = torch.take_along_dim(opr, rigids_dep[..., None, None], axis=1)
        R = rotate_vector(opr[:, :, :3], rigids) + opr[:, :, 3]
        return R

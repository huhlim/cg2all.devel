#!/usr/bin/env python

import sys
import copy
import functools
import logging
import numpy as np

import torch
import torch.nn as nn
from ml_collections import ConfigDict

import dgl

try:
    from torch._C import _nvtx
    from se3_transformer.model import Fiber, SE3Transformer
    from se3_transformer.model.layers import LinearSE3, NormSE3
except ImportError:
    from se3_transformer.model_no_cuda import Fiber, SE3Transformer
    from se3_transformer.model_no_cuda.layers import LinearSE3, NormSE3

from residue_constants import (
    MAX_RESIDUE_TYPE,
    MAX_TORSION,
    MAX_RIGID,
    ATOM_INDEX_CA,
    RIGID_TRANSFORMS_TENSOR,
    RIGID_TRANSFORMS_DEP,
    RIGID_GROUPS_TENSOR,
    RIGID_GROUPS_DEP,
)
from libloss_l1 import loss_f
from torch_basics import v_size, v_norm_safe, inner_product, rotate_matrix, rotate_vector
from libmetric import rmsd_CA, rmsd_rigid, rmsd_all, rmse_bonded
from libcg import get_residue_center_of_mass, get_backbone_angles
from libconfig import DTYPE


CONFIG = ConfigDict()

CONFIG["globals"] = ConfigDict()
CONFIG["globals"]["num_recycle"] = 1
CONFIG["globals"]["radius"] = 0.8
CONFIG["globals"]["loss_weight"] = ConfigDict()
CONFIG["globals"]["loss_weight"].update(
    {
        "rigid_body": 1.0,
        "FAPE_CA": 5.0,
        "FAPE_all": 5.0,
        "mse_R": 0.0,
        "v_cntr": 1.0,
        "bonded_energy": 1.0,
        "distance_matrix": 0.0,
        "rotation_matrix": 1.0,
        "torsion_angle": 2.5,
        "atomic_clash": 1.0,
    }
)

# embedding module
EMBEDDING_MODULE = ConfigDict()
EMBEDDING_MODULE["num_embeddings"] = MAX_RESIDUE_TYPE
EMBEDDING_MODULE["embedding_dim"] = 40
CONFIG["embedding_module"] = EMBEDDING_MODULE

# the base config for using ConvLayer or SE3Transformer
STRUCTURE_MODULE = ConfigDict()
STRUCTURE_MODULE["low_memory"] = True
STRUCTURE_MODULE["num_layers"] = 4
STRUCTURE_MODULE["num_linear_layers"] = 4
STRUCTURE_MODULE["num_heads"] = 8  # number of attention heads
STRUCTURE_MODULE["norm"] = [True, True]  # norm between attention blocks / within attention blocks
STRUCTURE_MODULE["nonlinearity"] = "relu"

# fiber_in: is determined by input features
STRUCTURE_MODULE["fiber_in"] = None
# fiber_out: is determined by outputs
# - degree 0: cosine/sine values of torsion angles
# - degree 1: two for BB rigid body rotation matrix and one for CA translation
STRUCTURE_MODULE["fiber_out_g"] = [(0, 32), (1, 16)]
STRUCTURE_MODULE["fiber_out"] = [(0, MAX_TORSION * 2), (1, 3)]
# num_degrees and num_channels are for fiber_hidden
# - they will be converted to Fiber using Fiber.create(num_degrees, num_channels)
# - which is {degree: num_channels for degree in range(num_degrees)}
STRUCTURE_MODULE["num_degrees"] = 3
STRUCTURE_MODULE["num_channels"] = 16
STRUCTURE_MODULE["channels_div"] = 2  # no idea... # of channels is divided by this number
STRUCTURE_MODULE["fiber_edge"] = None
#
STRUCTURE_MODULE["loss_weight"] = ConfigDict()
STRUCTURE_MODULE["loss_weight"].update(
    {
        "rigid_body": 1.0,
        "FAPE_CA": 5.0,
        "FAPE_all": 0.0,
        "mse_R": 0.0,
        "v_cntr": 1.0,
        "bonded_energy": 1.0,
        "distance_matrix": 0.0,
        "rotation_matrix": 1.0,
        "torsion_angle": 2.0,
        "atomic_clash": 0.0,
    }
)
CONFIG["structure_module"] = STRUCTURE_MODULE


def _get_gpu_mem():
    return (
        torch.cuda.memory_allocated() / 1024 / 1024,
        torch.cuda.memory_allocated() / 1024 / 1024,
    )


def set_model_config(arg: dict, cg_model) -> ConfigDict:
    config = copy.deepcopy(CONFIG)
    config.update_from_flattened_dict(arg)
    #
    n_node_scalar = cg_model.n_node_scalar + config.embedding_module.embedding_dim
    config.structure_module.fiber_in = []
    if n_node_scalar > 0:
        config.structure_module.fiber_in.append((0, n_node_scalar))
    if cg_model.n_node_vector > 0:
        config.structure_module.fiber_in.append((1, cg_model.n_node_vector))

    config.structure_module.fiber_edge = []
    if cg_model.n_edge_scalar > 0:
        config.structure_module.fiber_edge.append((0, cg_model.n_edge_scalar))
    if cg_model.n_edge_vector > 0:
        config.structure_module.fiber_edge.append((1, cg_model.n_edge_vector))
    #
    return config


class EmbeddingModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        #
        self.layer = nn.Embedding(config.num_embeddings, config.embedding_dim)

    def forward(self, batch: dgl.DGLGraph):
        return self.layer(batch.ndata["residue_type"])


class InteractionModule(nn.Module):
    def __init__(self, config, n_node_add=[0, 0], n_edge_add=[0, 0]):
        super().__init__()
        #
        if config.nonlinearity == "elu":
            nonlinearity = nn.ELU()
        elif config.nonlinearity == "relu":
            nonlinearity = nn.ReLU()
        #
        if sum(n_node_add) > 0:
            fiber_in = []
            for degree, n_feat in config.fiber_in:
                fiber_in.append((degree, n_feat + n_node_add[degree]))
        else:
            fiber_in = config.fiber_in

        if sum(n_edge_add) > 0:
            if config.fiber_edge is None:
                fiber_edge = [(degree, n_feat) for degree, n_feat in enumerate(n_edge_add)]
            else:
                fiber_edge = []
                for degree, n_feat in config.fiber_edge:
                    fiber_edge.append((degree, n_feat + n_edge_add[degree]))
        else:
            fiber_edge = config.fiber_edge

        #
        self.graph_module = SE3Transformer(
            num_layers=config.num_layers,
            fiber_in=Fiber(fiber_in),
            fiber_hidden=Fiber.create(config.num_degrees, config.num_channels),
            fiber_out=Fiber(config.fiber_out_g),
            num_heads=config.num_heads,
            channels_div=config.channels_div,
            fiber_edge=Fiber(fiber_edge or {}),
            norm=config.norm[0],
            use_layer_norm=config.norm[1],
            nonlinearity=nonlinearity,
            low_memory=config.low_memory,
        )

    def forward(self, batch: dgl.DGLGraph, node_feats, edge_feats):
        out = self.graph_module(batch, node_feats=node_feats, edge_feats=edge_feats)
        return out


class StructureModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        #
        self.loss_weight = config.loss_weight
        #
        if config.nonlinearity == "elu":
            nonlinearity = nn.ELU()
        elif config.nonlinearity == "relu":
            nonlinearity = nn.ReLU()
        #
        linear_module = []
        for _ in range(config.num_linear_layers - 1):
            if config.norm[0]:
                linear_module.append(NormSE3(Fiber(config.fiber_out_g), nonlinearity=nonlinearity))
            linear_module.append(LinearSE3(Fiber(config.fiber_out_g), Fiber(config.fiber_out_g)))
        #
        self.linear_module = nn.Sequential(*linear_module)
        #
        backbone_module = []
        if config.norm[0]:
            backbone_module.append(NormSE3(Fiber(config.fiber_out_g), nonlinearity=nonlinearity))
        backbone_module.append(LinearSE3(Fiber(config.fiber_out_g), Fiber(config.fiber_out[1])))
        self.backbone_module = nn.Sequential(*backbone_module)
        #
        sidechain_module = []
        for _ in range(2):
            if config.norm[0]:
                sidechain_module.append(
                    NormSE3(Fiber(config.fiber_out_g), nonlinearity=nonlinearity)
                )
            sidechain_module.append(LinearSE3(Fiber(config.fiber_out_g), Fiber(config.fiber_out_g)))
        #
        if config.norm[0]:
            sidechain_module.append(NormSE3(Fiber(config.fiber_out_g), nonlinearity=nonlinearity))
        sidechain_module.append(LinearSE3(Fiber(config.fiber_out_g), Fiber(config.fiber_out[0])))

    def forward(self, feats):
        out = self.linear_module(feats)
        bb = self.backbone_module(out)
        sc = self.sidechain_module(out)
        return bb|sc

    @staticmethod
    def output_to_opr(output, num_recycle=1):
        bb0 = output["1"][:, :2]
        v0 = bb0[:, 0]
        v1 = bb0[:, 1]
        e0 = v_norm_safe(v0, index=0)
        u1 = v1 - e0 * inner_product(e0, v1)[:, None]
        e1 = v_norm_safe(u1, index=1)
        e2 = torch.cross(e0, e1)
        rot = torch.stack([e0, e1, e2], dim=1).mT
        #
        t = 0.1 * output["1"][:, 2][..., None, :] / num_recycle
        bb = torch.cat([rot, t], dim=1)
        #
        sc0 = output["0"].reshape(-1, MAX_TORSION, 2)
        sc = v_norm_safe(sc0)
        #
        return bb, sc, bb0, sc0


class Model(nn.Module):
    def __init__(self, _config, cg_model, compute_loss=False):
        super().__init__()
        #
        self.cg_model = cg_model
        self.compute_loss = compute_loss
        self.num_recycle = _config.globals.num_recycle
        self.loss_weight = _config.globals.loss_weight
        #
        self.embedding_module = EmbeddingModule(_config.embedding_module)
        self.interaction_module_0 = InteractionModule(_config.structure_module)
        # self.interaction_module_1 = InteractionModule(
        #     _config.structure_module, n_node_add=[0, 1], n_edge_add=[1, 0]
        # )
        self.structure_module_0 = StructureModule(_config.structure_module)
        # self.structure_module_1 = StructureModule(_config.structure_module)

    def set_rigid_operations(self, device, dtype=DTYPE):
        _RIGID_TRANSFORMS_TENSOR = RIGID_TRANSFORMS_TENSOR.to(device)
        _RIGID_GROUPS_TENSOR = RIGID_GROUPS_TENSOR.to(device)
        if dtype != DTYPE:
            _RIGID_TRANSFORMS_TENSOR = _RIGID_TRANSFORMS_TENSOR.type(dtype)
            _RIGID_GROUPS_TENSOR = _RIGID_GROUPS_TENSOR.type(dtype)
        _RIGID_TRANSFORMS_DEP = RIGID_TRANSFORMS_DEP.to(device)
        _RIGID_GROUPS_DEP = RIGID_GROUPS_DEP.to(device)
        #
        self.RIGID_OPs = (
            (_RIGID_TRANSFORMS_TENSOR, _RIGID_GROUPS_TENSOR),
            (_RIGID_TRANSFORMS_DEP, _RIGID_GROUPS_DEP),
        )

    def forward(self, batch: dgl.DGLGraph):
        loss = {}
        ret = {}
        #
        # residue_type --> embedding
        embedding = self.embedding_module(batch)
        #
        n_intermediate = 0
        for k in range(self.num_recycle):
            # first-pass
            edge_feats = {"0": batch.edata["edge_feat_0"]}
            node_feats = {
                "0": torch.cat([batch.ndata["node_feat_0"], embedding[..., None]], dim=1),
                "1": batch.ndata["node_feat_1"],
            }

            out = self.interaction_module_0(batch, node_feats=node_feats, edge_feats=edge_feats)
            out = self.structure_module_0(out)
            bb, sc, bb0, sc0 = self.structure_module_0.output_to_opr(out, self.num_recycle)
            ret["bb"] = bb
            ret["sc"] = sc
            ret["bb0"] = bb0
            ret["sc0"] = sc0
            #
            # build structure (1)
            ret["R"], ret["opr_bb"] = build_structure(
                self.RIGID_OPs, batch, ret["bb"], sc=ret["sc"], stop_grad=(k + 1 < self.num_recycle)
            )
            #
            # if self.compute_loss or self.training:
            #     n_intermediate += 1
            #     loss["intermediate"] = loss_f(
            #         batch,
            #         ret,
            #         self.structure_module.loss_weight,
            #         loss_prev=loss.get("intermediate", {}),
            #         RIGID_OPs=self.RIGID_OPs,  # only for atomic_clash
            #     )
            # #
            # stop_grad
            # ret["R"] = ret["R"].detach()
            #
            # calculate backbone torsion angles
            # angles = get_backbone_angles(ret["R"])
            # node_feats["0"] = torch.cat([node_feats["0"], angles[..., None]], dim=1)
            #
            # # calculate v_cntr
            # R_cntr = get_residue_center_of_mass(ret["R"], batch.ndata["atomic_mass"])
            # v_cntr = R_cntr - ret["R"][:, ATOM_INDEX_CA, :]
            # node_feats["1"] = torch.cat([node_feats["1"], v_cntr[:, None, :]], dim=1)
            # #
            # i, j = batch.edges()
            # dij = v_size(R_cntr[j] - R_cntr[i])
            # edge_feats["0"] = torch.cat([edge_feats["0"], dij[..., None, None]], dim=1)
            # #
            # # second-pass with v_cntr
            # out = self.interaction_module_1(batch, node_feats=node_feats, edge_feats=edge_feats)
            # out = self.structure_module_1(out)
            # bb, sc, bb0, sc0 = self.structure_module_1.output_to_opr(out, self.num_recycle)
            # ret["bb"] = bb
            # ret["sc"] = sc
            # ret["bb0"] = bb0
            # ret["sc0"] = sc0
            # #
            # # build structure (2)
            # ret["R"], ret["opr_bb"] = build_structure(
            #     self.RIGID_OPs, batch, ret["bb"], sc=ret["sc"], stop_grad=(k + 1 < self.num_recycle)
            # )
            # #
            # if k < self.num_recycle - 1:
            #     if self.compute_loss or self.training:
            #         n_intermediate += 1
            #         loss["intermediate"] = loss_f(
            #             batch,
            #             ret,
            #             self.structure_module_1.loss_weight,
            #             loss_prev=loss.get("intermediate", {}),
            #             RIGID_OPs=self.RIGID_OPs,  # only for atomic_clash
            #         )
            #     #
            #     # self.update_graph(batch, ret)

        if self.compute_loss or self.training:
            loss["final"] = loss_f(batch, ret, self.loss_weight, RIGID_OPs=self.RIGID_OPs)
            if "intermediate" in loss:
                for k, v in loss["intermediate"].items():
                    loss["intermediate"][k] = v / n_intermediate

        metrics = self.calc_metrics(batch, ret)
        #
        return ret, loss, metrics

    def update_graph(self, batch, ret):
        raise NotImplementedError

    def calc_metrics(self, batch, ret):
        R = ret["R"]
        R_ref = batch.ndata["output_xyz"]
        #
        metric_s = {}
        metric_s["rmsd_CA"] = rmsd_CA(R, R_ref)
        metric_s["rmsd_rigid"] = rmsd_rigid(R, R_ref)
        metric_s["rmsd_all"] = rmsd_all(R, R_ref, batch.ndata["pdb_atom_mask"])
        #
        bonded = rmse_bonded(R, batch.ndata["continuous"])
        metric_s["bond_length"] = bonded[0]
        metric_s["bond_angle"] = bonded[1]
        metric_s["omega_angle"] = bonded[2]
        return metric_s


def combine_operations(X, Y):
    y = Y.clone()
    Y[..., :3, :] = rotate_matrix(X[..., :3, :], y[..., :3, :])
    Y[..., 3, :] = rotate_vector(X[..., :3, :], y[..., 3, :]) + X[..., 3, :]
    return Y


def build_structure(RIGID_OPs, batch, bb, sc=None, stop_grad=False):
    dtype = bb.dtype
    device = bb.device
    residue_type = batch.ndata["residue_type"]
    #
    transforms = RIGID_OPs[0][0][residue_type]
    rigids = RIGID_OPs[0][1][residue_type]
    transforms_dep = RIGID_OPs[1][0][residue_type]
    rigids_dep = RIGID_OPs[1][1][residue_type]
    # if dtype != DTYPE:
    #     transforms = RIGID_TRANSFORMS_TENSOR[residue_type].to(device).type(dtype)
    #     rigids = RIGID_GROUPS_TENSOR[residue_type].to(device).type(dtype)
    # else:
    #     transforms = RIGID_TRANSFORMS_TENSOR[residue_type].to(device)
    #     rigids = RIGID_GROUPS_TENSOR[residue_type].to(device)
    # transforms_dep = RIGID_TRANSFORMS_DEP[residue_type].to(device)
    # rigids_dep = RIGID_GROUPS_DEP[residue_type].to(device)
    #
    opr = torch.zeros_like(transforms, device=device)
    #
    # backbone operations
    if stop_grad:
        opr[:, 0, :3] = bb[:, :3].detach()
    else:
        opr[:, 0, :3] = bb[:, :3]
    # opr[:, 0, 3] = bb[:, 3] + batch.ndata["pos0"]
    opr[:, 0, 3] = bb[:, 3] + batch.ndata["pos"]

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

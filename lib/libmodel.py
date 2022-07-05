#!/usr/bin/env python

import sys
import copy
import functools
import logging

import torch
import torch.nn as nn
import torch.utils.checkpoint
from ml_collections import ConfigDict

import e3nn
import e3nn.nn
from e3nn import o3

import liblayer
import libnorm
from residue_constants import (
    MAX_RESIDUE_TYPE,
    MAX_TORSION,
    MAX_RIGID,
    rigid_transforms_tensor,
    rigid_transforms_dep,
    rigid_groups_tensor,
    rigid_groups_dep,
)
from libloss import (
    v_norm_safe,
    inner_product,
    loss_f_rigid_body,
    loss_f_FAPE_CA,
    loss_f_rotation_matrix,
    loss_f_mse_R,
    loss_f_distance_matrix,
    loss_f_torsion_angle,
    loss_f_bonded_energy,
)
from libmetric import rmsd_CA, rmsd_rigid, rmsd_all, bonded_energy
from libconfig import EQUIVARIANT_TOLERANCE

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
        "rigid_body": 1.0,
        "FAPE_CA": 5.0,
        "mse_R": 1.0,
        "bonded_energy": 1.0,
        "distance_matrix": 0.0,
        "rotation_matrix": 1.0,
        "torsion_angle": 1.0,
    }
)

# the base config for using ConvLayer or SE3Transformer
CONFIG_BASE = ConfigDict()
CONFIG_BASE["layer_type"] = "SE3Transformer"
CONFIG_BASE["num_layers"] = 4
CONFIG_BASE["in_Irreps"] = "40x0e + 10x1o"
CONFIG_BASE["out_Irreps"] = "40x0e + 10x1o"
CONFIG_BASE["mid_Irreps"] = "80x0e + 20x1o"
CONFIG_BASE["attn_Irreps"] = "80x0e + 20x1o"
CONFIG_BASE["l_max"] = 2
CONFIG_BASE["mlp_num_neurons"] = [20, 20]
CONFIG_BASE["activation"] = None
CONFIG_BASE["radius"] = 0.8
CONFIG_BASE["norm"] = [False, False, False]
CONFIG_BASE["skip_connection"] = True
CONFIG_BASE["preprocess"] = [False, False]  # rotation / translation
# CONFIG_BASE["preprocess"] = [False, True]  # rotation / translation
CONFIG_BASE["loss_weight"] = ConfigDict()

CONFIG["initialization"] = copy.deepcopy(CONFIG_BASE)
CONFIG["feature_extraction"] = copy.deepcopy(CONFIG_BASE)
CONFIG["transition"] = copy.deepcopy(CONFIG_BASE)
CONFIG["backbone"] = copy.deepcopy(CONFIG_BASE)
CONFIG["sidechain"] = copy.deepcopy(CONFIG_BASE)

CONFIG["embedding"] = ConfigDict()
CONFIG["embedding"]["num_embeddings"] = MAX_RESIDUE_TYPE
CONFIG["embedding"]["embedding_dim"] = 40

CONFIG["initialization"].update(
    {
        "layer_type": "Linear",
        "num_layers": 4,
        "in_Irreps": "16x0e + 4x1o",
        "out_Irreps": "40x0e + 10x1o",
        "mid_Irreps": "40x0e + 10x1o",
        "activation": "relu",
        "skip_connection": True,
        "norm": [False, True, True],
    }
)

CONFIG["feature_extraction"].update(
    {
        "layer_type": "SE3Transformer",
        "num_layers": 4,
        "in_Irreps": "40x0e + 10x1o",
        "out_Irreps": "40x0e + 10x1o",
        "mid_Irreps": "80x0e + 20x1o",
        "attn_Irreps": "80x0e + 20x1o",
        "activation": "relu",
        "skip_connection": True,
        "norm": [False, True, True],
    }
)

CONFIG["transition"].update(
    {
        "layer_type": "Linear",
        "num_layers": 4,
        "in_Irreps": "40x0e + 10x1o",
        "out_Irreps": "40x0e + 10x1o",
        "mid_Irreps": "40x0e + 10x1o",
        "skip_connection": True,
        "norm": [False, True, True],
    }
)

CONFIG["backbone"].update(
    {
        "layer_type": "Linear",
        "num_layers": 4,
        "in_Irreps": "40x0e + 10x1o",
        "out_Irreps": "3x1o",  # two for rotation, one for translation
        "mid_Irreps": "20x0e + 4x1o",
        "skip_connection": True,
        "norm": [False, True, False],
        "loss_weight": {
            "rigid_body": 0.0,
            "FAPE_CA": 0.0,
            "bonded_energy": 0.0,
            "distance_matrix": 0.0,
            "rotation_matrix": 0.0,
        },
    }
)

CONFIG["sidechain"].update(
    {
        "layer_type": "Linear",
        "num_layers": 4,
        "in_Irreps": "40x0e + 10x1o",
        "out_Irreps": f"{MAX_TORSION*2:d}x0e",
        "mid_Irreps": "20x0e + 4x1o",
        "skip_connection": True,
        "norm": [False, True, False],
        "loss_weight": {"torsion_angle": 0.0},
    }
)


def _get_gpu_mem():
    return (
        torch.cuda.memory_allocated() / 1024 / 1024,
        torch.cuda.memory_allocated() / 1024 / 1024,
    )


def set_model_config(arg: dict) -> ConfigDict:
    config = copy.deepcopy(CONFIG)
    config.update_from_flattened_dict(arg)
    #
    initialization_in_Irreps = " + ".join(
        [f"{config.embedding.embedding_dim:d}x0e", config.initialization.in_Irreps]
    )
    feature_extraction_in_Irreps = config.initialization.out_Irreps
    transition_in_Irreps = config.feature_extraction.out_Irreps
    backbone_in_Irreps = config.transition.out_Irreps
    sidechain_in_Irreps = config.transition.out_Irreps
    if config.globals.num_recycle > 1:
        for i, irreps in enumerate(["2x1o", "1x1o"]):
            if config.feature_extraction.preprocess[i]:
                feature_extraction_in_Irreps += f" + {irreps}"
            if config.sidechain.preprocess[i]:
                sidechain_in_Irreps += f" + {irreps}"
    #
    config.update_from_flattened_dict(
        {
            "initialization.in_Irreps": initialization_in_Irreps,
            "feature_extraction.in_Irreps": feature_extraction_in_Irreps,
            "transition.in_Irreps": transition_in_Irreps,
            "backbone.in_Irreps": backbone_in_Irreps,
            "sidechain.in_Irreps": sidechain_in_Irreps,
        }
    )
    #
    return config


gradient_checkpoint = functools.partial(
    torch.utils.checkpoint.checkpoint,
    use_reentrant=True,
)


class EmbeddingModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        #
        self.layer = nn.Embedding(config.num_embeddings, config.embedding_dim)

    def forward(self, batch):
        return self.layer(batch.residue_type)


class BaseModule(nn.Module):
    def __init__(self, config, compute_loss=False, checkpoint=False, num_recycle=1):
        super().__init__()
        #
        self.compute_loss = compute_loss
        self.checkpoint = checkpoint
        self.loss_weight = config.loss_weight
        self.preprocess = config.preprocess
        self.num_recycle = num_recycle
        #
        self.in_Irreps = o3.Irreps(config.in_Irreps).simplify()
        self.mid_Irreps = o3.Irreps(config.mid_Irreps).simplify()
        self.out_Irreps = o3.Irreps(config.out_Irreps).simplify()
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
        if config.layer_type not in ["ConvLayer", "SE3Transformer"] and self.activation:
            self.activation = e3nn.nn.Activation(
                self.mid_Irreps,
                [
                    self.activation if irrep.is_scalar() else None
                    for (n, irrep) in self.mid_Irreps
                ],
            )
        self.skip_connection = config.skip_connection
        for i, (norm, irreps) in enumerate(
            zip(config.norm, [self.in_Irreps, self.mid_Irreps, self.out_Irreps])
        ):
            if norm:
                self.add_module(f"norm_{i}", libnorm.LayerNorm(irreps))
            else:
                self.add_module(f"norm_{i}", None)
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
        if self.norm_0:
            feat = self.norm_0(feat)
        #
        if self.use_graph:
            out = self.forward_graph(batch, feat)
        else:
            out = self.forward_linear(feat)
        #
        if self.norm_2:
            out = self.norm_2(out)
        return out

    def forward_graph(self, batch, feat):
        if self.training and self.checkpoint:
            feat, graph = gradient_checkpoint(self.layer_0, batch, feat)
        else:
            feat, graph = self.layer_0(batch, feat)
        radius_prev = self.layer_0.radius
        #
        for i, layer in enumerate(self.layer_s):
            if self.skip_connection:
                feat0 = feat.clone()
            #
            if self.norm_1:
                feat = self.norm_1(feat)
            #
            if layer.radius == radius_prev:
                if self.training and self.checkpoint:
                    feat, graph = gradient_checkpoint(layer, batch, feat, graph)
                else:
                    feat, graph = layer(batch, feat, graph)
            else:
                if self.training and self.checkpoint:
                    feat, graph = gradient_checkpoint(layer, batch, feat)
                else:
                    feat, graph = layer(batch, feat)
            radius_prev = layer.radius
            #
            if self.skip_connection:
                feat = feat + feat0
        #
        if self.norm_1:
            feat = self.norm_1(feat)
        #
        if self.training and self.checkpoint:
            feat_out, graph = gradient_checkpoint(self.layer_1, batch, feat)
        else:
            feat_out, graph = self.layer_1(batch, feat)
        return feat_out

    def forward_linear(self, feat):
        if self.training and self.checkpoint:
            feat = gradient_checkpoint(self.layer_0, feat)
        else:
            feat = self.layer_0(feat)
        if self.activation is not None:
            feat = self.activation(feat)

        for i, layer in enumerate(self.layer_s):
            if self.skip_connection:
                feat0 = feat.clone()
            #
            if self.norm_1:
                feat = self.norm_1(feat)
            #
            if self.training and self.checkpoint:
                feat = gradient_checkpoint(layer, feat)
            else:
                feat = layer(feat)
            #
            if self.activation is not None:
                feat = self.activation(feat)
            #
            if self.skip_connection:
                feat = feat + feat0
        #
        if self.norm_1:
            feat = self.norm_1(feat)
        #
        if self.training and self.checkpoint:
            feat_out = gradient_checkpoint(self.layer_1, feat)
        else:
            feat_out = self.layer_1(feat)
        return feat_out

    def loss_f(self, batch, f_out):
        raise NotImplementedError

    def preprocess_feat(self, f_in):
        return f_in

    def test_equivariance(self, random_rotation, batch0, feat, *arg):
        device = feat.device
        feat = self.preprocess_feat(feat, *arg)
        #
        in_matrix = self.in_Irreps.D_from_matrix(random_rotation)
        out_matrix = self.out_Irreps.D_from_matrix(random_rotation)
        #
        random_rotation = random_rotation.to(device)
        in_matrix = in_matrix.to(device)
        out_matrix = out_matrix.to(device)
        #
        batch = copy.deepcopy(batch0)
        output = self.forward(batch, feat)
        output_0 = output.clone() @ out_matrix.T
        #
        batch.pos = batch.pos @ random_rotation.T
        batch.pos0 = batch.pos0 @ random_rotation.T
        batch.global_frame = rotate_vector(
            random_rotation, batch.global_frame.reshape(-1, 2, 3)
        ).reshape(-1, 6)
        feat = feat @ in_matrix.T
        output_1 = self.forward(batch, feat)
        #
        status = torch.allclose(
            output_0, output_1, rtol=EQUIVARIANT_TOLERANCE, atol=EQUIVARIANT_TOLERANCE
        )
        logging.debug(f"Equivariance test for {self.__class__.__name__} {status}")
        if not status:
            logging.debug(output_0[0])
            logging.debug(output_1[0])
            sys.exit("Could NOT pass equivariance test!")
        return output


class InitializationModule(BaseModule):
    def __init__(self, config, compute_loss=False, checkpoint=False, num_recycle=1):
        super().__init__(
            config,
            compute_loss=compute_loss,
            checkpoint=checkpoint,
            num_recycle=num_recycle,
        )

    def preprocess_feat(self, f_in, embedding):
        return torch.cat([embedding, f_in], dim=1)

    def forward(self, batch, f_in, embedding):
        feat = self.preprocess_feat(f_in, embedding)
        output = super().forward(batch, feat)
        return output

    def test_equivariance(self, random_rotation, batch0, feat, *arg):
        device = feat.device
        feat = self.preprocess_feat(feat, *arg)
        #
        in_matrix = self.in_Irreps.D_from_matrix(random_rotation)
        out_matrix = self.out_Irreps.D_from_matrix(random_rotation)
        #
        random_rotation = random_rotation.to(device)
        in_matrix = in_matrix.to(device)
        out_matrix = out_matrix.to(device)
        #
        batch = copy.deepcopy(batch0)
        output = super().forward(batch, feat)
        output_0 = output.clone() @ out_matrix.T
        #
        batch.pos = batch.pos @ random_rotation.T
        batch.pos0 = batch.pos0 @ random_rotation.T
        batch.global_frame = rotate_vector(
            random_rotation, batch.global_frame.reshape(-1, 2, 3)
        ).reshape(-1, 6)
        feat = feat @ in_matrix.T
        output_1 = super().forward(batch, feat)
        #
        status = torch.allclose(
            output_0, output_1, rtol=EQUIVARIANT_TOLERANCE, atol=EQUIVARIANT_TOLERANCE
        )
        logging.debug(f"Equivariance test for {self.__class__.__name__} {status}")
        if not status:
            logging.debug(output_0[0])
            logging.debug(output_1[0])
            sys.exit("Could NOT pass equivariance test!")
        return output


class TransitionModule(BaseModule):
    def __init__(self, config, compute_loss=False, checkpoint=False, num_recycle=1):
        super().__init__(
            config,
            compute_loss=compute_loss,
            checkpoint=checkpoint,
            num_recycle=num_recycle,
        )


class FeatureExtractionModule(BaseModule):
    def __init__(self, config, compute_loss=False, checkpoint=False, num_recycle=1):
        super().__init__(
            config,
            compute_loss=compute_loss,
            checkpoint=checkpoint,
            num_recycle=num_recycle,
        )

    def preprocess_feat(self, f_in, ret):
        if self.num_recycle < 2:
            return f_in
        #
        n_residue = f_in.size(0)
        feat = [f_in]
        if self.preprocess[0]:
            feat.append(ret["bb"][:, :3].mT[:, :2].reshape(n_residue, -1))
        if self.preprocess[1]:
            feat.append(ret["bb"][:, -1])
        feat = torch.cat(feat, dim=1)
        return feat

    def forward(self, batch, f_in, ret):
        feat = self.preprocess_feat(f_in, ret)
        output = super().forward(batch, feat)
        return output

    def test_equivariance(self, random_rotation, batch0, f_in, ret):
        device = f_in.device
        feat = self.preprocess_feat(f_in, ret)
        #
        in_matrix = self.in_Irreps.D_from_matrix(random_rotation)
        out_matrix = self.out_Irreps.D_from_matrix(random_rotation)
        #
        random_rotation = random_rotation.to(device)
        in_matrix = in_matrix.to(device)
        out_matrix = out_matrix.to(device)
        #
        batch = copy.deepcopy(batch0)
        output = super().forward(batch, feat)
        output_0 = output.clone() @ out_matrix.T
        #
        batch.pos = batch.pos @ random_rotation.T
        batch.pos0 = batch.pos0 @ random_rotation.T
        batch.global_frame = rotate_vector(
            random_rotation, batch.global_frame.reshape(-1, 2, 3)
        ).reshape(-1, 6)
        feat = feat @ in_matrix.T
        output_1 = super().forward(batch, feat)
        #
        status = torch.allclose(
            output_0, output_1, rtol=EQUIVARIANT_TOLERANCE, atol=EQUIVARIANT_TOLERANCE
        )
        logging.debug(f"Equivariance test for {self.__class__.__name__} {status}")
        if not status:
            logging.debug(output_0[0])
            logging.debug(output_1[0])
            sys.exit("Could NOT pass equivariance test!")
        return output


class BackboneModule(BaseModule):
    def __init__(self, config, compute_loss=False, checkpoint=False, num_recycle=1):
        super().__init__(
            config,
            compute_loss=compute_loss,
            checkpoint=checkpoint,
            num_recycle=num_recycle,
        )

    def loss_f(self, batch, bb, bb1, loss_prev, stop_grad=False):
        R, opr_bb = build_structure(batch, bb, sc=None, stop_grad=stop_grad)
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
        if self.loss_weight.get("rotation_matrix", 0.0) > 0.0:
            loss["rotation_matrix"] = (
                loss_f_rotation_matrix(bb, bb1, batch.correct_bb)
                * self.loss_weight.rotation_matrix
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
    def output_to_opr(bb):
        v0 = bb[:, 0:3]
        v1 = bb[:, 3:6]
        e0 = v_norm_safe(v0, index=0)
        u1 = v1 - e0 * inner_product(e0, v1)[:, None]
        e1 = v_norm_safe(u1, index=1)
        e2 = torch.cross(e0, e1)
        rot = torch.stack([e0, e1, e2], dim=1).mT
        #
        t = bb[:, 6:9][..., None, :]
        opr = torch.cat([rot, t], dim=1)
        return opr

    @staticmethod
    def compose(bb0, bb1):
        opr = BackboneModule.output_to_opr(bb1)
        opr[:, 3] = opr[:, 3] + bb0[:, 3]
        return opr

    def init_value(self, batch):
        device = batch.pos.device
        n_residue = batch.pos.size(0)
        t = torch.zeros((n_residue, 3), device=device)
        bb = torch.cat([batch.global_frame, t], dim=1)
        return BackboneModule.output_to_opr(bb)

    def test_equivariance(self, random_rotation, batch0, feat, *arg):
        device = feat.device
        feat = self.preprocess_feat(feat, *arg)
        #
        in_matrix = self.in_Irreps.D_from_matrix(random_rotation)
        out_matrix = self.out_Irreps.D_from_matrix(random_rotation)
        #
        random_rotation = random_rotation.to(device)
        in_matrix = in_matrix.to(device)
        out_matrix = out_matrix.to(device)
        #
        batch = copy.deepcopy(batch0)
        output = super().forward(batch, feat)
        output_0 = output.clone() @ out_matrix.T
        bb = self.output_to_opr(output)
        bb_0 = bb.clone()
        bb_0[:, :3] = rotate_matrix(random_rotation, bb_0[:, :3])
        bb_0[:, 3] = rotate_vector(random_rotation, bb_0[:, 3])
        #
        batch.pos = batch.pos @ random_rotation.T
        batch.pos0 = batch.pos0 @ random_rotation.T
        batch.global_frame = rotate_vector(
            random_rotation, batch.global_frame.reshape(-1, 2, 3)
        ).reshape(-1, 6)
        feat = feat @ in_matrix.T
        output_1 = super().forward(batch, feat)
        bb_1 = self.output_to_opr(output_1)
        #
        status = torch.allclose(
            output_0, output_1, rtol=EQUIVARIANT_TOLERANCE, atol=EQUIVARIANT_TOLERANCE
        ) and torch.allclose(
            bb_0, bb_1, rtol=EQUIVARIANT_TOLERANCE, atol=EQUIVARIANT_TOLERANCE
        )
        logging.debug(f"Equivariance test for {self.__class__.__name__} {status}")
        if not status:
            logging.debug(random_rotation)
            logging.debug(output[0])
            logging.debug(output_0[0])
            logging.debug(output_1[0])
            logging.debug(bb[0])
            logging.debug(bb_0[0])
            logging.debug(bb_1[0])
            sys.exit("Could NOT pass equivariance test!")
        return output


class SidechainModule(BaseModule):
    def __init__(self, config, compute_loss=False, checkpoint=False, num_recycle=1):
        super().__init__(
            config,
            compute_loss=compute_loss,
            checkpoint=checkpoint,
            num_recycle=num_recycle,
        )

    def preprocess_feat(self, f_in, bb):
        if self.num_recycle < 2:
            return f_in
        #
        n_residue = f_in.size(0)
        feat = [f_in]
        if self.preprocess[0]:
            feat.append(bb[:, :3].mT[:, :2].reshape(n_residue, -1))
        if self.preprocess[1]:
            feat.append(bb[:, -1])
        feat = torch.cat(feat, dim=1)
        return feat

    def forward(self, batch, feat, bb):
        feat = self.preprocess_feat(feat, bb)
        out = super().forward(batch, feat)
        return out

    @staticmethod
    def compose(sc):
        sc = sc.reshape(-1, MAX_TORSION, 2)
        return v_norm_safe(sc)

    def test_equivariance(self, random_rotation, batch0, feat, *arg):
        device = feat.device
        feat = self.preprocess_feat(feat, *arg)
        #
        in_matrix = self.in_Irreps.D_from_matrix(random_rotation)
        out_matrix = self.out_Irreps.D_from_matrix(random_rotation)
        #
        random_rotation = random_rotation.to(device)
        in_matrix = in_matrix.to(device)
        out_matrix = out_matrix.to(device)
        #
        batch = copy.deepcopy(batch0)
        output = super().forward(batch, feat)
        output_0 = output.clone() @ out_matrix.T
        #
        batch.pos = batch.pos @ random_rotation.T
        batch.pos0 = batch.pos0 @ random_rotation.T
        batch.global_frame = rotate_vector(
            random_rotation, batch.global_frame.reshape(-1, 2, 3)
        ).reshape(-1, 6)
        feat = feat @ in_matrix.T
        output_1 = super().forward(batch, feat)
        #
        status = torch.allclose(
            output_0, output_1, rtol=EQUIVARIANT_TOLERANCE, atol=EQUIVARIANT_TOLERANCE
        )
        logging.debug(f"Equivariance test for {self.__class__.__name__} {status}")
        if not status:
            logging.debug(output_0[0])
            logging.debug(output_1[0])
            sys.exit("Could NOT pass equivariance test!")
        return output

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


class Model(nn.Module):
    def __init__(self, _config, compute_loss=False, checkpoint=False):
        super().__init__()
        #
        self.compute_loss = compute_loss
        self.checkpoint = checkpoint
        self.num_recycle = _config.globals.num_recycle
        self.loss_weight = _config.globals.loss_weight
        #
        self.embedding_module = EmbeddingModule(_config.embedding)
        self.initialization_module = InitializationModule(
            _config.initialization,
            compute_loss=False,
            checkpoint=checkpoint,
            num_recycle=self.num_recycle,
        )
        self.feature_extraction_module = FeatureExtractionModule(
            _config.feature_extraction,
            compute_loss=False,
            checkpoint=checkpoint,
            num_recycle=self.num_recycle,
        )
        self.transition_module = TransitionModule(
            _config.transition,
            compute_loss=False,
            checkpoint=checkpoint,
            num_recycle=self.num_recycle,
        )
        self.backbone_module = BackboneModule(
            _config.backbone,
            compute_loss=compute_loss,
            checkpoint=checkpoint,
            num_recycle=self.num_recycle,
        )
        self.sidechain_module = SidechainModule(
            _config.sidechain,
            compute_loss=compute_loss,
            checkpoint=checkpoint,
            num_recycle=self.num_recycle,
        )

    def forward(self, batch):
        n_residue = batch.pos.size(0)
        device = batch.pos.device
        #
        loss = {}
        ret = {}
        ret["bb"] = self.backbone_module.init_value(batch).to(device)
        #
        # residue_type --> embedding
        embedding = self.embedding_module(batch)
        #
        # 16x0e + 4x1o --> 40x0e + 10x1o
        f_in = self.initialization_module(batch, batch.f_in, embedding)
        #
        for k in range(self.num_recycle):
            self.update_graph(batch, ret)
            #
            # 40x0e + 10x1o --> 40x0e + 10x1o
            f_out = self.feature_extraction_module(batch, f_in, ret)
            #
            # 40x0e + 10x1o --> 40x0e + 10x1o
            f_out = self.transition_module(batch, f_out)
            #
            # 40x0e + 10x1o --> 4x0e + 1x1o
            bb = self.backbone_module(batch, f_out)
            ret["bb"] = self.backbone_module.compose(ret["bb"], bb)
            #
            # 40x0e + 10x1o --> 14x0e
            sc = self.sidechain_module(batch, f_out, ret["bb"])
            ret["sc"] = self.sidechain_module.compose(sc)
            #
            if self.compute_loss or self.training:
                loss["bb"] = self.backbone_module.loss_f(
                    batch,
                    ret["bb"],
                    bb,
                    loss.get("bb", {}),
                    stop_grad=(k + 1 < self.num_recycle),
                )
                loss["sc"] = self.sidechain_module.loss_f(
                    batch, ret["sc"], None, loss.get("sc", {})
                )

        ret["R"], ret["opr_bb"] = build_structure(batch, ret["bb"], sc=ret["sc"])
        if self.compute_loss or self.training:
            loss["R"] = self.loss_f(batch, ret)
            for k, v in loss["bb"].items():
                loss["bb"][k] = v / self.num_recycle
            for k, v in loss["sc"].items():
                loss["sc"][k] = v / self.num_recycle

        metrics = self.calc_metrics(batch, ret)
        #
        return ret, loss, metrics

    def forward_for_develop(self, batch):
        device = batch.pos.device
        #
        intermediates = {}
        intermediates["embedding_module"] = []
        intermediates["initialization_module"] = []
        intermediates["feature_extraction_module"] = []
        intermediates["transition_module"] = []
        intermediates["backbone_module"] = []
        intermediates["sidechain_module"] = []
        intermediates["R"] = []
        intermediates["opr_bb"] = []
        intermediates["bb"] = []
        #
        ret = {}
        ret["bb"] = self.backbone_module.init_value(batch).to(device)
        #
        # residue_type --> embedding
        embedding = self.embedding_module(batch)
        intermediates["embedding_module"].append(embedding.clone())
        #
        # 38x0e + 4x1o --> 40x0e + 10x1o
        f_in = self.initialization_module(batch, batch.f_in, embedding)
        intermediates["initialization_module"].append(f_in.clone())
        #
        for k in range(self.num_recycle):
            self.update_graph(batch, ret)
            #
            # 40x0e + 10x1o --> 40x0e + 10x1o
            f_out = self.feature_extraction_module(batch, f_in, ret)
            intermediates["feature_extraction_module"].append(f_out.clone())
            #
            # 40x0e + 10x1o --> 40x0e + 10x1o
            f_out = self.transition_module(batch, f_out)
            intermediates["transition_module"].append(f_out.clone())
            #
            # 40x0e + 10x1o --> 4x0e + 1x1o
            bb = self.backbone_module(batch, f_out)
            ret["bb"] = self.backbone_module.compose(ret["bb"], bb)
            intermediates["backbone_module"].append(bb.clone())
            intermediates["bb"].append(ret["bb"].clone())
            #
            # 40x0e + 10x1o --> 14x0e
            sc = self.sidechain_module(batch, f_out, ret["bb"])
            ret["sc"] = self.sidechain_module.compose(sc)
            intermediates["sidechain_module"].append(sc.clone())
            #
            ret["R"], ret["opr_bb"] = build_structure(batch, ret["bb"], sc=ret["sc"])
            intermediates["R"].append(ret["R"].clone())
            intermediates["opr_bb"].append(ret["opr_bb"].clone())

        ret["R"], ret["opr_bb"] = build_structure(batch, ret["bb"], sc=ret["sc"])
        #
        return ret, intermediates

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
        if self.loss_weight.get("rotation_matrix", 0.0) > 0.0:
            loss["rotation_matrix"] = (
                loss_f_rotation_matrix(ret["bb"], None, batch.correct_bb)
                * self.loss_weight.rotation_matrix
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

    def update_graph(self, batch, ret):
        batch.pos = batch.pos0 + ret["bb"][:, 3]

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

    def test_equivariance(self, batch):
        device = batch.pos.device
        random_rotation = o3.rand_matrix()
        #
        ret = {}
        ret["bb"] = self.backbone_module.init_value(batch).to(device)
        #
        # sub-modules
        embedding = self.embedding_module(batch)
        #
        x = self.initialization_module.test_equivariance(
            random_rotation, batch, batch.f_in, embedding
        )
        x = self.feature_extraction_module.test_equivariance(
            random_rotation, batch, x, ret
        )
        x = self.transition_module.test_equivariance(random_rotation, batch, x)
        #
        bb = self.backbone_module.test_equivariance(random_rotation, batch, x)
        ret["bb"] = self.backbone_module.compose(ret["bb"], bb)
        #
        sc = self.sidechain_module.test_equivariance(
            random_rotation, batch, x, ret["bb"]
        )
        ret["sc"] = self.sidechain_module.compose(sc)

        # rotation matrices
        in_matrix = (
            o3.Irreps(batch.f_in_Irreps[0]).D_from_matrix(random_rotation).to(device)
        )
        random_rotation = random_rotation.to(device)
        #
        # the whole model
        batch_0 = copy.deepcopy(batch)
        out_0, X_0 = self.forward_for_develop(batch_0)
        r_0 = rotate_vector(random_rotation, out_0["R"])
        #
        batch_1 = copy.deepcopy(batch)
        batch_1.pos = batch_1.pos @ random_rotation.T
        batch_1.pos0 = batch_1.pos0 @ random_rotation.T
        batch_1.global_frame = rotate_vector(
            random_rotation, batch_1.global_frame.reshape(-1, 2, 3)
        ).reshape(-1, 6)
        batch_1.f_in = batch_1.f_in @ in_matrix.T
        out_1, X_1 = self.forward_for_develop(batch_1)
        r_1 = out_1["R"]
        #
        status = torch.allclose(
            r_0, r_1, rtol=EQUIVARIANT_TOLERANCE, atol=EQUIVARIANT_TOLERANCE
        )
        logging.debug(f"Equivariance test for {self.__class__.__name__} {status}")
        if status:
            return
        #
        D_from_matrix = {
            "initialization_module": self.initialization_module.out_Irreps.D_from_matrix(
                random_rotation.cpu()
            ),
            "feature_extraction_module": self.feature_extraction_module.out_Irreps.D_from_matrix(
                random_rotation.cpu()
            ),
            "transition_module": self.transition_module.out_Irreps.D_from_matrix(
                random_rotation.cpu()
            ),
            "backbone_module": self.backbone_module.out_Irreps.D_from_matrix(
                random_rotation.cpu()
            ),
            "sidechain_module": self.sidechain_module.out_Irreps.D_from_matrix(
                random_rotation.cpu()
            ),
        }
        #
        # test each step
        for k in range(self.num_recycle):
            for module_name, matrix in D_from_matrix.items():
                if k >= len(X_0[module_name]):
                    continue
                x0 = X_0[module_name][k] @ matrix.T.to(device)
                x1 = X_1[module_name][k]
                status = torch.allclose(
                    x0, x1, rtol=EQUIVARIANT_TOLERANCE, atol=EQUIVARIANT_TOLERANCE
                )
                logging.debug(
                    f"Equivariance test for model.{module_name}, iteration={k}, {status}"
                )
        #
        # opr_bb
        for k, (r0, r1) in enumerate(zip(X_0["opr_bb"], X_1["opr_bb"])):
            rot0 = rotate_matrix(random_rotation, r0[:, :3])
            rot1 = r1[:, :3]
            status = torch.allclose(
                rot0, rot1, rtol=EQUIVARIANT_TOLERANCE, atol=EQUIVARIANT_TOLERANCE
            )
            logging.debug(
                f"Equivariance test for model.opr_bb.rotation, iteration={k}, {status}"
            )
            tr0 = rotate_vector(random_rotation, r0[:, 3])
            tr1 = r1[:, 3]
            status = torch.allclose(
                tr0, tr1, rtol=EQUIVARIANT_TOLERANCE, atol=EQUIVARIANT_TOLERANCE
            )
            logging.debug(
                f"Equivariance test for model.opr_bb.translation, iteration={k}, {status}"
            )

        # R
        for k, (r0, r1) in enumerate(zip(X_0["R"], X_1["R"])):
            _r0 = rotate_vector(random_rotation, r0)
            status = torch.allclose(
                _r0, r1, rtol=EQUIVARIANT_TOLERANCE, atol=EQUIVARIANT_TOLERANCE
            )
            logging.debug(f"Equivariance test for model.R, iteration={k}, {status}")
        #
        torch.save(
            {
                "X_0": X_0,
                "X_1": X_1,
                "random_rotation": random_rotation,
                "D_from_matrix": D_from_matrix,
            },
            "equivariance_test.pt",
        )
        sys.exit("Could NOT pass equivariance test!")


def rotate_matrix(R, X):
    return R @ X


def rotate_vector(R, X):
    return (X[..., None, :] @ R.mT)[..., 0, :]


def combine_operations(X, Y):
    y = Y.clone()
    Y[..., :3, :] = rotate_matrix(X[..., :3, :], y[..., :3, :])
    Y[..., 3, :] = rotate_vector(X[..., :3, :], y[..., 3, :]) + X[..., 3, :]
    return Y


def build_structure(batch, bb, sc=None, stop_grad=False):
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
    # backbone operations
    if stop_grad:
        opr[:, 0, :3] = bb[:, :3].detach()
    else:
        opr[:, 0, :3] = bb[:, :3]
    opr[:, 0, 3] = bb[:, 3] + batch.pos0

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

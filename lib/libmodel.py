#!/usr/bin/env python

import sys
import copy
import functools
import logging
import numpy as np

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
    RIGID_TRANSFORMS_TENSOR,
    RIGID_TRANSFORMS_DEP,
    RIGID_GROUPS_TENSOR,
    RIGID_GROUPS_DEP,
)
from libloss_l1 import loss_f
from torch_basics import v_norm_safe, inner_product, rotate_matrix, rotate_vector
from libmetric import rmsd_CA, rmsd_rigid, rmsd_all, rmse_bonded
from libconfig import DTYPE, EQUIVARIANT_TOLERANCE


CONFIG = ConfigDict()

CONFIG["globals"] = ConfigDict()
CONFIG["globals"]["num_recycle"] = 2
CONFIG["globals"]["loss_weight"] = ConfigDict()
CONFIG["globals"]["loss_weight"].update(
    {
        "rigid_body": 1.0,
        "FAPE_CA": 5.0,
        "FAPE_all": 5.0,
        "mse_R": 1.0,
        "bonded_energy": 1.0,
        "distance_matrix": 0.0,
        "rotation_matrix": 1.0,
        "torsion_angle": 1.0,
        "atomic_clash": 1.0,
    }
)

# the base config for using ConvLayer or SE3Transformer
CONFIG_BASE = ConfigDict()
CONFIG_BASE["layer_type"] = "SE3Transformer"
CONFIG_BASE["num_layers"] = 2
CONFIG_BASE["in_Irreps"] = "40x0e + 10x1o"
CONFIG_BASE["out_Irreps"] = "40x0e + 10x1o"
CONFIG_BASE["mid_Irreps"] = "40x0e + 10x1o"
CONFIG_BASE["attn_Irreps"] = "40x0e + 10x1o"
CONFIG_BASE["l_max"] = 2
CONFIG_BASE["mlp_num_neurons"] = [20, 20]
CONFIG_BASE["activation"] = "elu"
CONFIG_BASE["radius"] = 0.8
CONFIG_BASE["norm"] = [False, False, False]
CONFIG_BASE["skip_connection"] = True
CONFIG_BASE["loss_weight"] = ConfigDict()

CONFIG["initialization"] = copy.deepcopy(CONFIG_BASE)
CONFIG["feature_extraction"] = copy.deepcopy(CONFIG_BASE)
CONFIG["output"] = copy.deepcopy(CONFIG_BASE)

CONFIG["embedding"] = ConfigDict()
CONFIG["embedding"]["num_embeddings"] = MAX_RESIDUE_TYPE
CONFIG["embedding"]["embedding_dim"] = 40

CONFIG["initialization"].update(
    {
        "layer_type": "Linear",
        "num_layers": 2,
        "in_Irreps": "16x0e + 4x1o",
        "out_Irreps": "40x0e + 10x1o",
        "mid_Irreps": "40x0e + 10x1o",
        "activation": "elu",
        "skip_connection": True,
        "norm": [False, True, True],
    }
)

CONFIG["feature_extraction"].update(
    {
        "layer_type": "SE3Transformer",
        "num_layers": 2,
        "radius": 0.8,
        "loop": False,
        "self_interaction": True,
        "in_Irreps": "40x0e + 10x1o",
        "out_Irreps": "40x0e + 10x1o",
        "mid_Irreps": "40x0e + 10x1o",
        "attn_Irreps": "80x0e + 20x1o",
        "activation": "elu",
        "skip_connection": True,
        "norm": [False, True, True],
    }
)

CONFIG["output"].update(
    {
        "layer_type": "Linear",
        "num_layers": 4,
        "in_Irreps": "40x0e + 10x1o",
        "mid_Irreps": "40x0e + 10x1o",
        "out_Irreps": f"3x1o + {MAX_TORSION*2:d}x0e",  # two for rotation, one for translation
        "activation": "elu",
        "skip_connection": True,
        "norm": [False, True, False],
        "loss_weight": {
            "rigid_body": 0.0,
            "FAPE_CA": 0.0,
            "bonded_energy": 0.0,
            "distance_matrix": 0.0,
            "rotation_matrix": 0.0,
            "torsion_angle": 0.0,
            "atomic_clash": 0.0,
        },
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
    config.update_from_flattened_dict(
        {
            "initialization.in_Irreps": initialization_in_Irreps,
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
    def __init__(self, config, compute_loss=False, checkpoint=False):
        super().__init__()
        #
        self.compute_loss = compute_loss
        self.checkpoint = checkpoint
        self.loss_weight = config.loss_weight
        #
        self.in_Irreps = o3.Irreps(config.in_Irreps).simplify()
        self.mid_Irreps = o3.Irreps(config.mid_Irreps).simplify()
        self.out_Irreps = o3.Irreps(config.out_Irreps).simplify()
        if config.activation is None:
            self.activation = None
        elif config.activation == "relu":
            self.activation = torch.nn.functional.relu
        elif config.activation == "elu":
            self.activation = torch.nn.functional.elu
        elif config.activation == "sigmoid":
            self.activation = torch.nn.functional.sigmoid
        else:
            raise NotImplementedError
        if config.layer_type not in ["ConvLayer", "SE3Transformer"] and self.activation:
            self.activation = e3nn.nn.Activation(
                self.mid_Irreps,
                [self.activation if irrep.is_scalar() else None for (n, irrep) in self.mid_Irreps],
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
                loop=config.get("loop", False),
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
                loop=config.get("loop", False),
                self_interaction=config.get("self_interaction", True),
                l_max=config.l_max,
                mlp_num_neurons=config.mlp_num_neurons,
                activation=self.activation,
            )

        elif config.layer_type == "Linear":
            self.use_graph = False
            layer_partial = functools.partial(o3.Linear, biases=True)
        #
        self.layer_0 = layer_partial(self.in_Irreps, self.mid_Irreps)
        self.layer_s = nn.ModuleList(
            [layer_partial(self.mid_Irreps, self.mid_Irreps) for _ in range(config.num_layers)]
        )
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
            feat, graph = gradient_checkpoint(self.layer_1, batch, feat)
        else:
            feat, graph = self.layer_1(batch, feat)
        return feat

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
            feat = gradient_checkpoint(self.layer_1, feat)
        else:
            feat = self.layer_1(feat)
        return feat

    def preprocess_feat(self, f_in, *arg):
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
    def __init__(self, config, compute_loss=False, checkpoint=False):
        super().__init__(
            config,
            compute_loss=compute_loss,
            checkpoint=checkpoint,
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


class FeatureExtractionModule(BaseModule):
    def __init__(self, config, compute_loss=False, checkpoint=False):
        super().__init__(
            config,
            compute_loss=compute_loss,
            checkpoint=checkpoint,
        )


class OutputModule(BaseModule):
    def __init__(self, config, compute_loss=False, checkpoint=False):
        super().__init__(
            config,
            compute_loss=compute_loss,
            checkpoint=checkpoint,
        )

    @staticmethod
    def output_to_opr(output):
        # rotation
        v0 = output[:, 0:3]
        v1 = output[:, 3:6]
        e0 = v_norm_safe(v0, index=0)
        u1 = v1 - e0 * inner_product(e0, v1)[:, None]
        e1 = v_norm_safe(u1, index=1)
        e2 = torch.cross(e0, e1)
        rot = torch.stack([e0, e1, e2], dim=1).mT
        #
        t = output[:, 6:9][..., None, :]
        bb = torch.cat([rot, t], dim=1)
        sc = output[:, 9:].reshape(-1, MAX_TORSION, 2)
        return bb, sc

    @staticmethod
    def compose(output, bb_prev, sc_prev):
        bb, sc = OutputModule.output_to_opr(output)
        bb[:, 3] = bb[:, 3] + bb_prev[:, 3]
        sc = sc_prev
        return bb, sc

    @staticmethod
    def init_value(batch):
        device = batch.pos.device
        n_residue = batch.pos.size(0)
        t = torch.zeros((n_residue, 3), device=device)
        sc = torch.zeros((n_residue, MAX_TORSION * 2), device=device)
        output = torch.cat([batch.global_frame, t, sc], dim=1)
        return OutputModule.output_to_opr(output)

    # def test_equivariance(self, random_rotation, batch0, feat, *arg):
    #     device = feat.device
    #     feat = self.preprocess_feat(feat, *arg)
    #     #
    #     in_matrix = self.in_Irreps.D_from_matrix(random_rotation)
    #     out_matrix = self.out_Irreps.D_from_matrix(random_rotation)
    #     #
    #     random_rotation = random_rotation.to(device)
    #     in_matrix = in_matrix.to(device)
    #     out_matrix = out_matrix.to(device)
    #     #
    #     batch = copy.deepcopy(batch0)
    #     output = super().forward(batch, feat)
    #     output_0 = output.clone() @ out_matrix.T
    #     bb = self.output_to_opr(output)
    #     bb_0 = bb.clone()
    #     bb_0[:, :3] = rotate_matrix(random_rotation, bb_0[:, :3])
    #     bb_0[:, 3] = rotate_vector(random_rotation, bb_0[:, 3])
    #     #
    #     batch.pos = batch.pos @ random_rotation.T
    #     batch.pos0 = batch.pos0 @ random_rotation.T
    #     batch.global_frame = rotate_vector(
    #         random_rotation, batch.global_frame.reshape(-1, 2, 3)
    #     ).reshape(-1, 6)
    #     feat = feat @ in_matrix.T
    #     output_1 = super().forward(batch, feat)
    #     bb_1 = self.output_to_opr(output_1)
    #     #
    #     status = torch.allclose(
    #         output_0, output_1, rtol=EQUIVARIANT_TOLERANCE, atol=EQUIVARIANT_TOLERANCE
    #     )
    #     logging.debug(f"Equivariance test for {self.__class__.__name__} {status}")
    #     if not status:
    #         logging.debug(random_rotation)
    #         logging.debug(output[0])
    #         logging.debug(output_0[0])
    #         logging.debug(output_1[0])
    #         logging.debug(torch.abs(output_0 - output_1).max())
    #         logging.debug(bb[0])
    #         logging.debug(bb_0[0])
    #         logging.debug(bb_1[0])
    #         logging.debug(torch.abs(bb_0 - bb_1).max())
    #         sys.exit("Could NOT pass equivariance test!")
    #     return output


class Model(nn.Module):
    def __init__(self, _config, cg_model, compute_loss=False, checkpoint=False):
        super().__init__()
        #
        self.cg_model = cg_model
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
        )
        self.feature_extraction_module = FeatureExtractionModule(
            _config.feature_extraction,
            compute_loss=False,
            checkpoint=checkpoint,
        )
        self.output_module = OutputModule(
            _config.output,
            compute_loss=compute_loss,
            checkpoint=checkpoint,
        )

    def forward(self, batch):
        n_residue = batch.pos.size(0)
        device = batch.pos.device
        #
        loss = {}
        ret = {}
        #
        # residue_type --> embedding
        embedding = self.embedding_module(batch)
        #
        # num_recycle = np.random.randint(1, self.num_recycle + 1)
        num_recycle = self.num_recycle
        for k in range(num_recycle):
            # 16x0e + 4x1o --> 40x0e + 10x1o
            f_in = self.initialization_module(batch, batch.f_in, embedding)
            #
            # 40x0e + 10x1o --> 40x0e + 10x1o
            f_out = self.feature_extraction_module(batch, f_in)
            #
            # 40x0e + 10x1o --> 3x1o + 14x0e
            f_out = self.output_module(batch, f_out)
            bb, sc = self.output_module.output_to_opr(f_out)
            ret["bb"] = bb
            ret["sc"] = sc
            #
            # build structure
            ret["R"], ret["opr_bb"] = build_structure(
                batch, ret["bb"], sc=ret["sc"], stop_grad=(k + 1 < num_recycle)
            )
            #
            if self.compute_loss or self.training:
                loss["intermediate"] = loss_f(
                    batch,
                    ret,
                    self.output_module.loss_weight,
                    loss.get("intermediate", {}),
                )
            #
            if k < num_recycle - 1:
                self.update_graph(batch, ret)

        if self.compute_loss or self.training:
            loss["final"] = loss_f(batch, ret, self.loss_weight)
            for k, v in loss["intermediate"].items():
                loss["intermediate"][k] = v / num_recycle

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
        intermediates["output_module"] = []
        intermediates["R"] = []
        intermediates["opr_bb"] = []
        intermediates["bb"] = []
        intermediates["sc"] = []
        #
        ret = {}
        #
        # residue_type --> embedding
        embedding = self.embedding_module(batch)
        intermediates["embedding_module"].append(embedding.clone())
        #
        for k in range(self.num_recycle):
            # 38x0e + 4x1o --> 40x0e + 10x1o
            f_in = self.initialization_module(batch, batch.f_in, embedding)
            intermediates["initialization_module"].append(f_in.clone())

            # 40x0e + 10x1o --> 40x0e + 10x1o
            f_out = self.feature_extraction_module(batch, f_in)
            intermediates["feature_extraction_module"].append(f_out.clone())
            #
            # 40x0e + 10x1o --> 3x1o + 14x0e
            f_out = self.output_module(batch, f_out)
            bb, sc = self.output_module.output_to_opr(f_out)
            ret["bb"] = bb
            ret["sc"] = sc
            intermediates["output_module"].append(f_out.clone())
            intermediates["bb"].append(ret["bb"].clone())
            intermediates["sc"].append(ret["sc"].clone())

            ret["R"], ret["opr_bb"] = build_structure(
                batch, ret["bb"], sc=ret["sc"], stop_grad=(k + 1 < self.num_recycle)
            )
            intermediates["R"].append(ret["R"].clone())
            intermediates["opr_bb"].append(ret["opr_bb"].clone())

            if k < self.num_recycle - 1:
                self.update_graph(batch, ret)
        #
        return ret, intermediates

    def update_graph(self, batch, ret):
        device = batch.f_in.device
        #
        pos = self.cg_model.convert_to_cg_tensor(ret["R"].detach(), batch.atomic_mass)
        err = ret.get("err", batch.f_in[:, 15])
        #
        f_in = []
        for k in range(batch.num_graphs):
            selected = batch.batch == k
            r = pos[selected]
            #
            geom_s = self.cg_model.get_geometry(
                r, batch.continuous[selected], batch.input_atom_mask[selected], pca=False
            )
            f_in.append(
                self.cg_model.geom_to_feature(geom_s, err[selected], dtype=batch.f_in.dtype)[0]
            )
        #
        batch.pos = pos[batch.input_atom_mask > 0.0]
        batch.f_in = torch.cat(f_in, axis=0)
        return batch

    def calc_metrics(self, batch, ret):
        R = ret["R"]
        R_ref = batch.output_xyz
        #
        metric_s = {}
        metric_s["rmsd_CA"] = rmsd_CA(R, R_ref)
        metric_s["rmsd_rigid"] = rmsd_rigid(R, R_ref)
        metric_s["rmsd_all"] = rmsd_all(R, R_ref, batch.output_atom_mask)
        #
        bonded = rmse_bonded(R, batch.continuous)
        metric_s["bond_length"] = bonded[0]
        metric_s["bond_angle"] = bonded[1]
        metric_s["omega_angle"] = bonded[2]
        return metric_s

    def test_equivariance(self, batch):
        device = batch.pos.device
        random_rotation = o3.rand_matrix()
        #
        # sub-modules
        embedding = self.embedding_module(batch)
        x = self.initialization_module.test_equivariance(
            random_rotation, batch, batch.f_in, embedding
        )
        x = self.feature_extraction_module.test_equivariance(random_rotation, batch, x)
        x = self.output_module.test_equivariance(random_rotation, batch, x)

        # rotation matrices
        in_matrix = o3.Irreps(batch.f_in_Irreps[0]).D_from_matrix(random_rotation).to(device)
        random_rotation = random_rotation.to(device)
        #
        # the whole model
        batch_0 = copy.deepcopy(batch)
        out_0, X_0 = self.forward_for_develop(batch_0)
        r_0 = rotate_vector(random_rotation, out_0["R"])
        r_0 = r_0[batch.output_atom_mask > 0.0]
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
        r_1 = r_1[batch.output_atom_mask > 0.0]
        #
        status = torch.allclose(r_0, r_1, rtol=EQUIVARIANT_TOLERANCE, atol=EQUIVARIANT_TOLERANCE)
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
            "output_module": self.output_module.out_Irreps.D_from_matrix(random_rotation.cpu()),
        }
        #
        # test each step
        for k in range(self.num_recycle):
            for module_name, matrix in D_from_matrix.items():
                if k >= len(X_0[module_name]):
                    continue
                x0 = X_0[module_name][k] @ matrix.T.to(device)
                x1 = X_1[module_name][k]
                delta = torch.abs(x0 - x1).max()
                status = torch.allclose(
                    x0, x1, rtol=EQUIVARIANT_TOLERANCE, atol=EQUIVARIANT_TOLERANCE
                )
                logging.debug(
                    f"Equivariance test for model.{module_name}, iteration={k}, {status} {delta}"
                )
        #
        # opr_bb
        for k, (r0, r1) in enumerate(zip(X_0["opr_bb"], X_1["opr_bb"])):
            rot0 = rotate_matrix(random_rotation, r0[:, :3])
            rot1 = r1[:, :3]
            delta = torch.abs(rot0 - rot1).max()
            status = torch.allclose(
                rot0, rot1, rtol=EQUIVARIANT_TOLERANCE, atol=EQUIVARIANT_TOLERANCE
            )
            logging.debug(
                f"Equivariance test for model.opr_bb.rotation, iteration={k}, {status} {delta}"
            )
            tr0 = rotate_vector(random_rotation, r0[:, 3])
            tr1 = r1[:, 3]
            delta = torch.abs(tr0 - tr1).max()
            status = torch.allclose(
                tr0, tr1, rtol=EQUIVARIANT_TOLERANCE, atol=EQUIVARIANT_TOLERANCE
            )
            logging.debug(
                f"Equivariance test for model.opr_bb.translation, iteration={k}, {status} {delta}"
            )

        # R
        for k, (r0, r1) in enumerate(zip(X_0["R"], X_1["R"])):
            _r0 = rotate_vector(random_rotation, r0)
            delta = torch.abs(_r0 - r1).max()
            status = torch.allclose(_r0, r1, rtol=EQUIVARIANT_TOLERANCE, atol=EQUIVARIANT_TOLERANCE)
            logging.debug(f"Equivariance test for model.R, iteration={k}, {status} {delta}")
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


def combine_operations(X, Y):
    y = Y.clone()
    Y[..., :3, :] = rotate_matrix(X[..., :3, :], y[..., :3, :])
    Y[..., 3, :] = rotate_vector(X[..., :3, :], y[..., 3, :]) + X[..., 3, :]
    return Y


def build_structure(batch, bb, sc=None, stop_grad=False):
    dtype = bb.dtype
    device = bb.device
    residue_type = batch.residue_type
    #
    if dtype != DTYPE:
        transforms = RIGID_TRANSFORMS_TENSOR[residue_type].to(device).type(dtype)
        rigids = RIGID_GROUPS_TENSOR[residue_type].to(device).type(dtype)
    else:
        transforms = RIGID_TRANSFORMS_TENSOR[residue_type].to(device)
        rigids = RIGID_GROUPS_TENSOR[residue_type].to(device)
    transforms_dep = RIGID_TRANSFORMS_DEP[residue_type].to(device)
    rigids_dep = RIGID_GROUPS_DEP[residue_type].to(device)
    #
    opr = torch.zeros_like(transforms, device=device)
    #
    # backbone operations
    if stop_grad:
        opr[:, 0, :3] = bb[:, :3].detach()
    else:
        opr[:, 0, :3] = bb[:, :3]
    # opr[:, 0, 3] = bb[:, 3] + batch.pos0
    opr[:, 0, 3] = bb[:, 3] + batch.pos

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

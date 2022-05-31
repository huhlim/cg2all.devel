#!/usr/bin/env python

import copy
import functools

import torch
import torch.nn as nn

import e3nn
import e3nn.nn
from e3nn import o3

import liblayer
from libconfig import DTYPE


CONFIG = {}

CONFIG["default"] = {}
CONFIG["default"]["num_layers"] = 3
CONFIG["default"]["layer_type"] = "ConvLayer"
CONFIG["default"]["in_Irreps"] = "22x0e + 2x1o"
CONFIG["default"]["out_Irreps"] = "20x0e + 10x1o"
CONFIG["default"]["mid_Irreps"] = "40x0e + 20x1o"
CONFIG["default"]["attn_Irreps"] = "40x0e + 20x1o"
CONFIG["default"]["l_max"] = 2
CONFIG["default"]["mlp_num_neurons"] = [20, 20]
CONFIG["default"]["activation"] = "relu"
CONFIG["default"]["norm"] = True
CONFIG["default"]["radius"] = 1.0
CONFIG["default"]["skip_connection"] = True

CONFIG["feature_extraction"] = copy.deepcopy(CONFIG["default"])
CONFIG["backbone"] = copy.deepcopy(CONFIG["default"])
CONFIG["sidechain"] = copy.deepcopy(CONFIG["default"])

CONFIG["backbone"]["num_layers"] = 2
CONFIG["backbone"]["in_Irreps"] = "20x0e + 10x1o"
CONFIG["backbone"]["out_Irreps"] = "1x0e + 1x1o"
CONFIG["backbone"]["mid_Irreps"] = "10x0e + 4x1o"
CONFIG["backbone"]["attn_Irreps"] = "10x0e + 4x1o"

CONFIG["sidechain"]["num_layers"] = 2
CONFIG["sidechain"]["in_Irreps"] = "20x0e + 10x1o"
CONFIG["sidechain"]["out_Irreps"] = "8x0e"
CONFIG["sidechain"]["mid_Irreps"] = "10x0e + 4x1o"
CONFIG["sidechain"]["attn_Irreps"] = "10x0e + 4x1o"


class BaseModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        #
        self.in_Irreps = o3.Irreps(config["in_Irreps"])
        self.mid_Irreps = o3.Irreps(config["mid_Irreps"])
        self.out_Irreps = o3.Irreps(config["out_Irreps"])
        if config["activation"] == "relu":
            activation = torch.relu
        else:
            raise NotImplementedError
        self.skip_connection = config["skip_connection"]
        #
        if config["layer_type"] == "ConvLayer":
            layer_partial = functools.partial(
                liblayer.ConvLayer,
                radius=config["radius"],
                l_max=config["l_max"],
                mlp_num_neurons=config["mlp_num_neurons"],
                activation=activation,
                norm=config["norm"],
            )

        elif config.layer_type == "SE3Transformer":
            layer_partial = functools.partial(
                liblayer.SE3Transformer,
                attn_Irreps=config["attn_Irreps"],
                radius=config["radius"],
                l_max=config["l_max"],
                mlp_num_neurons=config["mlp_num_neurons"],
                activation=activation,
                norm=config["norm"],
            )
        self.layer_0 = layer_partial(self.in_Irreps, self.mid_Irreps)
        self.layer_1 = layer_partial(self.mid_Irreps, self.out_Irreps)

        layer = layer_partial(self.mid_Irreps, self.mid_Irreps)
        self.layer_s = nn.ModuleList([layer for _ in range(config["num_layers"] - 2)])

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
        self.feature_extraction_module = BaseModule(_config["feature_extraction"])
        self.backbone_module = BaseModule(_config["backbone"])
        self.sidechain_module = BaseModule(_config["sidechain"])

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
        return ret

#!/usr/bin/env python

import sys
import functools
from typing import Optional, List

import torch
import torch.nn as nn

import torch_geometric
import torch_cluster
import torch_scatter

import e3nn
import e3nn.nn
from e3nn import o3

from libconfig import DTYPE, EPS


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_Irreps: str,
        out_Irreps: str,
        radius: float,
        loop: Optional[bool] = False,
        l_max: Optional[int] = 2,
        mlp_num_neurons: Optional[List[int]] = [20, 20],
        activation=torch.relu,
    ) -> None:
        #
        super().__init__()
        #
        self.in_Irreps = o3.Irreps(in_Irreps)
        self.out_Irreps = o3.Irreps(out_Irreps)
        self.sh_Irreps = o3.Irreps.spherical_harmonics(l_max)
        self.radius = radius
        self.loop = loop
        self.mlp_num_basis = mlp_num_neurons[0]
        #
        self.tensor_product = o3.FullyConnectedTensorProduct(
            self.in_Irreps, self.sh_Irreps, self.out_Irreps, shared_weights=False
        )
        #
        neurons = mlp_num_neurons + [self.tensor_product.weight_numel]
        self.mlp = e3nn.nn.FullyConnectedNet(neurons, act=activation)

    @staticmethod
    def count_neighbors(batch_index, edge_src):
        batch_size = batch_index.max().item() + 1
        n_neigh = torch.zeros(batch_size, device=edge_src.device, dtype=DTYPE)
        n_residue = torch.zeros(batch_size, device=edge_src.device, dtype=DTYPE)
        n_neigh.index_add_(0, batch_index[edge_src], torch.ones_like(edge_src, dtype=DTYPE))
        n_residue.index_add_(0, batch_index, torch.ones_like(batch_index, dtype=DTYPE))
        n_neigh = n_neigh / n_residue
        return n_neigh

    def forward(
        self,
        data: torch_geometric.data.Batch,
        f_in: torch.Tensor,
        graph=None,
    ) -> torch.Tensor:
        n_node = data.pos.size(0)
        if graph is None:
            edge_src, edge_dst = torch_cluster.radius_graph(
                data.pos,
                self.radius,
                loop=self.loop,
                batch=data.batch,
                max_num_neighbors=n_node - 1,
            )
            n_neigh = self.count_neighbors(data.batch, edge_src)
        else:
            edge_src, edge_dst, n_neigh = graph
        edge_vec = data.pos[edge_dst] - data.pos[edge_src]
        #
        sh = o3.spherical_harmonics(
            self.sh_Irreps, edge_vec, normalize=True, normalization="component"
        )
        #
        edge_length_embedding = e3nn.math.soft_one_hot_linspace(
            edge_vec.norm(dim=1),
            start=0.0,
            end=self.radius,
            number=self.mlp_num_basis,
            basis="smooth_finite",
            cutoff=True,
        )
        edge_length_embedding = edge_length_embedding.mul(self.mlp_num_basis**0.5)
        weight = self.mlp(edge_length_embedding)
        #
        f_out = self.tensor_product(f_in[edge_src], sh, weight)
        f_out = torch_scatter.scatter(f_out, edge_dst, dim=0, dim_size=n_node, reduce="sum")
        f_out = f_out.div(n_neigh[data.batch][:, None] ** 0.5)
        return f_out, (edge_src, edge_dst, n_neigh)


class SE3Transformer(nn.Module):
    def __init__(
        self,
        in_Irreps: str,
        out_Irreps: str,
        attn_Irreps: str,
        radius: float,
        loop: Optional[bool] = False,
        self_interaction: Optional[bool] = True,
        l_max: Optional[int] = 2,
        mlp_num_neurons: Optional[List[int]] = [20, 20],
        activation=torch.relu,
        return_attn: Optional[bool] = False,
    ) -> None:
        #
        super().__init__()
        #
        self.in_Irreps = o3.Irreps(in_Irreps)
        self.out_Irreps = o3.Irreps(out_Irreps)
        self.sh_Irreps = o3.Irreps.spherical_harmonics(l_max)
        self.radius = radius
        self.loop = loop
        self.mlp_num_basis = mlp_num_neurons[0]
        #
        self.attn_Irreps = o3.Irreps(attn_Irreps)  # for the query and key
        #
        if self_interaction:
            self.self_interaction = o3.Linear(self.in_Irreps, self.out_Irreps)
        else:
            self.self_interaction = None
        #
        # Query
        self.h_q = o3.Linear(self.in_Irreps, self.attn_Irreps)
        #
        # Key
        self.tensor_product_k = o3.FullyConnectedTensorProduct(
            self.in_Irreps, self.sh_Irreps, self.attn_Irreps, shared_weights=False
        )
        neurons_k = mlp_num_neurons + [self.tensor_product_k.weight_numel]
        self.mlp_k = e3nn.nn.FullyConnectedNet(neurons_k, act=activation)
        #
        # Value
        self.tensor_product_v = o3.FullyConnectedTensorProduct(
            self.in_Irreps, self.sh_Irreps, self.out_Irreps, shared_weights=False
        )
        neurons_v = mlp_num_neurons + [self.tensor_product_v.weight_numel]
        self.mlp_v = e3nn.nn.FullyConnectedNet(neurons_v, act=activation)
        #
        self.dot_product = o3.FullyConnectedTensorProduct(self.attn_Irreps, self.attn_Irreps, "0e")
        self.return_attn = return_attn

    def forward(
        self,
        data: torch_geometric.data.Batch,
        f_in: torch.Tensor,
        graph=None,
    ) -> torch.Tensor:
        n_node = data.pos.size(0)
        if graph is None:
            edge_src, edge_dst = torch_cluster.radius_graph(
                data.pos,
                self.radius,
                loop=self.loop,
                batch=data.batch,
                max_num_neighbors=n_node - 1,
            )
        else:
            edge_src, edge_dst = graph
        edge_vec = data.pos[edge_dst] - data.pos[edge_src]
        #
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = e3nn.math.soft_one_hot_linspace(
            edge_length,
            start=0.0,
            end=self.radius,
            number=self.mlp_num_basis,
            basis="smooth_finite",
            cutoff=True,
        )
        edge_length_embedding = edge_length_embedding.mul(self.mlp_num_basis**0.5)
        edge_weight_cutoff = e3nn.math.soft_unit_step(10.0 * (1.0 - edge_length / self.radius))
        #
        # compute the queries (per node), keys (per edge), and values (per edge)
        sh = o3.spherical_harmonics(
            self.sh_Irreps, edge_vec, normalize=True, normalization="component"
        )
        #
        q = self.h_q(f_in)
        k = self.tensor_product_k(f_in[edge_src], sh, self.mlp_k(edge_length_embedding))
        v = self.tensor_product_v(f_in[edge_src], sh, self.mlp_v(edge_length_embedding))
        #
        # compute the softmax (per edge)
        _dot = self.dot_product(q[edge_dst], k)
        dot = _dot - _dot.max()  # to avoid numerical issues
        exp = edge_weight_cutoff[:, None] * dot.exp()
        z = torch_scatter.scatter(exp, edge_dst, dim=0, dim_size=len(f_in))
        z[z < EPS] = 1.0
        alpha = exp / z[edge_dst]
        #
        f_out = torch_scatter.scatter(
            (alpha.relu() + EPS).sqrt() * v, edge_dst, dim=0, dim_size=len(f_in)
        )
        # self-interaction
        if self.self_interaction:
            f_out = f_out + self.self_interaction(f_in)
        #
        if self.return_attn:
            attn = torch.zeros((n_node, n_node), dtype=torch.float)
            attn[edge_src, edge_dst] = alpha[:, 0]
            return f_out, (edge_src, edge_dst), attn
        else:
            return f_out, (edge_src, edge_dst)

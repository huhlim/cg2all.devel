#!/usr/bin/env python

# %%
# load modules
import torch
import torch.nn as nn
from torch_cluster import radius_graph
from torch_scatter import scatter

import e3nn
import e3nn.o3
import e3nn.nn
import matplotlib.pyplot as plt
import numpy as np

# %%
class Convolution(nn.Module):
    def __init__(self, 
                input_rep: str, 
                output_rep: str,
                l_max: int, 
                num_radial_basis: int,
                num_radial_dimension: int,
                max_radius: float,
                radial_activation_function,
                *arg, **kwarg):
        super().__init__(*arg, **kwarg)
        #
        self.irreps_input = e3nn.o3.Irreps(input_rep)
        self.irreps_output = e3nn.o3.Irreps(output_rep)
        self.irreps_sh = e3nn.o3.Irreps.spherical_harmonics(lmax=l_max)
        #
        self.num_radial_basis = num_radial_basis
        self.num_radial_dimension = num_radial_dimension
        self.max_radius = max_radius
        #
        self.tp = e3nn.o3.FullyConnectedTensorProduct(self.irreps_input, 
                        self.irreps_sh, self.irreps_output, shared_weights=False)
        self.fc = e3nn.nn.FullyConnectedNet( \
            [self.num_radial_basis, self.num_radial_dimension, self.tp.weight_numel], \
            radial_activation_function)
    def get_graph(self, data):
        num_nodes = len(data.pos)
        edge_src, edge_dst = radius_graph(data.pos, self.max_radius, \
            max_num_neighbors=num_nodes-1, batch=data.batch)
        edge_vec = data.pos[edge_dst] - data.pos[edge_src]
        return edge_src, edge_dst, edge_vec
    def forward(self, data, f_in, edge_src=None, edge_dst=None, edge_vec=None):
        # define graph
        num_nodes = len(data.pos)
        if edge_vec is None:
            edge_src, edge_dst, edge_vec = self.get_graph(data)
        num_neighbors = len(edge_src) / len(data.pos)
        #
        # evaluate spherical harmonics, Y(x_{ij}/norm(x_{ij}))
        sh = e3nn.o3.spherical_harmonics(self.irreps_sh, \
                edge_vec, normalize=True, normalization='component')
        #
        # evaluate radial component, h(norm(x_{ij}))
        radial_embedding = e3nn.math.soft_one_hot_linspace( \
                edge_vec.norm(dim=1), start=0., end=self.max_radius, \
                number=self.num_radial_basis, \
                basis='smooth_finite', 
                cutoff=True)
        radial_embedding = radial_embedding.mul(self.num_radial_basis**0.5)
        weight = self.fc(radial_embedding)
        #
        # evaluate tensor product, f_j (x) hY()
        summand = self.tp(f_in[edge_src], sh, weight)

        # sum over neighboring points
        f_out = scatter(summand, edge_dst, dim=0, dim_size=num_nodes)
        f_out = f_out.div(num_neighbors**0.5)

        return f_out
# %%
class SE3_Transformer(nn.Module):
    def __init__(self, 
                input_rep: str, 
                query_rep: str,
                key_rep: str,
                value_rep: str,
                l_max: int, 
                num_radial_basis: int,
                num_radial_dimension: int,
                max_radius: float,
                radial_activation_function,
                *arg, **kwarg):
        super().__init__(*arg, **kwarg)
        #
        self.irreps_input = e3nn.o3.Irreps(input_rep)
        self.irreps_query = e3nn.o3.Irreps(query_rep)
        self.irreps_key = e3nn.o3.Irreps(key_rep)
        self.irreps_value = e3nn.o3.Irreps(value_rep)
        self.irreps_sh = e3nn.o3.Irreps.spherical_harmonics(lmax=l_max)
        #
        self.num_radial_basis = num_radial_basis
        self.num_radial_dimension = num_radial_dimension
        self.max_radius = max_radius
        #
        self.h_q = e3nn.o3.Linear(self.irreps_input, self.irreps_query)
        #
        self.tp_k = e3nn.o3.FullyConnectedTensorProduct(self.irreps_input, \
            self.irreps_sh, self.irreps_key, shared_weights=False)
        self.fc_k = e3nn.nn.FullyConnectedNet(\
            [self.num_radial_basis, self.num_radial_dimension, self.tp_k.weight_numel], 
            radial_activation_function)
        #
        self.tp_v = e3nn.o3.FullyConnectedTensorProduct(self.irreps_input, \
            self.irreps_sh, self.irreps_value, shared_weights=False)
        self.fc_v = e3nn.nn.FullyConnectedNet(\
            [self.num_radial_basis, self.num_radial_dimension, self.tp_v.weight_numel], 
            radial_activation_function)
        #
        self.dot_product = e3nn.o3.FullyConnectedTensorProduct(self.irreps_query, self.irreps_key, "0e")
    def get_graph(self, data):
        num_nodes = len(data.pos)
        edge_src, edge_dst = radius_graph(data.pos, self.max_radius, \
            max_num_neighbors=num_nodes-1, batch=data.batch)
        edge_vec = data.pos[edge_dst] - data.pos[edge_src]
        return edge_src, edge_dst, edge_vec
    def forward(self, data, f_in, edge_src=None, edge_dst=None, edge_vec=None):
        # define graph
        num_nodes = len(data.pos)
        if edge_vec is None:
            edge_src, edge_dst, edge_vec = self.get_graph(data)
        edge_len = edge_vec.norm(dim=1)
        #
        # evaluate spherical harmonics, Y(x_{ij}/norm(x_{ij}))
        sh = e3nn.o3.spherical_harmonics(self.irreps_sh, \
                edge_vec, normalize=True, normalization='component')
        #
        # evaluate radial component, h(norm(x_{ij}))
        radial_embedding = e3nn.math.soft_one_hot_linspace( \
                edge_len, start=0., end=self.max_radius, \
                number=self.num_radial_basis, \
                basis='smooth_finite', 
                cutoff=True)
        radial_embedding = radial_embedding.mul(self.num_radial_basis**0.5)
        edge_weight_cutoff = e3nn.math.soft_unit_step(10 * (1 - edge_len / self.max_radius))
        #
        # compute Q, K, V
        q = self.h_q(f_in)
        k = self.tp_k(f_in[edge_src], sh, self.fc_k(radial_embedding))
        v = self.tp_v(f_in[edge_src], sh, self.fc_v(radial_embedding))
        #
        # compute the softmax
        exp = edge_weight_cutoff[:, None] * self.dot_product(q[edge_dst], k).exp()
        z = scatter(exp, edge_dst, dim=0, dim_size=num_nodes)
        z[z == 0] = 1.
        alpha = exp / z[edge_dst]
        #
        # compute output
        f_out = scatter(alpha.relu().sqrt() * v, edge_dst, dim=0, dim_size=num_nodes)
        return f_out

#%%
import torch_geometric
def tetris():
    pos = [
        [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
        [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)],  # chiral_shape_2
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # L
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # T
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)],  # zigzag
    ]
    pos = torch.tensor(pos, dtype=torch.get_default_dtype())

    # Since chiral shapes are the mirror of one another we need an *odd* scalar to distinguish them
    labels = torch.tensor([
        [+1, 0, 0, 0, 0, 0, 0],  # chiral_shape_1
        [-1, 0, 0, 0, 0, 0, 0],  # chiral_shape_2
        [0, 1, 0, 0, 0, 0, 0],  # square
        [0, 0, 1, 0, 0, 0, 0],  # line
        [0, 0, 0, 1, 0, 0, 0],  # corner
        [0, 0, 0, 0, 1, 0, 0],  # L
        [0, 0, 0, 0, 0, 1, 0],  # T
        [0, 0, 0, 0, 0, 0, 1],  # zigzag
    ], dtype=torch.get_default_dtype())
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)

    # apply random rotation
    pos = torch.einsum('zij,zaj->zai', e3nn.o3.rand_matrix(len(pos)), pos)

    # put in torch_geometric format
    dataset = [torch_geometric.data.Data(pos=pos) for pos in pos]
    data = next(iter(torch_geometric.loader.DataLoader(dataset, batch_size=len(dataset))))

    return data, labels    
data, labels = tetris()
# %%
def calc_accuracy(pred, label):
    return ((torch.argmax(pred, dim=1) == label).sum() / len(label)).item()
# %%
class Model(torch.nn.Module):
    def __init__(self): 
        super().__init__()
        self.input_rep = e3nn.o3.Irreps.spherical_harmonics(3)
        self.layer = SE3_Transformer(self.input_rep,
                                    "10x0e + 5x1o",
                                    "10x0e + 5x1o",
                                    "10x0e + 5x1o",
                                    l_max=2,
                                    num_radial_basis=10,
                                    num_radial_dimension=10,
                                    max_radius=3.8,
                                    radial_activation_function=torch.nn.functional.elu)
        self.final = SE3_Transformer("10x0e + 5x1o",
                                    "10x0e + 5x1o",
                                    "10x0e + 5x1o",
                                    "8x0e",
                                    l_max=2,
                                    num_radial_basis=10,
                                    num_radial_dimension=10,
                                    max_radius=3.8,
                                    radial_activation_function=torch.nn.functional.elu)

    def forward(self, data):
        edge_src, edge_dst, edge_vec = self.layer.get_graph(data)
        x = e3nn.o3.spherical_harmonics(
            l=self.input_rep,
            x=edge_vec, normalize=True, normalization='component')
        x = self.layer(data, x, edge_src, edge_dst, edge_vec)
        x = self.final(data, x, edge_src, edge_dst, edge_vec)
        x = scatter(x, data.batch, dim=0)
        x = nn.Softmax(dim=1)(x)
        return x

model = Model()
print (calc_accuracy(model(data), labels))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_f = torch.nn.functional.cross_entropy
# %%
for step in range(200):
    pred = model(data)
    loss = loss_f(pred, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step%10 == 9:
        print(step+1, loss.item(), calc_accuracy(pred, labels))
model(data).detach().numpy().round(1)
# %%
data, labels = tetris()
output = model(data)
print (calc_accuracy(output, labels))
print (output.detach().numpy().round(2))
# %%

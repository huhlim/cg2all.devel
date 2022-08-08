#!/usr/bin/env python

import torch
from torch import nn

from e3nn import o3
from e3nn.util.jit import compile_mode


@compile_mode("unsupported")
class LayerNorm(nn.Module):
    __constants__ = ["irreps", "eps", "elementwise_affine"]
    irreps: o3.Irreps
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        irreps: o3.Irreps,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.irreps = o3.Irreps(irreps)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        self.dim = self.irreps.dim
        num_scalar = sum(mul for mul, ir in self.irreps if ir.is_scalar())
        num_features = self.irreps.num_irreps

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def extra_repr(self) -> str:
        return "{irreps}, eps={eps}, " "elementwise_affine={elementwise_affine}".format(
            **self.__dict__
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, *size, dim_total = input.shape
        assert dim_total == self.dim
        input = input.reshape(batch_size, -1, dim_total)  # [batch, sample, stacked features]

        ix = 0
        i_w = 0
        i_b = 0
        field_s = []
        for i, (n, irrep) in enumerate(self.irreps):
            dim = irrep.dim
            field = input[:, :, ix : ix + n * dim]  # [batch, sample, n * dim]
            ix += n * dim
            #
            if irrep.is_scalar():
                field_mean = field.mean(dim=(1, 2), keepdim=True)
                field = field - field_mean
                field_norm = field.pow(2).mean(dim=(1, 2), keepdim=True)
            else:
                field_norm = o3.Norm(n * irrep, squared=True)(field).mean(
                    dim=(1, 2), keepdim=True
                )  # [batch, 1, 1]
            field_norm = torch.pow(field_norm + self.eps, -0.5)
            field = field * field_norm
            #
            if self.elementwise_affine:
                field = field.reshape(batch_size, -1, n, dim)
                weight = self.weight[i_w : i_w + n][None, None, :, None]
                i_w += n
                #
                if irrep.is_scalar():
                    bias = self.bias[i_b : i_b + n][None, None, :, None]
                    i_b += n
                    #
                    field = field * weight + bias
                else:
                    field = field * weight
                field = field.reshape(batch_size, -1, n * dim)
            #
            field_s.append(field)
        output = torch.cat(field_s, dim=2)
        return output.reshape(batch_size, *size, dim_total)
        #

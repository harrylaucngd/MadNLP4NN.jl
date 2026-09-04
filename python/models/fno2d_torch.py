"""Compact PyTorch FNO2d used to train reproducible Darcy checkpoints.

The layer layout and parameter names intentionally match the MIT-licensed
PDEBench FNO2d implementation so that ``pdebench_fno.py`` can evaluate the
same checkpoint in JAX. The forward API is simplified to one coefficient
field and the left-endpoint grid used by NeuralOperator's ``GridEmbedding2D``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes1, modes2)
        self.weights1 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))

    def forward(self, value):
        transformed = torch.fft.rfft2(value)
        output = torch.zeros(
            value.shape[0],
            self.out_channels,
            value.shape[-2],
            value.shape[-1] // 2 + 1,
            dtype=transformed.dtype,
            device=value.device,
        )
        output[:, :, : self.modes1, : self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            transformed[:, :, : self.modes1, : self.modes2],
            self.weights1,
        )
        output[:, :, -self.modes1 :, : self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            transformed[:, :, -self.modes1 :, : self.modes2],
            self.weights2,
        )
        return torch.fft.irfft2(output, s=value.shape[-2:])


class FNO2d(nn.Module):
    def __init__(
        self,
        *,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        padding: int = 2,
        layer_count: int = 4,
    ):
        super().__init__()
        self.padding = padding
        self.layer_count = layer_count
        self.fc0 = nn.Linear(3, width)
        for index in range(layer_count):
            setattr(self, f"conv{index}", SpectralConv2d(width, width, modes1, modes2))
            setattr(self, f"w{index}", nn.Conv2d(width, width, 1))
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    @staticmethod
    def _grid(batch: int, nx: int, ny: int, *, device, dtype):
        coordinate_x = torch.arange(nx, device=device, dtype=dtype) / nx
        coordinate_y = torch.arange(ny, device=device, dtype=dtype) / ny
        grid_x, grid_y = torch.meshgrid(coordinate_x, coordinate_y, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1).expand(batch, -1, -1, -1)

    def forward(self, coefficient):
        if coefficient.ndim == 3:
            coefficient = coefficient.unsqueeze(-1)
        if coefficient.ndim != 4 or coefficient.shape[-1] != 1:
            raise ValueError("FNO2d expects [batch, nx, ny, 1] coefficients")
        batch, nx, ny, _ = coefficient.shape
        grid = self._grid(
            batch, nx, ny, device=coefficient.device, dtype=coefficient.dtype
        )
        value = self.fc0(torch.cat((coefficient, grid), dim=-1))
        value = value.permute(0, 3, 1, 2)
        value = functional.pad(value, (0, self.padding, 0, self.padding))
        for index in range(self.layer_count):
            value = getattr(self, f"conv{index}")(value) + getattr(
                self, f"w{index}"
            )(value)
            if index + 1 < self.layer_count:
                value = functional.gelu(value)
        value = value[..., :nx, :ny].permute(0, 2, 3, 1)
        value = functional.gelu(self.fc1(value))
        return self.fc2(value).squeeze(-1)

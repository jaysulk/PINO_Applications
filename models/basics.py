import numpy as np
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

def fht1d(x):
    N = x.shape[-1]
    # Ensure N is a power of 2
    if N % 2 != 0:
        next_pow_two = 1 << (N - 1).bit_length()
        pad_size = next_pow_two - N
        x = F.pad(x, (0, pad_size))
        N = next_pow_two

    if N == 1:
        return x
    else:
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        fht_even = fht1d(x_even)
        fht_odd = fht1d(x_odd)
        k = torch.arange(N // 2, device=x.device).reshape([1] * (x.ndim - 1) + [-1])
        theta = 2 * torch.pi * k / N
        cas = torch.cos(theta) + torch.sin(theta)
        temp = cas * fht_odd
        X = torch.cat([fht_even + temp, fht_even - temp], dim=-1)
        return X[..., :N]  # Truncate to original length if padded

def fht_along_dim(x, dim):
    # Move the target dimension to the last dimension
    x = x.transpose(dim, -1)
    original_shape = x.shape
    # Flatten the batch dimensions
    x = x.reshape(-1, x.shape[-1])
    # Apply fht1d
    x = fht1d(x)
    # Restore the original shape
    x = x.reshape(original_shape)
    # Move the last dimension back to its original position
    x = x.transpose(dim, -1)
    return x

def dht(x: torch.Tensor, dims=None) -> torch.Tensor:
    if dims is None:
        dims = list(range(2, x.ndim))
    for dim in dims:
        x = fht_along_dim(x, dim)
    return x

def idht(x: torch.Tensor, dims=None) -> torch.Tensor:
    if dims is None:
        dims = list(range(2, x.ndim))
    N = 1
    for dim in dims:
        N *= x.size(dim)
    # Compute the DHT (Inverse Hartley Transform)
    transformed = dht(x, dims=dims)
    # Normalize the result
    return transformed / N

def compl_mul1d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bi...,io...->bo...", x1, x2)

def compl_mul2d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bixy...,ioxy...->boxy...", x1, x2)

def compl_mul3d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bixyz...,ioxyz...->boxyz...", x1, x2)

################################################################
# 1D Hartley convolution layer
################################################################

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Hartley layer. It does DHT, linear transform, and Inverse DHT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Hartley modes to multiply
        self.modes1 = modes1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1))

    def forward(self, x):
        batchsize = x.shape[0]

        # Compute Hartley coefficients
        x_ht = dht(x, dims=[2])

        # Multiply relevant Hartley modes
        out_ht = torch.zeros(batchsize, self.out_channels, x.size(-1), device=x.device, dtype=x.dtype)
        out_ht[:, :, :self.modes1] = compl_mul1d(x_ht[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = idht(out_ht, dims=[2])

        return x

################################################################
# 2D Hartley convolution layer
################################################################

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))

    def forward(self, x):
        batchsize = x.shape[0]
        size1 = x.shape[-2]
        size2 = x.shape[-1]

        # Compute Hartley coefficients
        x_ht = dht(x, dims=[2, 3])

        # Multiply relevant Hartley modes
        out_ht = torch.zeros(batchsize, self.out_channels, size1, size2, device=x.device, dtype=x.dtype)
        out_ht[:, :, :self.modes1, :self.modes2] = compl_mul2d(
            x_ht[:, :, :self.modes1, :self.modes2], self.weights1)

        # Return to physical space
        x = idht(out_ht, dims=[2, 3])

        return x

################################################################
# 3D Hartley convolution layer
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Hartley modes to multiply
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(
            in_channels, out_channels, self.modes1, self.modes2, self.modes3))

    def forward(self, x):
        batchsize = x.shape[0]
        size1 = x.shape[-3]
        size2 = x.shape[-2]
        size3 = x.shape[-1]

        # Compute Hartley coefficients
        x_ht = dht(x, dims=[2, 3, 4])

        # Multiply relevant Hartley modes
        out_ht = torch.zeros(batchsize, self.out_channels, size1, size2, size3, device=x.device, dtype=x.dtype)
        out_ht[:, :, :self.modes1, :self.modes2, :self.modes3] = compl_mul3d(
            x_ht[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)

        # Return to physical space
        x = idht(out_ht, dims=[2, 3, 4])

        return x

################################################################
# FourierBlock (Using SpectralConv3d)
################################################################

class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, activation='tanh'):
        super(FourierBlock, self).__init__()

        # Spectral convolution layer (using 3D Hartley transform)
        self.speconv = SpectralConv3d(in_channels, out_channels, modes1, modes2, modes3)

        # Linear layer applied across the channel dimension
        self.linear = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        # Activation function selection
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError(f'{activation} is not supported')

    def forward(self, x):
        '''
        Input x: (batchsize, in_channels, x_grid, y_grid, t_grid)
        '''
        x1 = self.speconv(x)
        x2 = self.linear(x)
        x = x1 + x2

        if self.activation is not None:
            x = self.activation(x)

        return x

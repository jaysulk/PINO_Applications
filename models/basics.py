import numpy as np

import torch
import torch.nn as nn

from functools import partial

import torch.nn.functional as F
import math

from typing import List, Tuple

def dht(x: torch.Tensor, dims: List[int]) -> torch.Tensor:
    """
    Manually compute the Discrete Fourier Transform (DFT) along the specified dimensions.
    Args:
        x: Input tensor (3D, 4D, or 5D).
        dims: List of dimensions along which to compute the DFT.
    Returns:
        Real-valued tensor.
    """
    # Convert input to complex type if necessary
    x = x.to(torch.complex64) if x.dtype == torch.float32 else x.to(torch.complex128)
    
    for dim in dims:
        N = x.size(dim)
        
        # Create the frequency grid for the current dimension
        n = torch.arange(N, device=x.device).view([1] * dim + [N])
        k = torch.arange(N, device=x.device).view([1] * dim + [N]).transpose(dim, -1)
        
        # Compute the DFT matrix for the given dimension (complex exponential)
        W = torch.exp(-2j * math.pi * k * n / N)

        # Reshape x for matrix multiplication along the specified dimension
        x_shape = x.shape
        new_shape = (-1, N)  # Collapse all dimensions except the DFT dimension
        x = x.reshape(*x_shape[:-1], N)
        
        # Perform DFT along the specified dimension
        x = torch.matmul(x, W)

        # Reshape back to original dimensions
        x = x.reshape(x_shape)

    # Return the real part of the result
    return x.real
ermute and reshape x such that `dim` becomes the last dimension, perform the multiplication, and then restore the original order.
        x = torch.moveaxis(x, dim, -1)  # Move the target dimension to the last axis
        x = torch.matmul(x, W)  # Perform DFT along the last axis
        x = torch.moveaxis(x, -1, dim)  # Restore the original order

    # Return the real part of the result
    return x.real

def idht(X: torch.Tensor, dims: List[int], s: Tuple[int]) -> torch.Tensor:
    """
    Manually compute the Inverse Discrete Fourier Transform (IDFT) along the specified dimensions.
    Args:
        X: Input tensor in the frequency domain.
        dims: List of dimensions along which to compute the IDFT.
        s: Shape of the output tensor for the specified dimensions.
    Returns:
        Real-valued tensor.
    """
    # Convert input to complex type if necessary
    X = X.to(torch.complex64) if X.dtype == torch.float32 else X.to(torch.complex128)
    
    for dim in dims:
        N = s[dims.index(dim)]
        
        # Create the frequency grid for the current dimension
        n = torch.arange(N, device=X.device).view([1] * dim + [N])
        k = torch.arange(N, device=X.device).view([1] * dim + [N]).transpose(dim, -1)
        
        # Compute the IDFT matrix for the given dimension (inverse complex exponential)
        W_inv = torch.exp(2j * math.pi * k * n / N)

        # Reshape X for matrix multiplication along the specified dimension
        X_shape = X.shape
        new_shape = (-1, N)  # Collapse all dimensions except the IDFT dimension
        X = X.reshape(*X_shape[:-1], N)
        
        # Perform IDFT along the specified dimension
        X = torch.matmul(X, W_inv)

        # Reshape back to original dimensions
        X = X.reshape(X_shape)
        
        # Normalize the result by the size of the dimension
        X = X / N
    
    # Return the real part of the result
    return X.real

def compl_mul1d(a, b):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    return torch.einsum("bix,iox->box", a, b)


def compl_mul2d(a, b):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    return torch.einsum("bixy,ioxy->boxy", a, b)


def compl_mul3d(a, b):
    return torch.einsum("bixyz,ioxyz->boxyz", a, b)

################################################################
# 1d fourier layer
################################################################

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Hartley layer. It does HHT, linear transform, and Inverse HHT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Hartley modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Hartley coefficients up to factor of h^(- something constant)
        x_ht = dht(x, dims=[2])

        # Multiply relevant Hartley modes
        out_ht = torch.zeros(batchsize, self.in_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ht[:, :, :self.modes1] = compl_mul1d(x_ht[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = idht(out_ht, dims=[2], s=(x.size(-1),))
        return x



################################################################
# 2d fourier layer
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
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))

    def forward(self, x):
        batchsize = x.shape[0]
        size1 = x.shape[-2]
        size2 = x.shape[-1]
        
        # Compute DHT
        x_dht = dht(x, dims=[2, 3])
        
        # Multiply relevant Hartley modes
        out_dht = torch.zeros(batchsize, self.out_channels, size1, size2, device=x.device)
        out_dht[:, :, :self.modes1, :self.modes2] = compl_mul2d(x_dht[:, :, :self.modes1, :self.modes2], self.weights1)
        out_dht[:, :, -self.modes1:, :self.modes2] = compl_mul2d(x_dht[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        # Return to physical space
        x = idht(out_dht, dims=[2, 3], s=(x.size(-2), x.size(-1)))
        
        return x

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Hartley modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        
        # Compute DHT (Discrete Hartley Transform) along the last three dimensions
        x_dht = dht(x, dims=[2, 3, 4])
        
        # Multiply relevant Hartley modes
        out_dht = torch.zeros(batchsize, self.out_channels, x.size(2), x.size(3), x.size(4)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_dht[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            compl_mul3d(x_dht[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_dht[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            compl_mul3d(x_dht[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_dht[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_dht[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_dht[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_dht[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space using inverse DHT
        x = idht(out_dht, dims=[2, 3, 4], s=(x.size(2), x.size(3), x.size(4)))
        return x

class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, activation='tanh'):
        super(FourierBlock, self).__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.speconv = SpectralConv3d(in_channels, out_channels, modes1, modes2, modes3)
        self.linear = nn.Conv1d(in_channels, out_channels, 1)
        if activation == 'tanh':
            self.activation = torch.tanh_
        elif activation == 'gelu':
            self.activation = nn.GELU
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError(f'{activation} is not supported')

    def forward(self, x):
        '''
        input x: (batchsize, channel width, x_grid, y_grid, t_grid)
        '''
        x1 = self.speconv(x)
        x2 = self.linear(x.view(x.shape[0], self.in_channel, -1))
        out = x1 + x2.view(x.shape[0], self.out_channel, x.shape[2], x.shape[3], x.shape[4])
        if self.activation is not None:
            out = self.activation(out)
        return out


import numpy as np

import torch
import torch.nn as nn

from functools import partial

import torch.nn.functional as F

import torch
import math

def manual_dft(x: torch.Tensor) -> torch.Tensor:
    """
    Manually compute the Discrete Fourier Transform (DFT) and return the real part.
    This function handles 1D, 2D, and 3D cases, returning a real-valued tensor.

    Args:
        x: Input tensor (3D for 1D DFT, 4D for 2D DFT, 5D for 3D DFT)

    Returns:
        Real-valued tensor (same torch.dtype as input)
    """
    # Convert input to complex type (complex64 for float32, complex128 for float64)
    x = x.to(torch.complex64) if x.dtype == torch.float32 else x.to(torch.complex128)

    if x.ndim == 3:
        # 1D DFT case (input is a 3D tensor)
        D, M, N = x.size()
        n = torch.arange(N, device=x.device).view(1, 1, N)
        k = torch.arange(N, device=x.device).view(1, N, 1)

        # Compute the DFT matrix (complex exponential)
        W = torch.exp(-2j * math.pi * k * n / N)

        # Compute the DFT using matrix multiplication
        X_complex = torch.matmul(x, W)

        # Return the real part, maintaining the same type as input
        return X_complex.real

    elif x.ndim == 4:
        # 2D DFT case (input is a 4D tensor)
        B, D, M, N = x.size()

        # Compute the 1D DFT for rows
        m = torch.arange(M, device=x.device).view(1, 1, M, 1)
        k_m = torch.arange(M, device=x.device).view(1, 1, 1, M)
        W_m = torch.exp(-2j * math.pi * k_m * m / M)

        # Compute the 1D DFT for columns
        n = torch.arange(N, device=x.device).view(1, 1, 1, N)
        k_n = torch.arange(N, device=x.device).view(1, 1, N, 1)
        W_n = torch.exp(-2j * math.pi * k_n * n / N)

        # Perform the row-wise DFT
        X_complex_rows = torch.matmul(W_m, x)

        # Perform the column-wise DFT
        X_complex = torch.matmul(X_complex_rows, W_n)

        # Return the real part, maintaining the same type as input
        return X_complex.real

    elif x.ndim == 5:
        # 3D DFT case (input is a 5D tensor)
        B, C, D, M, N = x.size()

        # Compute the 1D DFT for depth
        d = torch.arange(D, device=x.device).view(1, 1, D, 1, 1)
        k_d = torch.arange(D, device=x.device).view(1, 1, 1, D, 1)
        W_d = torch.exp(-2j * math.pi * k_d * d / D)

        # Compute the 1D DFT for rows
        m = torch.arange(M, device=x.device).view(1, 1, 1, M, 1)
        k_m = torch.arange(M, device=x.device).view(1, 1, 1, 1, M)
        W_m = torch.exp(-2j * math.pi * k_m * m / M)

        # Compute the 1D DFT for columns
        n = torch.arange(N, device=x.device).view(1, 1, 1, 1, N)
        k_n = torch.arange(N, device=x.device).view(1, 1, N, 1, 1)
        W_n = torch.exp(-2j * math.pi * k_n * n / N)

        # Perform the depth-wise DFT
        X_complex_depth = torch.matmul(W_d, x)

        # Perform the row-wise DFT
        X_complex_rows = torch.matmul(W_m, X_complex_depth)

        # Perform the column-wise DFT
        X_complex = torch.matmul(W_n, X_complex_rows)

        # Return the real part, maintaining the same type as input
        return X_complex.real

    else:
        raise ValueError(f"Input tensor must be 3D, 4D, or 5D, but got {x.ndim}D with shape {x.shape}.")

def idht(X: torch.Tensor) -> torch.Tensor:
    """
    Manually compute the Inverse Discrete Fourier Transform (IDFT) and return the real part.
    This function handles 1D, 2D, and 3D cases, returning a real-valued tensor.

    Args:
        X: Input tensor (3D for 1D IDFT, 4D for 2D IDFT, 5D for 3D IDFT)

    Returns:
        Real-valued tensor (same torch.dtype as input)
    """
    if X.ndim == 3:
        # 1D IDFT case (input is a 3D tensor)
        D, M, N = X.size()
        n = torch.arange(N, device=X.device).view(1, 1, N)
        k = torch.arange(N, device=X.device).view(1, N, 1)

        # Compute the IDFT matrix (inverse complex exponential)
        W_inv = torch.exp(2j * math.pi * k * n / N)

        # Compute the IDFT using matrix multiplication
        x_complex = torch.matmul(X, W_inv)

        # Normalize by the length N and return the real part
        return (x_complex.real / N)

    elif X.ndim == 4:
        # 2D IDFT case (input is a 4D tensor)
        B, D, M, N = X.size()

        # Compute the 1D IDFT for rows
        m = torch.arange(M, device=X.device).view(1, 1, M, 1)
        k_m = torch.arange(M, device=X.device).view(1, 1, 1, M)
        W_m_inv = torch.exp(2j * math.pi * k_m * m / M)

        # Compute the 1D IDFT for columns
        n = torch.arange(N, device=X.device).view(1, 1, 1, N)
        k_n = torch.arange(N, device=X.device).view(1, 1, N, 1)
        W_n_inv = torch.exp(2j * math.pi * k_n * n / N)

        # Perform the row-wise IDFT
        x_complex_rows = torch.matmul(W_m_inv, X)

        # Perform the column-wise IDFT
        x_complex = torch.matmul(x_complex_rows, W_n_inv)

        # Normalize by M and N and return the real part
        return (x_complex.real / (M * N))

    elif X.ndim == 5:
        # 3D IDFT case (input is a 5D tensor)
        B, C, D, M, N = X.size()

        # Compute the 1D IDFT for depth
        d = torch.arange(D, device=X.device).view(1, 1, D, 1, 1)
        k_d = torch.arange(D, device=X.device).view(1, 1, 1, D, 1)
        W_d_inv = torch.exp(2j * math.pi * k_d * d / D)

        # Compute the 1D IDFT for rows
        m = torch.arange(M, device=X.device).view(1, 1, 1, M, 1)
        k_m = torch.arange(M, device=X.device).view(1, 1, 1, 1, M)
        W_m_inv = torch.exp(2j * math.pi * k_m * m / M)

        # Compute the 1D IDFT for columns
        n = torch.arange(N, device=X.device).view(1, 1, 1, 1, N)
        k_n = torch.arange(N, device=X.device).view(1, 1, N, 1, 1)
        W_n_inv = torch.exp(2j * math.pi * k_n * n / N)

        # Perform the depth-wise IDFT
        x_complex_depth = torch.matmul(W_d_inv, X)

        # Perform the row-wise IDFT
        x_complex_rows = torch.matmul(W_m_inv, x_complex_depth)

        # Perform the column-wise IDFT
        x_complex = torch.matmul(W_n_inv, x_complex_rows)

        # Normalize by D, M, and N and return the real part
        return (x_complex.real / (D * M * N))

    else:
        raise ValueError(f"Input tensor must be 3D, 4D, or 5D, but got {X.ndim}D with shape {X.shape}.")


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
        x_ht = dht(x)

        # Multiply relevant Hartley modes
        out_ht = torch.zeros(batchsize, self.in_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ht[:, :, :self.modes1] = compl_mul1d(x_ht[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = idht(out_ht)
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
        x_dht = dht(x)
        
        # Multiply relevant Hartley modes
        out_dht = torch.zeros(batchsize, self.out_channels, size1, size2, device=x.device)
        out_dht[:, :, :self.modes1, :self.modes2] = compl_mul2d(x_dht[:, :, :self.modes1, :self.modes2], self.weights1)
        out_dht[:, :, -self.modes1:, :self.modes2] = compl_mul2d(x_dht[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        # Return to physical space
        x = idht(out_dht)
        
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
        # Compute Hartley coefficients up to factor of h^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2, 3, 4])
        x_ft_mirror = torch.fft.rfftn(x.flip(dims=[2, 3, 4]), dim=[2, 3, 4])  # F(-u)
        x_ht = x_ft + x_ft_mirror

        # Multiply relevant Hartley modes
        out_ht = torch.zeros(batchsize, self.out_channels, x.size(2), x.size(3), x.size(4)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ht[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            compl_mul3d(x_ht[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ht[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            compl_mul3d(x_ht[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ht[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_ht[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ht[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_ht[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ht, s=(x.size(2), x.size(3), x.size(4)), dim=[2, 3, 4])
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


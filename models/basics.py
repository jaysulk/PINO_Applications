import numpy as np

import torch
import torch.nn as nn

from functools import partial

import torch.nn.functional as F

import torch
import math

import torch
import math

def dft(x: torch.Tensor) -> torch.Tensor:
    """
    Manually compute the Real-Valued Discrete Fourier Transform (DFT) for 1D, 2D, or 3D input,
    handling real input without redundant information similar to rfftn.

    Args:
        x: Input tensor (3D for 1D DFT, 4D for 2D DFT, 5D for 3D DFT).

    Returns:
        Real-valued tensor (same torch.dtype as input).
    """
    # Convert input to complex type (complex64 for float32, complex128 for float64)
    x = x.to(torch.complex64) if x.dtype == torch.float32 else x.to(torch.complex128)

    if x.ndim == 3:
        # 1D DFT case
        D, M, N = x.size()
        n = torch.arange(N, device=x.device).view(1, 1, N)
        k = torch.arange(N // 2 + 1, device=x.device).view(1, N // 2 + 1, 1)

        # Compute the DFT matrix (complex exponential for real input)
        W = torch.exp(-2j * math.pi * k * n / N)

        # Compute the real-valued DFT
        X_complex = torch.matmul(x, W)

        return X_complex.real

    elif x.ndim == 4:
        # 2D DFT case
        B, D, M, N = x.size()

        # Compute the DFT for rows (without redundant frequencies)
        m = torch.arange(M, device=x.device).view(1, 1, M, 1)
        k_m = torch.arange(M // 2 + 1, device=x.device).view(1, 1, 1, M // 2 + 1)
        W_m = torch.exp(-2j * math.pi * k_m * m / M)

        # Compute the DFT for columns (without redundant frequencies)
        n = torch.arange(N, device=x.device).view(1, 1, 1, N)
        k_n = torch.arange(N // 2 + 1, device=x.device).view(1, 1, N // 2 + 1, 1)
        W_n = torch.exp(-2j * math.pi * k_n * n / N)

        # Perform the row-wise and column-wise DFT
        X_complex_rows = torch.matmul(x, W_n)
        X_complex = torch.matmul(W_m, X_complex_rows)

        return X_complex.real

    elif x.ndim == 5:
        # 3D DFT case
        B, C, D, M, N = x.size()

        # Compute the DFT for depth (without redundant frequencies)
        d = torch.arange(D, device=x.device).view(1, 1, D, 1, 1)
        k_d = torch.arange(D // 2 + 1, device=x.device).view(1, 1, 1, D // 2 + 1, 1)
        W_d = torch.exp(-2j * math.pi * k_d * d / D)

        # Compute the DFT for rows and columns (without redundant frequencies)
        m = torch.arange(M, device=x.device).view(1, 1, 1, M, 1)
        k_m = torch.arange(M // 2 + 1, device=x.device).view(1, 1, 1, 1, M // 2 + 1)
        W_m = torch.exp(-2j * math.pi * k_m * m / M)

        n = torch.arange(N, device=x.device).view(1, 1, 1, 1, N)
        k_n = torch.arange(N // 2 + 1, device=x.device).view(1, 1, N // 2 + 1, 1, 1)
        W_n = torch.exp(-2j * math.pi * k_n * n / N)

        # Perform the DFT for depth, rows, and columns
        X_complex_depth = torch.matmul(W_d, x)
        X_complex_rows = torch.matmul(W_m, X_complex_depth)
        X_complex = torch.matmul(W_n, X_complex_rows)

        return X_complex.real

    else:
        raise ValueError(f"Input tensor must be 3D, 4D, or 5D, but got {x.ndim}D with shape {x.shape}.")


def idft(X: torch.Tensor) -> torch.Tensor:
    """
    Manually compute the Inverse Discrete Fourier Transform (IDFT) for real-valued input,
    handling real input without redundant information similar to irfftn.

    Args:
        X: Input tensor (3D for 1D IDFT, 4D for 2D IDFT, 5D for 3D IDFT).

    Returns:
        Real-valued tensor (same torch.dtype as input).
    """
    # Ensure X is in complex type
    X = X.to(torch.complex64) if X.dtype == torch.float32 else X.to(torch.complex128)

    if X.ndim == 3:
        # 1D IDFT case
        D, M, N = X.size()
        n = torch.arange(N, device=X.device).view(1, 1, N)
        k = torch.arange(N // 2 + 1, device=X.device).view(1, N // 2 + 1, 1)

        # Compute the IDFT matrix (inverse complex exponential for real input)
        W_inv = torch.exp(2j * math.pi * k * n / N)

        # Compute the inverse real-valued DFT
        x_complex = torch.matmul(X, W_inv)

        return (x_complex.real / N)

    elif X.ndim == 4:
        # 2D IDFT case
        B, D, M, N = X.size()

        # Compute the IDFT for rows (without redundant frequencies)
        m = torch.arange(M, device=X.device).view(1, 1, M, 1)
        k_m = torch.arange(M // 2 + 1, device=X.device).view(1, 1, 1, M // 2 + 1)
        W_m_inv = torch.exp(2j * math.pi * k_m * m / M)

        # Compute the IDFT for columns (without redundant frequencies)
        n = torch.arange(N, device=X.device).view(1, 1, 1, N)
        k_n = torch.arange(N // 2 + 1, device=X.device).view(1, 1, N // 2 + 1, 1)
        W_n_inv = torch.exp(2j * math.pi * k_n * n / N)

        # Perform the row-wise and column-wise IDFT
        x_complex_rows = torch.matmul(W_m_inv, X)
        x_complex = torch.matmul(x_complex_rows, W_n_inv)

        return (x_complex.real / (M * N))

    elif X.ndim == 5:
        # 3D IDFT case
        B, C, D, M, N = X.size()

        # Compute the IDFT for depth (without redundant frequencies)
        d = torch.arange(D, device=X.device).view(1, 1, D, 1, 1)
        k_d = torch.arange(D // 2 + 1, device=X.device).view(1, 1, 1, D // 2 + 1, 1)
        W_d_inv = torch.exp(2j * math.pi * k_d * d / D)

        # Compute the IDFT for rows and columns (without redundant frequencies)
        m = torch.arange(M, device=X.device).view(1, 1, 1, M, 1)
        k_m = torch.arange(M // 2 + 1, device=X.device).view(1, 1, 1, 1, M // 2 + 1)
        W_m_inv = torch.exp(2j * math.pi * k_m * m / M)

        n = torch.arange(N, device=X.device).view(1, 1, 1, 1, N)
        k_n = torch.arange(N // 2 + 1, device=X.device).view(1, 1, N // 2 + 1, 1, 1)
        W_n_inv = torch.exp(2j * math.pi * k_n * n / N)

        # Perform the IDFT for depth, rows, and columns
        x_complex_depth = torch.matmul(W_d_inv, X)
        x_complex_rows = torch.matmul(W_m_inv, x_complex_depth)
        x_complex = torch.matmul(W_n_inv, x_complex_rows)

        return (x_complex.real / (D * M * N))

    else:
        raise Value


def compl_mul1d(a, b):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    return torch.einsum("bix,iox->box", a, b)


def compl_mul2d(a, b):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    return torch.einsum("bixy,ioxy->boxy", a, b)


def compl_mul3d(a, b):
    return torch.einsum("bixyz,ioxyz->boxyz", a, b)

################################################################
# 1D Fourier Layer
################################################################

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier Layer: Performs DFT, multiplies relevant modes, 
        and transforms back via inverse DFT (IDFT).
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1

        # Scaling factor for the initialized weights
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute DFT coefficients
        x_dft = dft(x)

        # Allocate output in the frequency domain
        out_dft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)

        # Multiply relevant Fourier modes
        out_dft[:, :, :self.modes1] = compl_mul1d(x_dft[:, :, :self.modes1], self.weights1)

        # Return to physical space using inverse DFT
        x = idft(out_dft)
        return x


################################################################
# 2D Fourier Layer
################################################################

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier Layer: Performs DFT in 2D, multiplies relevant modes, 
        and transforms back via inverse DFT (IDFT).
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # Scaling factor for the initialized weights
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        size1 = x.shape[-2]
        size2 = x.shape[-1]

        # Compute DFT coefficients in 2D
        x_dft = dft(x)

        # Allocate output in the frequency domain
        out_dft = torch.zeros(batchsize, self.out_channels, size1, size2, device=x.device, dtype=torch.cfloat)

        # Multiply relevant Fourier modes
        out_dft[:, :, :self.modes1, :self.modes2] = compl_mul2d(x_dft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_dft[:, :, -self.modes1:, :self.modes2] = compl_mul2d(x_dft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space using inverse DFT
        x = idft(out_dft)
        return x


################################################################
# 3D Fourier Layer
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier Layer: Performs DFT in 3D, multiplies relevant modes,
        and transforms back via inverse DFT (IDFT).
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        # Scaling factor for the initialized weights
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]

        # Compute DFT coefficients in 3D
        x_dft = dft(x)

        # Allocate output in the frequency domain
        out_dft = torch.zeros(batchsize, self.out_channels, x.size(2), x.size(3), x.size(4)//2 + 1, device=x.device, dtype=torch.cfloat)

        # Multiply relevant Fourier modes
        out_dft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            compl_mul3d(x_dft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_dft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            compl_mul3d(x_dft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_dft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_dft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_dft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_dft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space using inverse DFT
        x = idft(out_dft)
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


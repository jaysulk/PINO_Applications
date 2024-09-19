import numpy as np

import torch
import torch.nn as nn

from functools import partial

import torch.nn.functional as F

import torch

def recursive_fht(x: torch.Tensor) -> torch.Tensor:
    """
    Recursive Fast Hartley Transform using butterfly equations.
    
    Parameters:
    x (torch.Tensor): Input tensor for the FHT (assumed to be 1D for each recursive step).
    
    Returns:
    torch.Tensor: Hartley transform of the input using recursive butterfly equations.
    """
    N = x.shape[-1]
    if N == 1:
        return x  # Base case for recursion, N = 1
    else:
        # Split the input into even and odd parts
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        
        # Recursively apply FHT to both parts
        FHT_even = recursive_fht(x_even)
        FHT_odd = recursive_fht(x_odd)
        
        # Butterfly combination using Hartley terms (cos + sin)
        n = torch.arange(N // 2, device=x.device).float()
        twiddle_factors = torch.cos(2 * torch.pi * n / N) + torch.sin(2 * torch.pi * n / N)
        
        # Combine the results
        combined_top = FHT_even + twiddle_factors * FHT_odd
        combined_bottom = FHT_even - twiddle_factors * FHT_odd
        
        return torch.cat([combined_top, combined_bottom], dim=-1)

def low_pass_filter(hartley_coeffs: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Apply a low-pass filter to the Hartley coefficients.
    Zero out or attenuate frequencies above the threshold.
    """
    assert 0 < threshold <= 1, "Threshold must be a value between 0 and 1."

    # Get the number of frequencies
    N = hartley_coeffs.shape[-1]
    freq_cutoff = int(N * threshold)

    # Zero out or attenuate high frequencies
    hartley_coeffs[..., freq_cutoff:] = 0.0
    return hartley_coeffs

def dht(x: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """
    Compute the Discrete Hartley Transform (DHT) with an optional low-pass filter,
    using the recursive Fast Hartley Transform.
    """
    if x.ndim == 3:
        # 1D case (input is a 3D tensor)
        D, M, N = x.size()
        N = M  # For 1D case, M and N should be the same size

        # Apply recursive FHT to each row
        X = torch.stack([recursive_fht(x[d]) for d in range(D)], dim=0)

        # Apply low-pass filter
        X = low_pass_filter(X, threshold)
        return X

    elif x.ndim == 4:
        # 2D case (input is a 4D tensor)
        B, D, M, N = x.size()

        # Apply recursive FHT along the last dimension (columns) first
        X = torch.stack([recursive_fht(x[b, d]) for b in range(B) for d in range(D)], dim=0).reshape(B, D, M, N)

        # Then apply recursive FHT along the second-last dimension (rows)
        X = torch.stack([recursive_fht(X[b, d].transpose(-1, -2)) for b in range(B) for d in range(D)], dim=0).reshape(B, D, M, N)

        # Apply low-pass filter
        X = low_pass_filter(X, threshold)
        return X

    elif x.ndim == 5:
        # 3D case (input is a 5D tensor)
        B, C, D, M, N = x.size()

        # Apply recursive FHT along the depth, row, and column dimensions
        X = torch.stack([recursive_fht(x[b, c, d]) for b in range(B) for c in range(C) for d in range(D)], dim=0).reshape(B, C, D, M, N)
        
        X = torch.stack([recursive_fht(X[b, c, d].transpose(-1, -2)) for b in range(B) for c in range(C) for d in range(D)], dim=0).reshape(B, C, D, M, N)
        
        X = torch.stack([recursive_fht(X[b, c, d].transpose(-2, -3)) for b in range(B) for c in range(C) for d in range(D)], dim=0).reshape(B, C, D, M, N)

        # Apply low-pass filter
        X = low_pass_filter(X, threshold)
        return X

    else:
        raise ValueError(f"Input tensor must be 3D, 4D, or 5D, but got {x.ndim}D with shape {x.shape}.")


def compl_mul1d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    # Compute the DHT of both signals
    X1_H_k = x1
    X2_H_k = x2
    X1_H_neg_k = x1.flip(0)
    X2_H_neg_k = x2.flip(0)
    N = x1.size(0)

    result = 0.5 * (torch.einsum('bix,iox->box', X1_H_k, X2_H_k) - 
                     torch.einsum('bix,iox->box', X1_H_neg_k, X2_H_neg_k) +
                     torch.einsum('bix,iox->box', X1_H_k, X2_H_neg_k) + 
                     torch.einsum('bix,iox->box', X1_H_neg_k, X2_H_k))

    return result

def compl_mul2d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    # Compute the DHT of both signals
    X1_H_k = x1
    X2_H_k = x2
    X1_H_neg_k = x1.flip(0)
    X2_H_neg_k = x2.flip(0)
    N = x1.size(0)

    result = 0.5 * (torch.einsum('bixy,ioxy->boxy', X1_H_k, X2_H_k) - 
                     torch.einsum('bixy,ioxy->boxy', X1_H_neg_k, X2_H_neg_k) +
                     torch.einsum('bixy,ioxy->boxy', X1_H_k, X2_H_neg_k) + 
                     torch.einsum('bixy,ioxy->boxy', X1_H_neg_k, X2_H_k))

    return result
    
def compl_mul3d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    # Compute the DHT of both signals
    X1_H_k = x1
    X2_H_k = x2
    X1_H_neg_k = x1.flip(0)
    X2_H_neg_k = x2.flip(0)
    N = x1.size(0)

    result = 0.5 * (torch.einsum('bixyz,ioxyz->boxyz', X1_H_k, X2_H_k) - 
                     torch.einsum('bixyz,ioxyz->boxyz', X1_H_neg_k, X2_H_neg_k) +
                     torch.einsum('bixyz,ioxyz->boxyz', X1_H_k, X2_H_neg_k) + 
                     torch.einsum('bixyz,ioxyz->boxyz', X1_H_neg_k, X2_H_k))

    return result

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


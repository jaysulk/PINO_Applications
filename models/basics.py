import numpy as np

import torch
import torch.nn as nn

from functools import partial

import torch.nn.functional as F

import torch

import torch

def recursive_butterfly(x: torch.Tensor) -> torch.Tensor:
    """
    Recursive Butterfly calculation for the Regularized Fast Hartley Transform (RFHT).
    
    This function works on the last dimension of the tensor, handling odd-sized inputs.
    
    Parameters:
    x (torch.Tensor): The input tensor.
    
    Returns:
    torch.Tensor: The transformed tensor with the same size as input.
    """
    N = x.shape[-1]
    
    # Base case: If size is 1, return the input
    if N == 1:
        return x
    
    # Split input into even and odd indexed parts
    even_part = x[..., 0:N:2]  # Elements at even indices
    odd_part = x[..., 1:N:2]   # Elements at odd indices
    
    # Recursive butterfly transform on both parts
    transformed_even = recursive_butterfly(even_part)
    transformed_odd = recursive_butterfly(odd_part)
    
    # Create butterfly operations for merging
    n = torch.arange(N // 2, device=x.device).float()
    factor = 2 * torch.pi * n / N
    
    cos_term = torch.cos(factor)
    sin_term = torch.sin(factor)
    
    # Combine the even and odd parts using butterfly operation
    combined_part1 = transformed_even + cos_term * transformed_odd + sin_term * transformed_odd
    combined_part2 = transformed_even - cos_term * transformed_odd - sin_term * transformed_odd
    
    # Handle odd-sized input: last element needs special handling
    if N % 2 == 1:
        last_element = x[..., -1:]
        last_transformed = torch.cat((combined_part1, last_element), dim=-1)
        return last_transformed
    else:
        return torch.cat((combined_part1, combined_part2), dim=-1)


def low_pass_filter(hartley_coeffs: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Apply a low-pass filter to the Hartley coefficients.
    Zero out or attenuate frequencies above the threshold.
    
    Parameters:
    hartley_coeffs (torch.Tensor): Hartley coefficients after RFHT.
    threshold (float): Fraction of frequencies to keep (0 < threshold <= 1).
    
    Returns:
    torch.Tensor: Filtered Hartley coefficients.
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
    Compute the Regularized Fast Hartley Transform (RFHT) with an optional low-pass filter.
    Uses recursive butterfly equations for efficiency and odd-sized input support.

    Parameters:
    x (torch.Tensor): Input tensor (3D, 4D, or 5D).
    threshold (float): Fraction of frequencies to keep after RFHT (0 < threshold <= 1).

    Returns:
    torch.Tensor: RFHT of the input tensor with optional low-pass filtering.
    """
    
    if x.ndim == 3:
        # 1D case (input is a 3D tensor)
        D, M, N = x.size()

        # Apply recursive butterfly transform along the last dimension
        X = recursive_butterfly(x)

        # Apply low-pass filter
        X = low_pass_filter(X, threshold)
        return X

    elif x.ndim == 4:
        # 2D case (input is a 4D tensor)
        B, D, M, N = x.size()

        # Apply recursive butterfly transform to both dimensions
        X = recursive_butterfly(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)  # First along N
        X = recursive_butterfly(X)                                           # Then along M

        # Apply low-pass filter
        X = low_pass_filter(X, threshold)
        return X

    elif x.ndim == 5:
        # 3D case (input is a 5D tensor)
        B, C, D, M, N = x.size()

        # Apply recursive butterfly transform to each dimension
        X = recursive_butterfly(x.permute(0, 1, 4, 3, 2)).permute(0, 1, 4, 3, 2)  # First along N
        X = recursive_butterfly(X.permute(0, 1, 3, 2, 4)).permute(0, 1, 3, 2, 4)  # Then along M
        X = recursive_butterfly(X)                                                 # Finally along D

        # Apply low-pass filter
        X = low_pass_filter(X, threshold)
        return X

    else:
        raise ValueError(f"Input tensor must be 3D, 4D, or 5D, but got {x.ndim}D with shape {x.shape}.")

def idht(x: torch.Tensor) -> torch.Tensor:
    # Compute the DHT
    transformed = dht(x)
    
    # Determine normalization factor
    if x.ndim == 3:
        # 1D case (3D tensor input)
        N = x.size(1)  # N is the size of the last dimension
        normalization_factor = N
    elif x.ndim == 4:
        # 2D case (4D tensor input)
        M, N = x.size(2), x.size(3)
        normalization_factor = M * N
    elif x.ndim == 5:
        # 3D case (5D tensor input)
        D, M, N = x.size(2), x.size(3), x.size(4)
        normalization_factor = D * M * N
    else:
        raise ValueError(f"Input tensor must be 3D, 4D, or 5D, but got {x.ndim}D with shape {x.shape}.")

def compl_mul1d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    # Compute the DHT of both signals
    X1_H_k = x1
    X2_H_k = x2
    X1_H_neg_k = torch.roll(torch.flip(x1, dims=[-1]), shifts=1, dims=[-1])
    X2_H_neg_k = torch.roll(torch.flip(x2, dims=[-1]), shifts=1, dims=[-1])

    result = 0.5 * (torch.einsum('bix,iox->box', X1_H_k, X2_H_k) - 
                     torch.einsum('bix,iox->box', X1_H_neg_k, X2_H_neg_k) +
                     torch.einsum('bix,iox->box', X1_H_k, X2_H_neg_k) + 
                     torch.einsum('bix,iox->box', X1_H_neg_k, X2_H_k))

    return result

def compl_mul2d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    # Compute the DHT of both signals
    X1_H_k = x1
    X2_H_k = x2
    X1_H_neg_k = torch.roll(torch.flip(x1, dims=[-1, -2]), shifts=(1, 1), dims=[-1, -2])
    X2_H_neg_k = torch.roll(torch.flip(x2, dims=[-1, -2]), shifts=(1, 1), dims=[-1, -2])
    
    # Perform the convolution using DHT components
    result = 0.5 * (torch.einsum('bixy,ioxy->boxy', X1_H_k, X2_H_k) - 
                    torch.einsum('bixy,ioxy->boxy', X1_H_neg_k, X2_H_neg_k) +
                    torch.einsum('bixy,ioxy->boxy', X1_H_k, X2_H_neg_k) + 
                    torch.einsum('bixy,ioxy->boxy', X1_H_neg_k, X2_H_k))
    
    return result

    
def compl_mul3d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    # Compute the DHT of both signals
    X1_H_k = x1
    X2_H_k = x2
    X1_H_neg_k = torch.roll(torch.flip(x1, dims=[-3, -2, -1]), shifts=(1, 1, 1), dims=[-3, -2, -1])
    X2_H_neg_k = torch.roll(torch.flip(x2, dims=[-3, -2, -1]), shifts=(1, 1, 1), dims=[-3, -2, -1])

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


import numpy as np
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

def dht(x: torch.Tensor, dims=None) -> torch.Tensor:
    if dims is None:
        dims = [2] if x.ndim == 3 else [2, 3] if x.ndim == 4 else [2, 3, 4]
    
    if x.ndim == 3:
        # 1D case (3D tensor)
        D, M, N = x.size()
        n = torch.arange(N, device=x.device).float()

        # Hartley kernel for 1D
        cas = torch.cos(2 * torch.pi * n.view(-1, 1) * n / N) + torch.sin(2 * torch.pi * n.view(-1, 1) * n / N)

        # Perform the DHT
        X = torch.matmul(cas, x.view(D, N, M).permute(1, 0, 2).reshape(N, -1))
        return X.reshape(N, D, M).permute(1, 2, 0)

    elif x.ndim == 4:
        # 2D case (4D tensor)
        B, D, M, N = x.size()
        m = torch.arange(M, device=x.device).float()
        n = torch.arange(N, device=x.device).float()

        # Hartley kernels for rows and columns
        cas_row = torch.cos(2 * torch.pi * m.view(-1, 1) * m / M) + torch.sin(2 * torch.pi * m.view(-1, 1) * m / M)
        cas_col = torch.cos(2 * torch.pi * n.view(-1, 1) * n / N) + torch.sin(2 * torch.pi * n.view(-1, 1) * n / N)

        # Perform the DHT
        x_reshaped = x.reshape(B * D, M, N)

        # Apply the column transform
        intermediate = torch.matmul(x_reshaped, cas_col.T)
        
        # Apply the row transform
        X = torch.matmul(cas_row.T, intermediate)

        # Reshape to original size
        return X.reshape(B, D, M, N)

    elif x.ndim == 5:
        # 3D case (5D tensor)
        B, C, D, M, N = x.size()
        d = torch.arange(D, device=x.device).float()
        m = torch.arange(M, device=x.device).float()
        n = torch.arange(N, device=x.device).float()

        # Hartley kernels for depth, rows, and columns
        cas_depth = torch.cos(2 * torch.pi * d.view(-1, 1) * d / D) + torch.sin(2 * torch.pi * d.view(-1, 1) * d / D)
        cas_row = torch.cos(2 * torch.pi * m.view(-1, 1) * m / M) + torch.sin(2 * torch.pi * m.view(-1, 1) * m / M)
        cas_col = torch.cos(2 * torch.pi * n.view(-1, 1) * n / N) + torch.sin(2 * torch.pi * n.view(-1, 1) * n / N)

        # Perform the DHT
        x_reshaped = x.reshape(B * C, D, M, N)

        # Apply depth, row, and column transforms
        intermediate = torch.matmul(x_reshaped, cas_col.T)
        intermediate = torch.matmul(cas_row.T, intermediate)
        X = torch.matmul(cas_depth.T, intermediate)

        # Reshape to original size
        return X.reshape(B, C, D, M, N)

    else:
        raise ValueError(f"Input tensor must be 3D, 4D, or 5D, but got {x.ndim}D with shape {x.shape}.")

def idht(x: torch.Tensor) -> torch.Tensor:
    # Compute the DHT (Direct Hartley Transform)
    transformed = dht(x)
    
    # Determine normalization factor
    if x.ndim == 3:
        # 1D case (3D tensor input)
        N = x.size(2)  # N is the size of the last dimension
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

    # Normalize the result to undo the scaling effect of the DHT
    return transformed / normalization_factor

def compl_mul1d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#    X1_H_k = x1
#    X2_H_k = x2
#    X1_H_neg_k = torch.roll(torch.flip(x1, dims=[-1]), shifts=1, dims=[-1])
#    X2_H_neg_k = torch.roll(torch.flip(x2, dims=[-1]), shifts=1, dims=[-1])

#    result = 0.5 * (torch.einsum('bix,iox->box', X1_H_k, X2_H_k) -
#                    torch.einsum('bix,iox->box', X1_H_neg_k, X2_H_neg_k) +
#                    torch.einsum('bix,iox->box', X1_H_k, X2_H_neg_k) +
#                    torch.einsum('bix,iox->box', X1_H_neg_k, X2_H_k))

#    return result
    return torch.einsum("bi...,io...->bo...", x1, x2)

def compl_mul2d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#    X1_H_k = x1
#    X2_H_k = x2
#    X1_H_neg_k = torch.roll(torch.flip(x1, dims=[-1, -2]), shifts=(1, 1), dims=[-1, -2])
#    X2_H_neg_k = torch.roll(torch.flip(x2, dims=[-1, -2]), shifts=(1, 1), dims=[-1, -2])

#    result = 0.5 * (torch.einsum('bixy,ioxy->boxy', X1_H_k, X2_H_k) -
#                    torch.einsum('bixy,ioxy->boxy', X1_H_neg_k, X2_H_neg_k) +
#                    torch.einsum('bixy,ioxy->boxy', X1_H_k, X2_H_neg_k) +
#                    torch.einsum('bixy,ioxy->boxy', X1_H_neg_k, X2_H_k))

#    return result
    return torch.einsum("bixy...,ioxy...->boxy...", x1, x2)

def compl_mul3d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#    X1_H_k = x1
#    X2_H_k = x2
#    X1_H_neg_k = torch.roll(torch.flip(x1, dims=[-3, -2, -1]), shifts=(1, 1, 1), dims=[-3, -2, -1])
#    X2_H_neg_k = torch.roll(torch.flip(x2, dims=[-3, -2, -1]), shifts=(1, 1, 1), dims=[-3, -2, -1])

#    result = 0.5 * (torch.einsum('bixyz,ioxyz->boxyz', X1_H_k, X2_H_k) -
#                    torch.einsum('bixyz,ioxyz->boxyz', X1_H_neg_k, X2_H_neg_k) +
#                    torch.einsum('bixyz,ioxyz->boxyz', X1_H_k, X2_H_neg_k) +
#                    torch.einsum('bixyz,ioxyz->boxyz', X1_H_neg_k, X2_H_k))

#    return result
    return torch.einsum("bixyz...,ioxyz...->boxyz...", x1, x2)

################################################################
# Low-Pass Filter Function
################################################################

#def low_pass_filter(x_ht, cutoff):
#    """
#    Applies a low-pass filter to the spectral coefficients (DHT output).
#    Frequencies higher than `cutoff` are dampened.
#    """
#    size = x_ht.shape[-1]  # Get the last dimension (frequency axis)
#    frequencies = torch.fft.fftfreq(size, d=1.0)  # Compute frequency bins
#    filter_mask = torch.abs(frequencies) <= cutoff  # Mask for low frequencies
#    return x_ht * filter_mask.to(x_ht.device)

################################################################
# Gaussian Smoothing Function
################################################################

#def gaussian_smoothing(x, kernel_size=5, sigma=1.0):
#    """
#    Applies Gaussian smoothing to the output.
#    """
#    # Apply Gaussian blur (use 2D or 3D kernel as needed)
#    return F.gaussian_blur(x, kernel_size=[kernel_size], sigma=[sigma])

################################################################
# Data Augmentation Function
################################################################

#def augment_data(inputs, shift_range=0.1, scale_range=0.05):
#    """
#    Augment input data by applying random shifts and scaling.
#    
#    Parameters:
#    - inputs: torch.Tensor, the input data to be augmented
#    - shift_range: float, the maximum range for random shifts
#    - scale_range: float, the maximum range for random scaling
#    
#    Returns:
#    - augmented_inputs: torch.Tensor, the augmented input data
#    """
#    # Apply random shifts
#    shifts = torch.rand(inputs.size()) * shift_range
#    augmented_inputs = inputs + shifts
#    
#    # Apply random scaling
#    scales = 1 + torch.rand(inputs.size()) * scale_range
#    augmented_inputs = augmented_inputs * scales
#    
#    return augmented_inputs

################################################################
# 1D Hartley convolution layer
################################################################

################################################################
# 1D Hartley convolution layer with phase reconstruction
################################################################

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1))

    def forward(self, x):
        batchsize = x.shape[0]
        
        x_ht = dht(x)

        out_ht = torch.zeros(batchsize, self.in_channels, x.size(-1)//2 + 1, device=x.device)
        out_ht[:, :, :self.modes1] = compl_mul1d(x_ht[:, :, :self.modes1], self.weights1)

        x = idht(out_ht)

        # Cosine part is the real part
        cos_part = x
        
        # Sine part is the flipped tensor
        sin_part = torch.flip(x, dims=[-1])

        # Create complex tensor and reconstruct phase
        complex_tensor = torch.complex(cos_part, sin_part)
        phase_reconstructed = torch.angle(complex_tensor)

        return phase_reconstructed

################################################################
# 2D Hartley convolution layer with phase reconstruction
################################################################

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))

    def forward(self, x):
        batchsize = x.shape[0]
        size1 = x.shape[-2]
        size2 = x.shape[-1]
        
        x_dht = dht(x)

        out_dht = torch.zeros(batchsize, self.out_channels, size1, size2, device=x.device)
        out_dht[:, :, :self.modes1, :self.modes2] = compl_mul2d(x_dht[:, :, :self.modes1, :self.modes2], self.weights1)
        out_dht[:, :, -self.modes1:, :self.modes2] = compl_mul2d(x_dht[:, :, -self.modes1:, :self.modes2], self.weights2)

        x = idht(out_dht)

        # Cosine part is the real part
        cos_part = x
        
        # Sine part is the flipped tensor along both dimensions
        sin_part = torch.flip(x, dims=[-2, -1])

        # Create complex tensor and reconstruct phase
        complex_tensor = torch.complex(cos_part, sin_part)
        phase_reconstructed = torch.angle(complex_tensor)

        return phase_reconstructed

################################################################
# 3D Hartley convolution layer with phase reconstruction
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))

    def forward(self, x):
        batchsize = x.shape[0]
        
        x_ht = dht(x)

        out_ht = torch.zeros(batchsize, self.out_channels, x.size(2), x.size(3), x.size(4)//2 + 1, device=x.device)
        out_ht[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            compl_mul3d(x_ht[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ht[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            compl_mul3d(x_ht[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ht[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_ht[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ht[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_ht[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        x = idht(out_ht)

        # Cosine part is the real part
        cos_part = x
        
        # Sine part is the flipped tensor along all three dimensions
        sin_part = torch.flip(x, dims=[-3, -2, -1])

        # Create complex tensor and reconstruct phase
        complex_tensor = torch.complex(cos_part, sin_part)
        phase_reconstructed = torch.angle(complex_tensor)

        return phase_reconstructed

################################################################
# FourierBlock (Using SpectralConv3d)
################################################################

class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, activation='tanh'):
        super(FourierBlock, self).__init__()
        
        # Spectral convolution layer (using 3D Hartley transform)
        self.speconv = SpectralConv3d(in_channels, out_channels, modes1, modes2, modes3)
        
        # Linear layer applied across the channel dimension
        self.linear = nn.Conv1d(in_channels, out_channels, 1)
        
        # Activation function selection
        if activation == 'tanh':
            self.activation = nn.Tanh()  # Use nn.Tanh() for module (not in-place operation)
        elif activation == 'gelu':
            self.activation = nn.GELU()  # Apply GELU non-linearity
        elif activation == 'none':
            self.activation = None  # No activation
        else:
            raise ValueError(f'{activation} is not supported')

    def forward(self, x):
        '''
        Input x: (batchsize, in_channels, x_grid, y_grid, t_grid)
        '''
        # Apply spectral convolution (3D Hartley convolution)
        x1 = self.speconv(x)
        
        # Apply 1D convolution across the channel dimension
        # Flattening the last three dimensions into one while keeping the batch and channel
        x2 = self.linear(x.view(x.shape[0], self.in_channel, -1))
        
        # Reshape x2 back to match the original spatial and temporal grid structure
        x2 = x2.view(x.shape[0], self.out_channel, x.shape[2], x.shape[3], x.shape[4])
        
        # Combine spectral and linear outputs (skip connection)
        out = x1 + x2
        
        # Apply activation function (non-linearity)
        if self.activation is not None:
            out = self.activation(out)
        
        return out




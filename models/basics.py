import numpy as np
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

def hilbert_transform(x: torch.Tensor) -> torch.Tensor:
    """Compute the Hilbert transform to recover imaginary components."""
    X_rfft = torch.fft.rfftn(x, dim=tuple(range(1, x.ndim)))
    h = torch.zeros_like(X_rfft)
    N = X_rfft.shape[-1]
    if N % 2 == 0:  # if even, handle Nyquist frequency
        h[..., 1:N//2] = 2
        h[..., N//2] = 1
    else:
        h[..., 1:(N + 1)//2] = 2
    return torch.fft.irfftn(h * X_rfft, s=x.shape[1:])

def dht(x: torch.Tensor) -> torch.Tensor:
    # Perform rfftn on input of any dimensionality
    X_rfft = torch.fft.rfftn(x, dim=tuple(range(1, x.ndim)))
    
    # Mirror the Fourier components and exclude the first column if the input size is even
    mirrored_part = torch.flip(X_rfft, dims=[i + 1 for i in range(x.ndim - 1)]).conj()
    
    # Concatenate the original rfft result with the mirrored part
    X_rfft_full = torch.cat([X_rfft, mirrored_part[..., 1:]], dim=-1)
    
    # Hartley transform computation
    X = torch.real(X_rfft_full) - torch.imag(X_rfft_full)
    
    return X

def idht(X: torch.Tensor) -> torch.Tensor:
    n = X.numel()  # Total number of elements in the tensor
    X = dht(X)  # Apply DHT
    x = X / n  # Normalize by the total number of elements
    return x

def compl_mul1d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
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
    X1_H_k = x1
    X2_H_k = x2
    X1_H_neg_k = torch.roll(torch.flip(x1, dims=[-1, -2]), shifts=(1, 1), dims=[-1, -2])
    X2_H_neg_k = torch.roll(torch.flip(x2, dims=[-1, -2]), shifts=(1, 1), dims=[-1, -2])
    
    result = 0.5 * (torch.einsum('bixy,ioxy->boxy', X1_H_k, X2_H_k) - 
                    torch.einsum('bixy,ioxy->boxy', X1_H_neg_k, X2_H_neg_k) +
                    torch.einsum('bixy,ioxy->boxy', X1_H_k, X2_H_neg_k) + 
                    torch.einsum('bixy,ioxy->boxy', X1_H_neg_k, X2_H_k))
    
    return result

def compl_mul3d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
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
# 1D Spectral Convolution Layer with DHT and Hilbert Recovery
################################################################

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, 2)
        )

    def forward(self, x):
        batchsize = x.shape[0]

        # Apply Hartley and Hilbert transforms to recover full complex info
        x_ht = dht(x)
        x_hilbert = hilbert_transform(x)
        
        # Combine real and imaginary parts to mimic Fourier
        x_combined = x_ht + 1j * x_hilbert

        # Perform convolution in spectral (Hartley) space
        out_ht = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ht[:, :, :self.modes1] = compl_mul1d(x_combined[:, :, :self.modes1], self.weights1)

        # Apply inverse DHT to recover the output in the time domain
        x = idht(out_ht.real)  # Only the real part is used for the final result
        return x

################################################################
# 2D Spectral Convolution Layer with DHT and Hilbert Recovery
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
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2)
        )

    def forward(self, x):
        batchsize = x.shape[0]
        size1 = x.shape[-2]
        size2 = x.shape[-1]

        # Apply Hartley and Hilbert transforms for full complex recovery
        x_ht = dht(x)
        x_hilbert = hilbert_transform(x)
        
        # Combine real and imaginary parts
        x_combined = x_ht + 1j * x_hilbert

        # Perform convolution in the Hartley space
        out_ht = torch.zeros(batchsize, self.out_channels, size1, size2, device=x.device, dtype=torch.cfloat)
        out_ht[:, :, :self.modes1, :self.modes2] = compl_mul2d(x_combined[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ht[:, :, -self.modes1:, :self.modes2] = compl_mul2d(x_combined[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Apply inverse DHT to recover the output in the time domain
        x = idht(out_ht.real)  # Final result is real part
        return x

################################################################
# 3D Spectral Convolution Layer with DHT and Hilbert Recovery
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
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]

        # Apply Hartley and Hilbert transforms for full complex recovery
        x_ht = dht(x)
        x_hilbert = hilbert_transform(x)
        
        # Combine real and imaginary parts
        x_combined = x_ht + 1j * x_hilbert

        # Perform convolution in Hartley space
        out_ht = torch.zeros(batchsize, self.out_channels, x.size(2), x.size(3), x.size(4) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ht[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            compl_mul3d(x_combined[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ht[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            compl_mul3d(x_combined[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ht[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_combined[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)

        # Apply inverse DHT to recover the output in the time domain
        x = idht(out_ht.real)  # Only the real part is used for the final result
        return x


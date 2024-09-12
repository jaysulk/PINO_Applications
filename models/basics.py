import numpy as np
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F


def dht_rfft(x: torch.Tensor, dim: torch.int32)-> torch.Tensor:
    # Compute the real FFT of the input tensor over the specified dimensions
    X_rfft = torch.fft.rfftn(x, dim=dim)
    
    # Compute the real and imaginary parts
    real_part = X_rfft.real
    imag_part = X_rfft.imag
    
    # DHT is the sum of the real part and the negative of the imaginary part
    dht_result = real_part - imag_part
    
    # Now we need to pad the result along the truncated dimensions to match the input size
    original_shape = list(x.shape)
    
    # For each dimension in dims, pad the result tensor to match the original shape
    for d in dim:
        size_diff = original_shape[d] - dht_result.shape[d]
        if size_diff > 0:
            # Padding is applied at the end of the dimension (not both sides)
            pad = [0] * (2 * len(dht_result.shape))  # Initialize no padding
            pad[2 * (d + 1) - 1] = size_diff  # Set padding at the end of the dimension
            dht_result = torch.nn.functional.pad(dht_result, pad)
    
    return dht_result

def dht(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:  # For 1D DHT
        return dht_rfft(x, dim=[2])
    elif x.ndim == 4:  # For 2D DHT
        return dht_rfft(x, dim=[2, 3])
    elif x.ndim == 5:  # For 3D DHT
        return dht_rfft(x, dim=[2, 3, 4])
    else:
        raise ValueError("Only 1D (3D tensors), 2D (4D tensors), and 3D (5D tensors) tensors are supported.")

def idht(X: torch.Tensor) -> torch.Tensor:
    n = X.numel()  # Total number of elements in the 2D tensor
    X = dht(X)  # Apply DHT
    x = X / n  # Normalize by the total number of elements
    return x

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

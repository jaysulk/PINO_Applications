import numpy as np

import torch
import torch.nn as nn

from functools import partial

import torch
import torch.nn.functional as F

import torch

import torch

def dht(x: torch.Tensor):
    X = torch.fft.fft(x)
    X = X.real - X.imag
    return X

def idht(X: torch.Tensor):
    dims = X.size()
    n = torch.prod(torch.tensor(dims)).item()
    X = dht(X)
    x = X / n
    return x

#def flip_periodic(x: torch.Tensor):
#    flipped_x = torch.cat((x[..., 0:1], torch.flip(x[..., 1:], dims=[-1])), dim=-1)
#    flipped_x = torch.cat((flipped_x[..., 0:1, :], torch.flip(flipped_x[..., 1:, :], dims=[-2])), dim=-2)
#    return flipped_x

def compl_mul1d(x, y):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    #return torch.einsum("bix,iox->box", a, b)
    X = dht(x)
    Y = dht(y)
    Xflip = torch.roll(torch.flip(x, [0]), 1, dims=0)
    Yflip = torch.roll(torch.flip(y, [0]), 1, dims=0)
    Yplus = 0.5*(Y + Yflip)
    Yminus = 0.5*(Y - Yflip)
    Z = torch.einsum("bix,iox->box", X, Yplus) + torch.einsum("bix,iox->box", Xflip, Yminus)
    z = idht(Z)
    
    return z

def compl_mul2d(x, y):
    """ Multiplies tensors a and b using the convolution theorem for the DHT.
    Assumes hartley_transform and inverse_hartley_transform are defined.
    """
    X = dht(x)
    Y = dht(y)
    Xflip = torch.roll(torch.flip(x, [0, 1]), shifts=(1, 1), dims=(0, 1))
    Yflip = torch.roll(torch.flip(y, [0, 1]), shifts=(1, 1), dims=(0, 1))

    Yplus = 0.5*(Y + Yflip)
    Yminus = 0.5*(Y - Yflip)
    Z = torch.einsum("bixy,ioxy->boxy", x, Yplus) + torch.einsum("bixy,ioxy->boxy",  Xflip, Yminus)
    z = idht(Z)
    
    return z

def compl_mul3d(x, y): 
    """ Multiplies tensors a and b using the convolution theorem for the DHT.
    Assumes hartley_transform and inverse_hartley_transform are defined.
    """
    X = dht(x)
    Y = dht(y)
    Xflip = torch.roll(torch.flip(x, [0, 1, 2]), shifts=(1, 1, 1), dims=(0, 1, 2))
    Yflip = torch.roll(torch.flip(y, [0, 1, 2]), shifts=(1, 1, 1), dims=(0, 1, 2))

    Yplus = 0.5*(Y + Yflip)
    Yminus = 0.5*(Y - Yflip)
    Z = torch.einsum("bixyz,ioxyz->boxyz", x, Yplus) + torch.einsum("bixyz,ioxyz->boxyz",  Xflip, Yminus)
    z = idht(Z)
    
    return z

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


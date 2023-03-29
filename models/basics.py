import numpy as np

import torch
import torch.nn as nn

from functools import partial

def DiscreteHartleyTransform(X:torch.Tensor,s,dim):
	fft = torch.fft.fftn(X, s=s, dim=dim, norm="forward")
	return torch.real(fft) - torch.imag(fft)

def InverseDiscreteHartleyTransform(X:torch.Tensor, s, dim):
    return (1.0/len(X))*DiscreteHartleyTransform(X, s=s, dim=dim)

def flip_periodic(x:torch.Tensor, dims):
    if dims == 1:
        x = torch.roll(torch.flip(x,dims=[0]),1)
    elif dims == 2:
        x = torch.roll(torch.flip(x,dims=[0,1]),1)
    return x

def compl_mul1d(a, b): 
    aflip = flip_periodic(a,1)
    bflip = flip_periodic(b,1)
    beven = 0.5 * (b + bflip)
    bodd  = 0.5 * (b - bflip)
    return torch.einsum("bix,iox->box", a, beven) +  torch.einsum("bix,iox->box", aflip, bodd)    

def compl_mul2d(a, b):
    aflip = flip_periodic(a,2)
    bflip = flip_periodic(b,2)
    beven = 0.5 * (b + bflip)
    bodd  = 0.5 * (b - bflip)  
    return torch.einsum("bixy,ioxy->boxy", a, beven) + torch.einsum("bixy,ioxy->boxy", aflip, bodd)

def compl_mul3d(a, b):
    aflip = flip_periodic(a,2)
    bflip = flip_periodic(b,2)
    beven = 0.5 * (b + bflip)
    bodd  = 0.5 * (b - bflip)
    return torch.einsum("bixyz,ioxyz->boxyz", a, beven) + torch.einsum("bixyz,ioxyz->boxyz", aflip, bodd)

################################################################
# 1d fourier layer
################################################################


import torch
import torch.nn as nn
import torch_fft

class SpectralConv1dDHT(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1dDHT, self).__init__()

        """
        1D Hartley layer. It does DHT, linear transform, and Inverse DHT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Hartley modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(out_channels, in_channels, self.modes1))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Hartley coefficients
        x_ht = torch_fft.dht(x, dim=2)

        # Multiply relevant Hartley modes and sum along the input channels
        out_ht = torch.zeros(batchsize, self.out_channels, x.size(-1), device=x.device)
        for i in range(self.in_channels):
            out_ht[:, :, :self.modes1] += x_ht[:, i, :self.modes1] * self.weights1[:, i, :self.modes1]

        # Return to physical space
        x = torch_fft.idht(out_ht, dim=2)
        return x


################################################################
# 2d fourier layer
################################################################


import torch
import torch.nn as nn
import torch_fft

class SpectralConv2dDHT(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2dDHT, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Hartley modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(
            self.scale * torch.rand(out_channels, in_channels, self.modes1, self.modes2, dtype=torch.float))

    def forward(self, x):
        batchsize = x.shape[0]

        # Compute Hartley coefficients
        x_ht = torch_fft.dht2(x, dim=(2, 3))

        # Multiply relevant Hartley modes and sum along the input channels
        out_ht = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1), device=x.device)
        for i in range(self.in_channels):
            out_ht[:, :, :self.modes1, :self.modes2] += x_ht[:, i, :self.modes1, :self.modes2] * self.weights[:, i, :self.modes1, :self.modes2]

        # Return to physical space
        x = torch_fft.idht2(out_ht, dim=(2, 3))
        return x



class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2,3,4])
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(2), x.size(3), x.size(4)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(2), x.size(3), x.size(4)), dim=[2,3,4])
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

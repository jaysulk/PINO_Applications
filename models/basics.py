import numpy as np

import torch
import torch.nn as nn

from functools import partial

import torch

def compl_mul1d(a, b):
    """
    Perform cyclic convolution for 1D Hartley-transformed inputs a and b.
    """
    # Pad inputs for cyclic convolution
    a_padded = torch.cat((a, torch.zeros_like(a)), dim=-1)
    b_padded = torch.cat((b, torch.zeros_like(b)), dim=-1)
    
    # Compute convolution in the frequency domain
    return torch.irfft(torch.fft(a_padded, 1) * torch.fft(b_padded, 1), 1)[:, :, :a.size(-1)]

def compl_mul2d(a, b):
    """
    Perform cyclic convolution for 2D Hartley-transformed inputs a and b.
    """
    # Pad inputs for cyclic convolution
    a_padded = torch.cat((a, torch.zeros_like(a)), dim=-1)
    b_padded = torch.cat((b, torch.zeros_like(b)), dim=-1)
    
    # Compute convolution in the frequency domain
    return torch.irfft(torch.fft(a_padded, 2) * torch.fft(b_padded, 2), 2)[:, :, :a.size(-2), :a.size(-1)]

def compl_mul3d(a, b):
    """
    Perform cyclic convolution for 3D Hartley-transformed inputs a and b.
    """
    # Pad inputs for cyclic convolution
    a_padded = torch.cat((a, torch.zeros_like(a)), dim=-1)
    b_padded = torch.cat((b, torch.zeros_like(b)), dim=-1)
    
    # Compute convolution in the frequency domain
    return torch.irfft(torch.fft(a_padded, 3) * torch.fft(b_padded, 3), 3)[:, :, :a.size(-3), :a.size(-2), :a.size(-1)]


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
            self.scale * torch.rand(in_channels, out_channels, self.modes1))

    def forward(self, x):
        batchsize = x.shape[0]
        x = x.detach().cpu().numpy()

        # Compute Hartley coeffcients
        x_ht = np.fft.hfft(x, axis=2)

        # Convert numpy array to torch tensor and multiply relevant Hartley modes
        x_ht = torch.from_numpy(x_ht).to(x.device)
        out_ht = torch.zeros(batchsize, self.in_channels, x.shape[-1]//2 + 1, device=x.device)
        out_ht[:, :, :self.modes1] = x_ht[:, :, :self.modes1] * self.weights1

        # Convert torch tensor to numpy array for inverse Hartley transform
        out_ht = out_ht.detach().cpu().numpy()
        
        # Return to physical space
        x = np.fft.ihfft(out_ht, axis=2)

        return torch.from_numpy(x).to(out_ht.device)


################################################################
# 2d fourier layer
################################################################

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Hartley layer. It does HHT, linear transform, and Inverse HHT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Hartley modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))

    def forward(self, x):
        batchsize = x.shape[0]
        x = x.detach().cpu().numpy()

        # Compute Hartley coefficients
        x_ht = np.fft.hfft(x, axes=[2, 3])

        # Convert numpy array to torch tensor and multiply relevant Hartley modes
        x_ht = torch.from_numpy(x_ht).to(x.device)
        out_ht = torch.zeros(batchsize, self.out_channels, x.shape[-2], x.shape[-1]//2 + 1, device=x.device)
        out_ht[:, :, :self.modes1, :self.modes2] = x_ht[:, :, :self.modes1, :self.modes2] * self.weights1

        # Convert torch tensor to numpy array for inverse Hartley transform
        out_ht = out_ht.detach().cpu().numpy()

        # Return to physical space
        x = np.fft.ihfft(out_ht, axes=[2, 3])

        return torch.from_numpy(x).to(out_ht.device)

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Hartley layer. It does HHT, linear transform, and Inverse HHT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Hartley modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        
    def forward(self, x):
        batchsize = x.shape[0]
        x = x.detach().cpu().numpy()

        # Compute Hartley coeffcients
        x_ht = np.fft.hfft(x, axes=[2, 3, 4])

        # Convert numpy array to torch tensor and multiply relevant Hartley modes
        x_ht = torch.from_numpy(x_ht).to(x.device)
        out_ht = torch.zeros(batchsize, self.out_channels, x.shape[2], x.shape[3], x.shape[4]//2 + 1, device=x.device)
        out_ht[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            x_ht[:, :, :self.modes1, :self.modes2, :self.modes3] * self.weights1

        # Convert torch tensor to numpy array for inverse Hartley transform
        out_ht = out_ht.detach().cpu().numpy()

        # Return to physical space
        x = np.fft.ihfft(out_ht, axes=[2, 3, 4])

        return torch.from_numpy(x).to(out_ht.device)


import torch
import torch.nn as nn

class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, activation='tanh'):
        super(FourierBlock, self).__init__()

        """
        Fourier Block is a transformation block consisting of spectral convolution, a linear transformation,
        and a non-linear activation function.
        """

        self.in_channel = in_channels
        self.out_channel = out_channels
        self.speconv = SpectralConv3d(in_channels, out_channels, modes1, modes2, modes3)
        self.linear = nn.Linear(in_channels, out_channels)

        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'gelu':
            self.activation = nn.GELU()
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

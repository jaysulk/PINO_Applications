import numpy as np

import torch
import torch.nn as nn

from functools import partial

def dht(x: torch.Tensor):
    X = torch.fft.fft(x)
    X = X.real - X.imag
    return X

def idht(X: torch.Tensor):
    # Get the size of each dimension
    dims = X.size()
    
    # Calculate the normalization factor
    n = torch.prod(torch.tensor(dims)).item()
    
    # Compute the DHT
    X = dht(X)
    
    # Element-wise division for normalization
    x = X / n
    
    return x

def hcompl_mul1d(x, y): 
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    #return torch.einsum("bix,iox->box", a, b)
    X = dht(x)
    Y = dht(y)
    Xflip = torch.roll(torch.flip(x, [0]), 1, dims=0)
    Yflip = torch.roll(torch.flip(y, [0]), 1, dims=0)
    Yplus = Y + Yflip
    Yminus = Y - Yflip
    Z = torch.einsum("bix,iox->box", X, Yplus) + torch.einsum("bix,iox->box", Xflip, Yminus)
    Z *= 0.5
    z = idht(Z)
    
    return z


def hcompl_mul2d(x, y): 
    """ Multiplies tensors a and b using the convolution theorem for the DHT.
    Assumes hartley_transform and inverse_hartley_transform are defined.
    """
    X = dht(x)
    Y = dht(y)
    Xflip = torch.roll(torch.flip(x, [0, 1]), shifts=(1, 1), dims=(0, 1))
    Yflip = torch.roll(torch.flip(y, [0, 1]), shifts=(1, 1), dims=(0, 1))

    Yplus = y + Yflip
    Yminus = y - Yflip
    Z = torch.einsum("bixy,ioxy->boxy", x, Yplus) + torch.einsum("bixy,ioxy->boxy",  Xflip, Yminus)
    Z *= 0.5
    z = idht(Z)
    
    return z


def hcompl_mul3d(x, y): 
    """ Multiplies tensors a and b using the convolution theorem for the DHT.
    Assumes hartley_transform and inverse_hartley_transform are defined.
    """
    X = dht(x)
    Y = dht(y)
    Xflip = torch.roll(torch.flip(x, [0, 1]), shifts=(1, 1), dims=(0, 1))
    Yflip = torch.roll(torch.flip(y, [0, 1]), shifts=(1, 1), dims=(0, 1))

    Yplus = y + Yflip
    Yminus = y - Yflip
    Z = torch.einsum("bixyz,ioxyz->boxyz", x, Yplus) + torch.einsum("bixyz,ioxyz->boxyz",  Xflip, Yminus)
    Z *= 0.5
    z = idht(Z)
    
    return z

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
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, s=[x.size(-1)], dim=[2])
        return x

class HartleyConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(DHartleyConv1d, self).__init__()

        """
        1D Hartley layer. It does DHT, linear transform, and Inverse DHT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Hartley modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Hartley coeffcients up to factor of e^(- something constant)
        x_ft = dht(x)

        # Multiply relevant Hartley modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.float)
        out_ft[:, :, :self.modes1] = dcompl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        # Return to physical space
        x = idht(out_ft)

        print(f"Output shape of HartleyConv2d: {x.shape}")
        return x


class HSpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(HSpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, s=[x.size(-1)], dim=[2])
        return x




################################################################
# 2d fourier layer
################################################################


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x, gridy=None):
        batchsize = x.shape[0]
        size1 = x.shape[-2]
        size2 = x.shape[-1]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2, 3])

        if gridy is None:
            # Multiply relevant Fourier modes
            out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, device=x.device,
                                 dtype=torch.cfloat)
            out_ft[:, :, :self.modes1, :self.modes2] = \
                compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
            out_ft[:, :, -self.modes1:, :self.modes2] = \
                compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

            # Return to physical space
            x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)), dim=[2, 3])
        else:
            factor1 = compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
            factor2 = compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
            x = self.ifft2d(gridy, factor1, factor2, self.modes1, self.modes2) / (size1 * size2)
        return x

    def ifft2d(self, gridy, coeff1, coeff2, k1, k2):

        # y (batch, N, 2) locations in [0,1]*[0,1]
        # coeff (batch, channels, kmax, kmax)

        batchsize = gridy.shape[0]
        N = gridy.shape[1]
        device = gridy.device
        m1 = 2 * k1
        m2 = 2 * k2 - 1

        # wavenumber (m1, m2)
        k_x1 =  torch.cat((torch.arange(start=0, end=k1, step=1), \
                            torch.arange(start=-(k1), end=0, step=1)), 0).reshape(m1,1).repeat(1,m2).to(device)
        k_x2 =  torch.cat((torch.arange(start=0, end=k2, step=1), \
                            torch.arange(start=-(k2-1), end=0, step=1)), 0).reshape(1,m2).repeat(m1,1).to(device)

        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(gridy[:,:,0].view(-1), k_x1.view(-1)).reshape(batchsize, N, m1, m2)
        K2 = torch.outer(gridy[:,:,1].view(-1), k_x2.view(-1)).reshape(batchsize, N, m1, m2)
        K = K1 + K2

        # basis (N, m1, m2)
        basis = torch.exp( 1j * 2* np.pi * K).to(device)

        # coeff (batch, channels, m1, m2)
        coeff3 = coeff1[:,:,1:,1:].flip(-1, -2).conj()
        coeff4 = torch.cat([coeff1[:,:,0:1,1:].flip(-1).conj(), coeff2[:,:,:,1:].flip(-1, -2).conj()], dim=-2)
        coeff12 = torch.cat([coeff1, coeff2], dim=-2)
        coeff43 = torch.cat([coeff4, coeff3], dim=-2)
        coeff = torch.cat([coeff12, coeff43], dim=-1)

        # Y (batch, channels, N)
        Y = torch.einsum("bcxy,bnxy->bcn", coeff, basis)
        Y = Y.real
        return Y

class HartleyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(HartleyConv2d, self).__init__() # Corrected super call
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.float))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.float))

    def forward(self, x): # Removed gridy parameter
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = dht(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, device=x.device,
                                dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            hcompl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            hcompl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = idht(out_ft)
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

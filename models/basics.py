import torch
import torch.nn as nn
import torch.nn.functional as F

def dht(x: torch.Tensor, dim=None) -> torch.Tensor:
    # Compute the N-dimensional FFT of the input tensor with orthonormal normalization
    result = torch.fft.fftn(x, dim=dim, norm='ortho')
    
    # Combine real and imaginary parts to compute the DHT
    return result.real - result.imag  # Use subtraction to match DHT definition

def idht(x: torch.Tensor, dim=None) -> torch.Tensor:
    # Since the DHT is its own inverse (up to a scaling factor), we can use the same function
    return dht(x, dim=dim)

def compl_mul1d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    # x1: (batch_size, in_channels, sequence_length)
    # x2: (in_channels, out_channels, sequence_length)
    X1_H_k = x1
    X2_H_k = x2
    # Compute H_x[N - k] and H_y[N - k]
    X1_H_neg_k = torch.roll(torch.flip(x1, dims=[-1]), shifts=1, dims=[-1])
    X2_H_neg_k = torch.roll(torch.flip(x2, dims=[-1]), shifts=1, dims=[-1])

    # Compute the convolution using the DHT convolution theorem
    result = 0.5 * (torch.einsum('bix,iox->box', X1_H_k, X2_H_k) -
                    torch.einsum('bix,iox->box', X1_H_neg_k, X2_H_neg_k) +
                    torch.einsum('bix,iox->box', X1_H_k, X2_H_neg_k) +
                    torch.einsum('bix,iox->box', X1_H_neg_k, X2_H_k))

    return result

def compl_mul2d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    # x1: (batch_size, in_channels, height, width)
    # x2: (in_channels, out_channels, height, width)
    X1_H_k = x1
    X2_H_k = x2
    # Compute H_x[N1 - k1, N2 - k2] and H_y[N1 - k1, N2 - k2]
    X1_H_neg_k = torch.roll(torch.flip(x1, dims=[-2, -1]), shifts=(1, 1), dims=[-2, -1])
    X2_H_neg_k = torch.roll(torch.flip(x2, dims=[-2, -1]), shifts=(1, 1), dims=[-2, -1])

    result = 0.5 * (torch.einsum('bixy,ioxy->boxy', X1_H_k, X2_H_k) -
                    torch.einsum('bixy,ioxy->boxy', X1_H_neg_k, X2_H_neg_k) +
                    torch.einsum('bixy,ioxy->boxy', X1_H_k, X2_H_neg_k) +
                    torch.einsum('bixy,ioxy->boxy', X1_H_neg_k, X2_H_k))

    return result

def compl_mul3d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    # x1: (batch_size, in_channels, depth, height, width)
    # x2: (in_channels, out_channels, depth, height, width)
    X1_H_k = x1
    X2_H_k = x2
    # Compute H_x[N1 - k1, N2 - k2, N3 - k3] and H_y[N1 - k1, N2 - k2, N3 - k3]
    X1_H_neg_k = torch.roll(torch.flip(x1, dims=[-3, -2, -1]), shifts=(1, 1, 1), dims=[-3, -2, -1])
    X2_H_neg_k = torch.roll(torch.flip(x2, dims=[-3, -2, -1]), shifts=(1, 1, 1), dims=[-3, -2, -1])

    result = 0.5 * (torch.einsum('bixyz,ioxyz->boxyz', X1_H_k, X2_H_k) -
                    torch.einsum('bixyz,ioxyz->boxyz', X1_H_neg_k, X2_H_neg_k) +
                    torch.einsum('bixyz,ioxyz->boxyz', X1_H_k, X2_H_neg_k) +
                    torch.einsum('bixyz,ioxyz->boxyz', X1_H_neg_k, X2_H_k))

    return result

################################################################
# 1D Hartley convolution layer
################################################################

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Hartley layer. It does DHT, linear transform, and Inverse DHT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Hartley modes to multiply
        self.modes1 = modes1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1))

    def forward(self, x):
        batchsize = x.shape[0]

        # Compute Hartley coefficients
        x_ht = dht(x, dim=[2])

        # Multiply relevant Hartley modes
        out_ht = torch.zeros(batchsize, self.out_channels, x.size(-1), device=x.device, dtype=x.dtype)
        out_ht[:, :, :self.modes1] = compl_mul1d(x_ht[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = idht(out_ht, dim=[2])

        return x

################################################################
# 2D Hartley convolution layer
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

    def forward(self, x):
        batchsize = x.shape[0]
        size1 = x.shape[-2]
        size2 = x.shape[-1]

        # Compute Hartley coefficients
        x_ht = dht(x, dim=[2, 3])

        # Multiply relevant Hartley modes
        out_ht = torch.zeros(batchsize, self.out_channels, size1, size2, device=x.device, dtype=x.dtype)
        out_ht[:, :, :self.modes1, :self.modes2] = compl_mul2d(
            x_ht[:, :, :self.modes1, :self.modes2], self.weights1)

        # Return to physical space
        x = idht(out_ht, dim=[2, 3])

        return x

################################################################
# 3D Hartley convolution layer
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Hartley modes to multiply
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(
            in_channels, out_channels, self.modes1, self.modes2, self.modes3))

    def forward(self, x):
        batchsize = x.shape[0]
        size1 = x.shape[-3]
        size2 = x.shape[-2]
        size3 = x.shape[-1]

        # Compute Hartley coefficients
        x_ht = dht(x, dim=[2, 3, 4])

        # Multiply relevant Hartley modes
        out_ht = torch.zeros(batchsize, self.out_channels, size1, size2, size3, device=x.device, dtype=x.dtype)
        out_ht[:, :, :self.modes1, :self.modes2, :self.modes3] = compl_mul3d(
            x_ht[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)

        # Return to physical space
        x = idht(out_ht, dim=[2, 3, 4])

        return x

################################################################
# FourierBlock (Using SpectralConv3d)
################################################################

class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, activation='tanh'):
        super(FourierBlock, self).__init__()

        # Spectral convolution layer (using 3D Hartley transform)
        self.speconv = SpectralConv3d(in_channels, out_channels, modes1, modes2, modes3)

        # Linear layer applied across the channel dimension
        self.linear = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        # Activation function selection
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError(f'{activation} is not supported')

    def forward(self, x):
        '''
        Input x: (batchsize, in_channels, x_grid, y_grid, t_grid)
        '''
        x1 = self.speconv(x)
        x2 = self.linear(x)
        x = x1 + x2

        if self.activation is not None:
            x = self.activation(x)

        return x

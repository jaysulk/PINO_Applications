import torch
import torch.nn as nn
import torch.fft

# Define the Discrete Hartley Transform (DHT) and inverse DHT
def dht(x: torch.Tensor, dim=None) -> torch.Tensor:
    # Compute the N-dimensional FFT of the input tensor
    result = torch.fft.fftn(x, dim=dim)
    # Combine real and imaginary parts to compute the DHT
    return result.real + result.imag

def idht(x: torch.Tensor, dim=None) -> torch.Tensor:
    # Compute the DHT of the input tensor
    transformed = dht(x, dim=dim)
    
    # Determine normalization factor based on the specified dimensions
    if dim is None:
        # If dim is None, use the total number of elements
        normalization_factor = x.numel()
    else:
        # Ensure dim is a list of dimensions
        if isinstance(dim, int):
            dim = [dim]
        normalization_factor = 1
        for d in dim:
            normalization_factor *= x.size(d)
    
    # Return the normalized inverse DHT
    return transformed / normalization_factor

# 1D Complex Multiplication (DHT-based)
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

# 2D Complex Multiplication (DHT-based)
def compl_mul2d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    X1_H_k = x1
    X2_H_k = x2
    X1_H_neg_k = torch.roll(torch.flip(x1, dims=[-2, -1]), shifts=(1, 1), dims=[-2, -1])
    X2_H_neg_k = torch.roll(torch.flip(x2, dims=[-2, -1]), shifts=(1, 1), dims=[-2, -1])

    result = 0.5 * (torch.einsum('bixy,ioxy->boxy', X1_H_k, X2_H_k) -
                    torch.einsum('bixy,ioxy->boxy', X1_H_neg_k, X2_H_neg_k) +
                    torch.einsum('bixy,ioxy->boxy', X1_H_k, X2_H_neg_k) +
                    torch.einsum('bixy,ioxy->boxy', X1_H_neg_k, X2_H_k))
    return result

# 3D Complex Multiplication (DHT-based)
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

# Adaptive Spectral Convolution for 1D
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, max_modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_modes1 = max_modes1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, max_modes1, 2))

    def forward(self, x, error_estimate):
        batchsize = x.shape[0]
        x_ft = dht(x, dim=[2])

        # Dynamically select the number of modes based on the error estimate
        modes1 = min(self.max_modes1, max(1, int(self.max_modes1 * error_estimate)))

        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :modes1] = compl_mul1d(x_ft[:, :, :modes1], self.weights1)

        x = idht(x)
        return x

# Adaptive Spectral Convolution for 2D
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, max_modes1, max_modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_modes1 = max_modes1
        self.max_modes2 = max_modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, max_modes1, max_modes2, dtype=torch.cfloat))

    def forward(self, x, error_estimate):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[2, 3])

        # Dynamically select the number of modes based on the error estimate
        modes1 = min(self.max_modes1, max(1, int(self.max_modes1 * error_estimate)))
        modes2 = min(self.max_modes2, max(1, int(self.max_modes2 * error_estimate)))

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :modes1, :modes2] = compl_mul2d(x_ft[:, :, :modes1, :modes2], self.weights1)

        x = idht(x)
        return x

# Adaptive Spectral Convolution for 3D
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, max_modes1, max_modes2, max_modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_modes1 = max_modes1
        self.max_modes2 = max_modes2
        self.max_modes3 = max_modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, max_modes1, max_modes2, max_modes3, dtype=torch.cfloat))

    def forward(self, x, error_estimate):
        batchsize = x.shape[0]
        x_ft = dht(x)

        # Dynamically select the number of modes based on the error estimate
        modes1 = min(self.max_modes1, max(1, int(self.max_modes1 * error_estimate)))
        modes2 = min(self.max_modes2, max(1, int(self.max_modes2 * error_estimate)))
        modes3 = min(self.max_modes3, max(1, int(self.max_modes3 * error_estimate)))

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(2), x.size(3), x.size(4) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :modes1, :modes2, :modes3] = compl_mul3d(x_ft[:, :, :modes1, :modes2, :modes3], self.weights1)

        x = idht(x)
        return x

# Example FourierBlock with adaptive refinement (3D case)
class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, max_modes1, max_modes2, max_modes3, activation='tanh'):
        super(FourierBlock, self).__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.speconv = AdaptiveSpectralConv3d(in_channels, out_channels, max_modes1, max_modes2, max_modes3)
        self.linear = nn.Conv1d(in_channels, out_channels, 1)

        if activation == 'tanh':
            self.activation = torch.tanh_
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = self.swish
        elif activation == 'none':
            self.activation = None

    @staticmethod
    def swish(x):
        return x * torch.sigmoid(x)

    def forward(self, x, error_estimate):
        '''
        input x: (batchsize, channel width, x_grid, y_grid, z_grid)
        '''
        x1 = self.speconv(x, error_estimate)
        x2 = self.linear(x.view(x.shape[0], self.in_channel, -1))
        out = x1 + x2.view(x.shape[0], self.out_channel, x.shape[2], x.shape[3], x.shape[4])
        if self.activation is not None:
            out = self.activation(out)
        return out

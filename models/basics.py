import torch
import torch.nn as nn
import torch.nn.functional as F

def dht_fft(x: torch.Tensor, dim: int) -> torch.Tensor:
    # Compute the 1D FFT of the input tensor along the specified dimension
    X_fft = torch.fft.fft(x, dim=dim, norm="ortho")
    
    # Compute the real and imaginary parts
    real_part = X_fft.real
    imag_part = X_fft.imag
    
    # DHT is the sum of the real part and the negative of the imaginary part
    dht_result = real_part - imag_part
    
    return dht_result

def dht(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:  # For 1D DHT
        return dht_fft(x, dim=2)
    elif x.ndim == 4:  # For 2D DHT
        result = dht_fft(x, dim=2)
        return dht_fft(result, dim=3)
    elif x.ndim == 5:  # For 3D DHT
        result = dht_fft(x, dim=2)
        result = dht_fft(result, dim=3)
        return dht_fft(result, dim=4)
    else:
        raise ValueError("Only 1D (3D tensors), 2D (4D tensors), and 3D (5D tensors) tensors are supported.")

def idht(x: torch.Tensor) -> torch.Tensor:
    # Compute the DHT
    transformed = dht(x)
    
    # Determine normalization factor
    if x.ndim == 3:
        N = x.size(2)
        normalization_factor = N
    elif x.ndim == 4:
        M, N = x.size(2), x.size(3)
        normalization_factor = M * N
    elif x.ndim == 5:
        D, M, N = x.size(2), x.size(3), x.size(4)
        normalization_factor = D * M * N
    else:
        raise ValueError(f"Input tensor must be 3D, 4D, or 5D, but got {x.ndim}D with shape {x.shape}.")

    return transformed / normalization_factor

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
# Fourier Block
################################################################

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
        
        # Flatten spatial dimensions for linear operation
        batchsize, in_channels, x_grid, y_grid, t_grid = x.shape
        x2 = self.linear(x.view(batchsize, in_channels, -1))
        x2 = x2.view(batchsize, self.out_channel, x_grid, y_grid, t_grid)
        
        out = x1 + x2
        if self.activation is not None:
            out = self.activation(out)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.filters import gaussian

################################################################
# Gaussian Smoothing Function using scikit-image
################################################################

#def gaussian_smoothing(x, sigma=1.0):
#    """
#    Applies Gaussian smoothing to the output using scikit-image.
#    """
#    # Convert PyTorch tensor to NumPy array
#    x_np = x.detach().cpu().numpy()
#
#    # Apply Gaussian smoothing
#    if x.dim() == 3:  # 1D case
#        x_smoothed = gaussian(x_np, sigma=sigma, mode='wrap')
#    elif x.dim() == 4:  # 2D case
#        x_smoothed = gaussian(x_np, sigma=sigma)
#    elif x.dim() == 5:  # 3D case
#        x_smoothed = gaussian(x_np, sigma=sigma)
#    else:
#        raise ValueError("Input tensor must have 3, 4, or 5 dimensions.")
#   
#    # Convert back to PyTorch tensor
#    return torch.tensor(x_smoothed, device=x.device)

################################################################
# Low-Pass Filter Function
################################################################

#def low_pass_filter(x_ht, cutoff):
#    """
#    Applies a low-pass filter to the spectral coefficients (DHT output).
#    Frequencies higher than `cutoff` are dampened.
#    """
#    size = x_ht.shape[-1]  # Get the last dimension (frequency axis)
#    frequencies = torch.fft.fftfreq(size, d=1.0).to(x_ht.device)  # Compute frequency bins
#    filter_mask = torch.abs(frequencies) <= cutoff  # Mask for low frequencies
#    return x_ht * filter_mask.view(1, 1, -1).expand_as(x_ht)  # Apply mask

################################################################
# Discrete Hartley Transforms (DHT)
################################################################

def dht_1d(x: torch.Tensor) -> torch.Tensor:
    transform_dims = [2]  # Length dimension
    return torch.fft.fftn(x, dim=transform_dims).real - torch.fft.fftn(x.flip(-1), dim=transform_dims).imag

def dht_2d(x: torch.Tensor) -> torch.Tensor:
    transform_dims = [2, 3]  # Height and Width dimensions
    return torch.fft.fftn(x, dim=transform_dims).real - torch.fft.fftn(x.flip(-2, -1), dim=transform_dims).imag

def dht_3d(x: torch.Tensor) -> torch.Tensor:
    transform_dims = [2, 3, 4]  # Depth, Height, and Width dimensions
    return torch.fft.fftn(x, dim=transform_dims).real - torch.fft.fftn(x.flip(-3,-2,-1), dim=transform_dims).imag

def idht_1d(X: torch.Tensor) -> torch.Tensor:
    n = X.shape[2]  # Length
    x = dht_1d(X)
    x = x / n
    return x

def idht_2d(X: torch.Tensor) -> torch.Tensor:
    n = X.shape[2] * X.shape[3]  # Height * Width
    x = dht_2d(X)
    x = x / n
    return x

def idht_3d(X: torch.Tensor) -> torch.Tensor:
    n = X.shape[2] * X.shape[3] * X.shape[4]  # Depth * Height * Width
    x = dht_3d(X)
    x = x / n
    return x

################################################################
# Convolutions
################################################################

def compl_mul1d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    return torch.einsum("bix,iox->box", x1, x2)

def compl_mul2d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    return torch.einsum("bixy,ioxy->boxy", x1, x2)

def compl_mul3d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bixyz,ioxyz->boxyz", x1, x2)

#def compl_mul1d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#    X1_H_k = x1
#    X2_H_k = x2
#    X1_H_neg_k = torch.roll(torch.flip(x1, dims=[-1]), shifts=1, dims=[-1])
#    X2_H_neg_k = torch.roll(torch.flip(x2, dims=[-1]), shifts=1, dims=[-1])
#
#    result = 0.5 * (torch.einsum('bix,iox->box', X1_H_k, X2_H_k) - 
#                     torch.einsum('bix,iox->box', X1_H_neg_k, X2_H_neg_k) +
#                     torch.einsum('bix,iox->box', X1_H_k, X2_H_neg_k) + 
#                     torch.einsum('bix,iox->box', X1_H_neg_k, X2_H_k))
#
#    return result

#def compl_mul2d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#    X1_H_k = x1
#    X2_H_k = x2
#    X1_H_neg_k = torch.roll(torch.flip(x1, dims=[-1, -2]), shifts=(1, 1), dims=[-1, -2])
#    X2_H_neg_k = torch.roll(torch.flip(x2, dims=[-1, -2]), shifts=(1, 1), dims=[-1, -2])
#    
#    result = 0.5 * (torch.einsum('bixy,ioxy->boxy', X1_H_k, X2_H_k) - 
#                    torch.einsum('bixy,ioxy->boxy', X1_H_neg_k, X2_H_neg_k) +
#                    torch.einsum('bixy,ioxy->boxy', X1_H_k, X2_H_neg_k) + 
#                    torch.einsum('bixy,ioxy->boxy', X1_H_neg_k, X2_H_k))
#    
#    return result

#def compl_mul3d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#    X1_H_k = x1
#    X2_H_k = x2
#    X1_H_neg_k = torch.roll(torch.flip(x1, dims=[-3, -2, -1]), shifts=(1, 1, 1), dims=[-3, -2, -1])
#    X2_H_neg_k = torch.roll(torch.flip(x2, dims=[-3, -2, -1]), shifts=(1, 1, 1), dims=[-3, -2, -1])
#
#    result = 0.5 * (torch.einsum('bixyz,ioxyz->boxyz', X1_H_k, X2_H_k) - 
#                     torch.einsum('bixyz,ioxyz->boxyz', X1_H_neg_k, X2_H_neg_k) +
#                     torch.einsum('bixyz,ioxyz->boxyz', X1_H_k, X2_H_neg_k) + 
#                     torch.einsum('bixyz,ioxyz->boxyz', X1_H_neg_k, X2_H_k))
#
#    return result

################################################################
# 1D Hartley convolution layer with adaptive basis refinement
################################################################

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        assert self.in_channels == self.out_channels, "For adaptive refinement, in_channels must equal out_channels in SpectralConv1d."

        self.modes1 = modes1
        self.refine_modes1 = modes1  # Can be adjusted separately if needed

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1))
        self.refine_weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.refine_modes1))

        # Set a default error threshold
        self.error_threshold = 0.1

    def forward(self, x):
        batchsize = x.shape[0]

        # Compute Hartley coefficients
        x_ht = dht_1d(x)

        # Multiply relevant Hartley modes
        out_ht = torch.zeros(batchsize, self.out_channels, x.size(-1), device=x.device, dtype=x.dtype)
        out_ht[:, :, :self.modes1] = conv_1d(x_ht[:, :, :self.modes1], self.weights1)

        # Inverse Hartley to reconstruct the signal
        x_reconstructed = idht_1d(out_ht)

        # Compute reconstruction error
        error = torch.abs(x - x_reconstructed)

        # Create a mask for refinement based on error threshold
        mask = (error > self.error_threshold).float()
        x_refine = x * mask

        # Compute Hartley coefficients for refined regions
        x_refine_ht = dht_1d(x_refine)

        # Multiply relevant Hartley modes for refinement
        refine_out_ht = torch.zeros(batchsize, self.out_channels, x.size(-1), device=x.device, dtype=x.dtype)
        refine_out_ht[:, :, :self.refine_modes1] = conv_1d(x_refine_ht[:, :, :self.refine_modes1], self.refine_weights1)

        # Combine refined output with initial output
        out_ht += refine_out_ht

        # Return to physical space
        x_final = idht_1d(out_ht)

        return x_final

################################################################
# 2D Hartley convolution layer with adaptive basis refinement
################################################################

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert self.in_channels == self.out_channels, "For adaptive refinement, in_channels must equal out_channels in SpectralConv2d."

        self.modes1 = modes1
        self.modes2 = modes2
        self.refine_modes1 = modes1
        self.refine_modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.refine_weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.refine_modes1, self.refine_modes2))

        self.error_threshold = 0.1

    def forward(self, x):
        batchsize = x.shape[0]
        size1, size2 = x.shape[-2], x.shape[-1]

        # Compute Hartley coefficients
        x_ht = dht_2d(x)

        # Multiply relevant Hartley modes
        out_ht = torch.zeros(batchsize, self.out_channels, size1, size2, device=x.device, dtype=x.dtype)
        out_ht[:, :, :self.modes1, :self.modes2] = conv_2d(x_ht[:, :, :self.modes1, :self.modes2], self.weights1)

        # Inverse Hartley to reconstruct the signal
        x_reconstructed = idht_2d(out_ht)

        # Compute reconstruction error
        error = torch.abs(x - x_reconstructed)

        # Create a mask for refinement
        mask = (error > self.error_threshold).float()
        x_refine = x * mask

        # Compute Hartley coefficients for refined regions
        x_refine_ht = dht_2d(x_refine)

        # Multiply relevant Hartley modes for refinement
        refine_out_ht = torch.zeros(batchsize, self.out_channels, size1, size2, device=x.device, dtype=x.dtype)
        refine_out_ht[:, :, :self.refine_modes1, :self.refine_modes2] = conv_2d(x_refine_ht[:, :, :self.refine_modes1, :self.refine_modes2], self.refine_weights1)

        # Combine refined output with initial output
        out_ht += refine_out_ht

        # Return to physical space
        x_final = idht_2d(out_ht)

        return x_final

################################################################
# 3D Hartley convolution layer with adaptive basis refinement
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert self.in_channels == self.out_channels, "For adaptive refinement, in_channels must equal out_channels in SpectralConv3d."

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.refine_modes1 = modes1
        self.refine_modes2 = modes2
        self.refine_modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.refine_weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.refine_modes1, self.refine_modes2, self.refine_modes3))

        self.error_threshold = 0.1

    def forward(self, x):
        batchsize = x.shape[0]
        size1, size2, size3 = x.shape[-3], x.shape[-2], x.shape[-1]

        # Compute Hartley coefficients
        x_ht = dht_3d(x)

        # Multiply relevant Hartley modes
        out_ht = torch.zeros(batchsize, self.out_channels, size1, size2, size3, device=x.device, dtype=x.dtype)
        out_ht[:, :, :self.modes1, :self.modes2, :self.modes3] = conv_3d(x_ht[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)

        # Inverse Hartley to reconstruct the signal
        x_reconstructed = idht_3d(out_ht)

        # Compute reconstruction error
        error = torch.abs(x - x_reconstructed)

        # Create a mask for refinement
        mask = (error > self.error_threshold).float()
        x_refine = x * mask

        # Compute Hartley coefficients for refined regions
        x_refine_ht = dht_3d(x_refine)

        # Multiply relevant Hartley modes for refinement
        refine_out_ht = torch.zeros(batchsize, self.out_channels, size1, size2, size3, device=x.device, dtype=x.dtype)
        refine_out_ht[:, :, :self.refine_modes1, :self.refine_modes2, :self.refine_modes3] = conv_3d(x_refine_ht[:, :, :self.refine_modes1, :self.refine_modes2, :self.refine_modes3], self.refine_weights1)

        # Combine refined output with initial output
        out_ht += refine_out_ht

        # Return to physical space
        x_final = idht_3d(out_ht)

        return x_final

################################################################
# FourierBlock (Using SpectralConv1d, SpectralConv2d, SpectralConv3d)
################################################################

class FourierBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, activation='tanh'):
        super(FourierBlock1d, self).__init__()
        assert in_channels == out_channels, "For adaptive refinement, in_channels must equal out_channels in FourierBlock1d."

        self.speconv = SpectralConv1d(in_channels, out_channels, modes1)
        self.linear = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError(f'{activation} is not supported')

    def forward(self, x):
        x1 = self.speconv(x)  # [batch, out_channels, length]
        x2 = self.linear(x)   # [batch, out_channels, length]
        x = x1 + x2           # [batch, out_channels, length]

        if self.activation is not None:
            x = self.activation(x)

        return x

class FourierBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, activation='tanh'):
        super(FourierBlock2d, self).__init__()
        assert in_channels == out_channels, "For adaptive refinement, in_channels must equal out_channels in FourierBlock2d."

        self.speconv = SpectralConv2d(in_channels, out_channels, modes1, modes2)
        self.linear = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError(f'{activation} is not supported')

    def forward(self, x):
        x1 = self.speconv(x)  # [batch, out_channels, height, width]
        x2 = self.linear(x)   # [batch, out_channels, height, width]
        x = x1 + x2           # [batch, out_channels, height, width]

        if self.activation is not None:
            x = self.activation(x)

        return x

class FourierBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, activation='tanh'):
        super(FourierBlock3d, self).__init__()
        assert in_channels == out_channels, "For adaptive refinement, in_channels must equal out_channels in FourierBlock3d."

        self.speconv = SpectralConv3d(in_channels, out_channels, modes1, modes2, modes3)
        self.linear = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError(f'{activation} is not supported')

    def forward(self, x):
        x1 = self.speconv(x)  # [batch, out_channels, depth, height, width]
        x2 = self.linear(x)   # [batch, out_channels, depth, height, width]
        x = x1 + x2           # [batch, out_channels, depth, height, width]

        if self.activation is not None:
            x = self.activation(x)

        return x

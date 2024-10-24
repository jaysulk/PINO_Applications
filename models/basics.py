import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

################################################################
# Low-Pass Filter Function
################################################################

# def low_pass_filter(x_ht, cutoff):
#     """
#     Applies a low-pass filter to the spectral coefficients (DHT output).
#     Frequencies higher than `cutoff` are dampened.
#     """
#     size = x_ht.shape[-1]  # Get the last dimension (frequency axis)
#     frequencies = torch.fft.fftfreq(size, d=1.0)  # Compute frequency bins
#     filter_mask = torch.abs(frequencies) <= cutoff  # Mask for low frequencies
#     return x_ht * filter_mask.to(x_ht.device)

################################################################
# Gaussian Smoothing Function
################################################################

# def gaussian_smoothing(x, kernel_size=5, sigma=1.0):
#     """
#     Applies Gaussian smoothing to the output.
#     """
#     # Apply Gaussian blur (use 2D or 3D kernel as needed)
#     return F.gaussian_blur(x, kernel_size=[kernel_size], sigma=[sigma])

################################################################
# Data Augmentation Function
################################################################

# def augment_data(inputs, shift_range=0.1, scale_range=0.05):
#     """
#     Augment input data by applying random shifts and scaling.
#     
#     Parameters:
#     - inputs: torch.Tensor, the input data to be augmented
#     - shift_range: float, the maximum range for random shifts
#     - scale_range: float, the maximum range for random scaling
#     
#     Returns:
#     - augmented_inputs: torch.Tensor, the augmented input data
#     """
#     # Apply random shifts
#     shifts = torch.rand(inputs.size()) * shift_range
#     augmented_inputs = inputs + shifts
#     
#     # Apply random scaling
#     scales = 1 + torch.rand(inputs.size()) * scale_range
#     augmented_inputs = augmented_inputs * scales
#     
#     return augmented_inputs

################################################################
# Transforms
################################################################

################################################################
# Discrete Hartley Transforms (DHT)
################################################################

def dht_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the 1D Discrete Hartley Transform (DHT) of the input tensor.

    Args:
        x (torch.Tensor): Input tensor with shape [batch, channels, length].

    Returns:
        torch.Tensor: DHT of the input tensor.
    """
    transform_dims = [2]  # Length dimension
    result = torch.fft.fftn(x, dim=transform_dims).real - torch.fft.fftn(x.flip(-1), dim=transform_dims).imag

    # Compute the norm of the result
    norm = result.norm(dim=transform_dims, keepdim=True)
    
    # Normalize the result to make it orthonormal
    orthonormal_result = result / norm
    
    # If you want to ensure no division by zero occurs
    return torch.where(norm != 0, orthonormal_result, torch.zeros_like(orthonormal_result))

# orthonormal_result is now orthonormal


def dht_2d(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the 2D Discrete Hartley Transform (DHT) of the input tensor.

    Args:
        x (torch.Tensor): Input tensor with shape [batch, channels, height, width].

    Returns:
        torch.Tensor: DHT of the input tensor.
    """
    transform_dims = [2, 3]  # Height and Width dimensions
    result = torch.fft.fftn(x, dim=transform_dim).real - torch.fft.fftn(x.flip(-2, -1), dim=transform_dims).imag

    # Compute the norm of the result
    norm = result.norm(dim=transform_dims, keepdim=True)
    
    # Normalize the result to make it orthonormal
    orthonormal_result = result / norm
    
    # If you want to ensure no division by zero occurs
    return torch.where(norm != 0, orthonormal_result, torch.zeros_like(orthonormal_result))

def dht_3d(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the 3D Discrete Hartley Transform (DHT) of the input tensor.

    Args:
        x (torch.Tensor): Input tensor with shape [batch, channels, depth, height, width].

    Returns:
        torch.Tensor: DHT of the input tensor.
    """
    transform_dims = [2, 3, 4]  # Depth, Height, and Width dimensions
    result = torch.fft.fftn(x, dim=transform_dims).real - torch.fft.fftn(x.flip(-1), dim=transform_dims).imag

    # Compute the norm of the result
    norm = result.norm(dim=transform_dims, keepdim=True)
    
    # Normalize the result to make it orthonormal
    orthonormal_result = result / norm
    
    # If you want to ensure no division by zero occurs
    return torch.where(norm != 0, orthonormal_result, torch.zeros_like(orthonormal_result))

def idht_1d(X: torch.Tensor) -> torch.Tensor:
    """
    Compute the Inverse 1D Discrete Hartley Transform (IDHT) of the input tensor.

    Since the DHT is involutory, IDHT(x) = (1/n) * DHT(DHT(x))

    Args:
        X (torch.Tensor): Input tensor in the DHT domain with shape [batch, channels, length].

    Returns:
        torch.Tensor: Inverse DHT of the input tensor.
    """
    n = X.shape[2]  # Length
    x = dht_1d(X)
    x = x / n
    return x

def idht_2d(X: torch.Tensor) -> torch.Tensor:
    """
    Compute the Inverse 2D Discrete Hartley Transform (IDHT) of the input tensor.

    Since the DHT is involutory, IDHT(x) = (1/n) * DHT(DHT(x))

    Args:
        X (torch.Tensor): Input tensor in the DHT domain with shape [batch, channels, height, width].

    Returns:
        torch.Tensor: Inverse DHT of the input tensor.
    """
    n = X.shape[2] * X.shape[3]  # Height * Width
    x = dht_2d(X)
    x = x / n
    return x

def idht_3d(X: torch.Tensor) -> torch.Tensor:
    """
    Compute the Inverse 3D Discrete Hartley Transform (IDHT) of the input tensor.

    Since the DHT is involutory, IDHT(x) = (1/n) * DHT(DHT(x))

    Args:
        X (torch.Tensor): Input tensor in the DHT domain with shape [batch, channels, depth, height, width].

    Returns:
        torch.Tensor: Inverse DHT of the input tensor.
    """
    n = X.shape[2] * X.shape[3] * X.shape[4]  # Depth * Height * Width
    x = dht_3d(X)
    x = x / n
    return x

################################################################
# Convolutions
################################################################

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
# Direct Convolution in Hartley Domain
################################################################

def conv_1d(x_ht: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Perform 1D convolution in the Hartley domain using the generalized compl_mult function.

    Args:
        x_ht (torch.Tensor): Hartley-transformed input tensor [batch, in_channels, modes1]
        weights (torch.Tensor): Hartley-transformed weights [in_channels, out_channels, modes1]

    Returns:
        torch.Tensor: Convolved tensor in the Hartley domain [batch, out_channels, modes1]
    """
    # For 1D, flip and shift along the last dimension (-1)
    return compl_mul1d(x_ht, weights)

def conv_2d(x_ht: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Perform 2D convolution in the Hartley domain using the generalized compl_mult function.

    Args:
        x_ht (torch.Tensor): Hartley-transformed input tensor [batch, in_channels, modes1, modes2]
        weights (torch.Tensor): Hartley-transformed weights [in_channels, out_channels, modes1, modes2]

    Returns:
        torch.Tensor: Convolved tensor in the Hartley domain [batch, out_channels, modes1, modes2]
    """
    # For 2D, flip and shift along the last two dimensions (-1, -2)
    return compl_mul2d(x_ht, weights)

def conv_3d(x_ht: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Perform 3D convolution in the Hartley domain using the generalized compl_mult function.

    Args:
        x_ht (torch.Tensor): Hartley-transformed input tensor [batch, in_channels, modes1, modes2, modes3]
        weights (torch.Tensor): Hartley-transformed weights [in_channels, out_channels, modes1, modes2, modes3]

    Returns:
        torch.Tensor: Convolved tensor in the Hartley domain [batch, out_channels, modes1, modes2, modes3]
    """
    # For 3D, flip and shift along the last three dimensions (-1, -2, -3)
    return compl_mul3d(x_ht, weights)

################################################################
# Spectral Convolution Layers
################################################################

################################################################
# 1D Hartley Convolution Layer
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
            self.scale * torch.rand(in_channels, out_channels, self.modes1)
        )

    def forward(self, x):
        batchsize = x.shape[0]

        # Compute Hartley coefficients
        x_ht = dht_1d(x)  # [batch, in_channels, length]

        # Multiply relevant Hartley modes
        out_ht = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1),
            device=x.device,
            dtype=x.dtype
        )
        out_ht[:, :, :self.modes1] = conv_1d(
            x_ht[:, :, :self.modes1],
            self.weights1
        )

        # Return to physical space
        x = idht_1d(out_ht)  # [batch, out_channels, length]

        return x

################################################################
# 2D Hartley Convolution Layer
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

    def forward(self, x):
        batchsize = x.shape[0]
        size1 = x.shape[-2]
        size2 = x.shape[-1]

        # Compute Hartley coefficients
        x_ht = dht_2d(x)  # [batch, in_channels, height, width]

        # Multiply relevant Hartley modes
        out_ht = torch.zeros(
            batchsize,
            self.out_channels,
            size1,
            size2,
            device=x.device,
            dtype=x.dtype
        )
        out_ht[:, :, :self.modes1, :self.modes2] = conv_2d(
            x_ht[:, :, :self.modes1, :self.modes2],
            self.weights1
        )

        # Return to physical space
        x = idht_2d(out_ht)  # [batch, out_channels, height, width]

        return x

################################################################
# 3D Hartley Convolution Layer
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
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, self.modes3
            )
        )

    def forward(self, x):
        batchsize = x.shape[0]
        size1, size2, size3 = x.shape[-3], x.shape[-2], x.shape[-1]

        # Compute Hartley coefficients
        x_ht = dht_3d(x)  # [batch, in_channels, depth, height, width]

        # Multiply relevant Hartley modes using the corrected conv_3d
        out_ht = torch.zeros(
            batchsize,
            self.out_channels,
            size1,
            size2,
            size3,
            device=x.device,
            dtype=x.dtype
        )
        out_ht[:, :, :self.modes1, :self.modes2, :self.modes3] = conv_3d(
            x_ht[:, :, :self.modes1, :self.modes2, :self.modes3],
            self.weights1
        )

        # Return to physical space
        x = idht_3d(out_ht)  # [batch, out_channels, depth, height, width]

        return x

################################################################
# FourierBlock 
################################################################

class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, activation='tanh'):
        super(FourierBlock, self).__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.speconv = SpectralConv3d(in_channels, out_channels, modes1, modes2, modes3)  # Assuming 3D
        self.linear = nn.Conv1d(in_channels, out_channels, 1)

        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'gelu':
            self.activation = nn.GELU()  # Corrected to instantiate GELU
        elif activation == 'swish':
            self.activation = self.swish
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    @staticmethod
    def swish(x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        '''
        input x: (batchsize, channel width, x_grid, y_grid, z_grid)
        '''
        x1 = self.speconv(x)
        x2 = self.linear(x.view(x.shape[0], self.in_channel, -1))
        x2 = x2.view(x.shape[0], self.out_channel, x.shape[2], x.shape[3], x.shape[4])
        out = x1 + x2
        if self.activation is not None:
            out = self.activation(out)
        return out  # Removed the redundant return statement

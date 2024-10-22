import numpy as np
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

def fht1d(x):
    N = x.shape[-1]
    # Check if N is a power of 2
    if (N & (N - 1)) != 0:
        # Pad x to the next power of 2
        next_pow_two = 1 << (N - 1).bit_length()
        pad_size = next_pow_two - N
        x = F.pad(x, (0, pad_size))
        N = next_pow_two

    if N == 1:
        return x
    else:
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        fht_even = fht1d(x_even)
        fht_odd = fht1d(x_odd)
        k = torch.arange(N // 2, device=x.device).reshape([1] * (x.ndim - 1) + [-1])
        theta = 2 * torch.pi * k / N
        cas = torch.cos(theta) + torch.sin(theta)
        temp = cas * fht_odd
        X = torch.cat([fht_even + temp, fht_even - temp], dim=-1)
        return X

def fht_along_dim(x, dim):
    # Move the target dimension to the last dimension
    x = x.transpose(dim, -1)
    original_shape = x.shape
    N = x.shape[-1]
    # Flatten the batch dimensions
    x = x.reshape(-1, N)
    # Apply fht1d
    x = fht1d(x)
    # Now x may have a different size in the last dimension
    new_N = x.shape[-1]
    # Restore the original shape with the new last dimension size
    x = x.reshape(*original_shape[:-1], new_N)
    # Truncate or pad the last dimension back to original N
    if new_N > N:
        x = x[..., :N]
    elif new_N < N:
        pad_size = N - new_N
        x = F.pad(x, (0, pad_size))
    # Move the last dimension back to its original position
    x = x.transpose(dim, -1)
    return x

def dht(x: torch.Tensor, dims=None) -> torch.Tensor:
    if dims is None:
        dims = list(range(2, x.ndim))
    for dim in dims:
        x = fht_along_dim(x, dim)
    return x

def idht(x: torch.Tensor, dims=None) -> torch.Tensor:
    if dims is None:
        dims = list(range(2, x.ndim))
    N = 1
    for dim in dims:
        N *= x.size(dim)
    # Compute the DHT (Inverse Hartley Transform)
    transformed = dht(x, dims=dims)
    # Normalize the result
    return transformed / N

def dht_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Perform the 1D Discrete Hartley Transform along the last dimension.

    Args:
        x (torch.Tensor): Input tensor of shape [..., N]

    Returns:
        torch.Tensor: DHT-transformed tensor of the same shape.
    """
    return dht(x, dims=[-1])

def idht_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Perform the inverse 1D Discrete Hartley Transform along the last dimension.

    Args:
        x (torch.Tensor): DHT-transformed tensor of shape [..., N]

    Returns:
        torch.Tensor: Inverse DHT-transformed tensor of the same shape.
    """
    return idht(x, dims=[-1])

def dht_2d(x: torch.Tensor) -> torch.Tensor:
    """
    Perform the 2D Discrete Hartley Transform along the last two dimensions.

    Args:
        x (torch.Tensor): Input tensor of shape [..., H, W]

    Returns:
        torch.Tensor: DHT-transformed tensor of the same shape.
    """
    return dht(x, dims=[-2, -1])

def idht_2d(x: torch.Tensor) -> torch.Tensor:
    """
    Perform the inverse 2D Discrete Hartley Transform along the last two dimensions.

    Args:
        x (torch.Tensor): DHT-transformed tensor of shape [..., H, W]

    Returns:
        torch.Tensor: Inverse DHT-transformed tensor of the same shape.
    """
    return idht(x, dims=[-2, -1])

def dht_3d(x: torch.Tensor) -> torch.Tensor:
    """
    Perform the 3D Discrete Hartley Transform along the last three dimensions.

    Args:
        x (torch.Tensor): Input tensor of shape [..., D, H, W]

    Returns:
        torch.Tensor: DHT-transformed tensor of the same shape.
    """
    return dht(x, dims=[-3, -2, -1])

def idht_3d(x: torch.Tensor) -> torch.Tensor:
    """
    Perform the inverse 3D Discrete Hartley Transform along the last three dimensions.

    Args:
        x (torch.Tensor): DHT-transformed tensor of shape [..., D, H, W]

    Returns:
        torch.Tensor: Inverse DHT-transformed tensor of the same shape.
    """
    return idht(x, dims=[-3, -2, -1])

def compl_mul1d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bi...,io...->bo...", x1, x2)

def compl_mul2d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bixy...,ioxy...->boxy...", x1, x2)

def compl_mul3d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bixyz...,ioxyz...->boxyz...", x1, x2)

def flip_periodic_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Perform a periodic flip of the tensor along the length dimension.

    Args:
        x (torch.Tensor): Input tensor of shape [batch, channels, length].

    Returns:
        torch.Tensor: Periodically flipped tensor with the same shape as input.
    """
    dim = 2  # Length dimension

    if x.size(dim) < 1:
        raise ValueError(f"Dimension {dim} is too small to perform flip.")

    # Initialize Z as a copy of x to avoid modifying the original tensor
    Z = x.clone()

    # Extract the first element
    first = Z.index_select(dim, torch.tensor([0], device=x.device))

    if Z.size(dim) > 1:
        # Select all elements from index 1 onwards and flip them
        remaining = Z.index_select(dim, torch.arange(1, Z.size(dim), device=x.device)).flip(dims=[dim])
        # Concatenate first and flipped remaining along the current dimension
        Z = torch.cat([first, remaining], dim=dim)
    else:
        # If there's only one element, no flipping needed
        Z = first

    return Z

def flip_periodic_2d(x: torch.Tensor) -> torch.Tensor:
    """
    Perform a periodic flip of the tensor along height and width dimensions.

    Args:
        x (torch.Tensor): Input tensor of shape [batch, channels, height, width].

    Returns:
        torch.Tensor: Periodically flipped tensor with the same shape as input.
    """
    dims = [2, 3]  # Height and Width dimensions

    Z = x.clone()

    for dim in dims:
        if Z.size(dim) < 1:
            raise ValueError(f"Dimension {dim} is too small to perform flip.")

        # Extract the first element
        first = Z.index_select(dim, torch.tensor([0], device=x.device))

        if Z.size(dim) > 1:
            # Select all elements from index 1 onwards and flip them
            remaining = Z.index_select(dim, torch.arange(1, Z.size(dim), device=x.device)).flip(dims=[dim])
            # Concatenate first and flipped remaining along the current dimension
            Z = torch.cat([first, remaining], dim=dim)
        else:
            # If there's only one element, no flipping needed
            Z = first

    return Z

def flip_periodic_3d(x: torch.Tensor) -> torch.Tensor:
    """
    Perform a periodic flip of the tensor along depth, height, and width dimensions.

    Args:
        x (torch.Tensor): Input tensor of shape [batch, channels, depth, height, width].

    Returns:
        torch.Tensor: Periodically flipped tensor with the same shape as input.
    """
    dims = [2, 3, 4]  # Depth, Height, and Width dimensions
    Z = x.clone()

    for dim in dims:
        if Z.size(dim) < 1:
            raise ValueError(f"Dimension {dim} is too small to perform flip.")

        # Extract the first element
        first = Z.index_select(dim, torch.tensor([0], device=x.device))

        if Z.size(dim) > 1:
            # Select all elements from index 1 onwards and flip them
            remaining = Z.index_select(dim, torch.arange(1, Z.size(dim), device=x.device)).flip(dims=[dim])
            # Concatenate first and flipped remaining along the current dimension
            Z = torch.cat([first, remaining], dim=dim)
        else:
            # If there's only one element, no flipping needed
            Z = first

    return Z

################################################################
# Spectral Convolution Functions
################################################################

def dht_conv_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the DHT of the convolution of two 1D tensors using the convolution theorem.

    Args:
        x (torch.Tensor): First input tensor with shape [batch, in_channels, length]
        y (torch.Tensor): Second input tensor with shape [in_channels, out_channels, modes1]

    Returns:
        torch.Tensor: DHT of the convolution of x and y.
    """
    # Compute flipped versions
    Xflip = flip_periodic_1d(x)
    Yflip = flip_periodic_1d(y)

    # Compute even and odd components
    Yeven = 0.5 * (y + Yflip)
    Yodd  = 0.5 * (y - Yflip)

    # Perform convolution using compl_mul
    term1 = compl_mul1d(x, Yeven)
    term2 = compl_mul1d(Xflip, Yodd)

    # Combine terms
    Z = term1 + term2  # [batch, out_channels, length]

    return Z

def dht_conv_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the DHT of the convolution of two 2D tensors using the convolution theorem.

    Args:
        x (torch.Tensor): First input tensor with shape [batch, in_channels, height, width]
        y (torch.Tensor): Second input tensor with shape [in_channels, out_channels, modes1, modes2]

    Returns:
        torch.Tensor: DHT of the convolution of x and y.
    """
    # Compute flipped versions
    Xflip = flip_periodic_2d(x)
    Yflip = flip_periodic_2d(y)

    # Compute even and odd components
    Yeven = 0.5 * (y + Yflip)
    Yodd  = 0.5 * (y - Yflip)

    # Perform convolution using compl_mul
    term1 = compl_mul2d(x, Yeven)
    term2 = compl_mul2d(Xflip, Yodd)

    # Combine terms
    Z = term1 + term2  # [batch, out_channels, height, width]

    return Z

def dht_conv_3d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the DHT of the convolution of two 3D tensors using the convolution theorem.

    Args:
        x (torch.Tensor): First input tensor with shape [batch, in_channels, depth, height, width]
        y (torch.Tensor): Second input tensor with shape [in_channels, out_channels, modes1, modes2, modes3]

    Returns:
        torch.Tensor: DHT of the convolution of x and y.
    """
    # Compute flipped versions
    Xflip = flip_periodic_3d(x)
    Yflip = flip_periodic_3d(y)

    # Compute even and odd components
    Yeven = 0.5 * (y + Yflip)
    Yodd  = 0.5 * (y - Yflip)

    # Perform convolution using compl_mul
    term1 = compl_mul3d(x, Yeven)
    term2 = compl_mul3d(Xflip, Yodd)

    # Combine terms
    Z = term1 + term2  # [batch, out_channels, depth, height, width]

    return Z

################################################################
# Direct Convolution in Hartley Domain
################################################################

def conv_1d(x_ht: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Perform 1D convolution in the Hartley domain.

    Args:
        x_ht (torch.Tensor): Hartley-transformed input tensor [batch, in_channels, modes1]
        weights (torch.Tensor): Hartley-transformed weights [in_channels, out_channels, modes1]

    Returns:
        torch.Tensor: Convolved tensor in the Hartley domain [batch, out_channels, modes1]
    """
    return compl_mul1d(x_ht, weights)

def conv_2d(x_ht: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Perform 2D convolution in the Hartley domain.

    Args:
        x_ht (torch.Tensor): Hartley-transformed input tensor [batch, in_channels, modes1, modes2]
        weights (torch.Tensor): Hartley-transformed weights [in_channels, out_channels, modes1, modes2]

    Returns:
        torch.Tensor: Convolved tensor in the Hartley domain [batch, out_channels, modes1, modes2]
    """
    return compl_mul2d(x_ht, weights)

def conv_3d(x_ht: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Perform 3D convolution in the Hartley domain.

    Args:
        x_ht (torch.Tensor): Hartley-transformed input tensor [batch, in_channels, modes1, modes2, modes3]
        weights (torch.Tensor): Hartley-transformed weights [in_channels, out_channels, modes1, modes2, modes3]

    Returns:
        torch.Tensor: Convolved tensor in the Hartley domain [batch, out_channels, modes1, modes2, modes3]
    """
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
        out_ht[:, :, :self.modes1] = dht_conv_1d(
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
        out_ht[:, :, :self.modes1, :self.modes2] = dht_conv_2d(
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

        # Multiply relevant Hartley modes using the corrected dht_conv_3d
        out_ht = torch.zeros(
            batchsize,
            self.out_channels,
            size1,
            size2,
            size3,
            device=x.device,
            dtype=x.dtype
        )
        out_ht[:, :, :self.modes1, :self.modes2, :self.modes3] = dht_conv_3d(
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
            self.activation = nn.GELU()
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
        return out

        return out

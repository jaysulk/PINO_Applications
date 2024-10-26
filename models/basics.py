import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.filters import gaussian

################################################################
# Gaussian Smoothing Function using scikit-image
################################################################

def gaussian_smoothing(x, sigma=1.0):
    """
    Applies Gaussian smoothing to the input tensor using scikit-image.

    Args:
        x (torch.Tensor): Input tensor.
        sigma (float): Standard deviation for Gaussian kernel.

    Returns:
        torch.Tensor: Smoothed tensor.
    """
    # Convert PyTorch tensor to NumPy array
    x_np = x.detach().cpu().numpy()

    # Apply Gaussian smoothing
    if x.dim() == 3:  # 1D case (e.g., [batch, channels, length])
        x_smoothed = gaussian(x_np, sigma=sigma, mode='wrap')
    elif x.dim() == 4:  # 2D case (e.g., [batch, channels, height, width])
        x_smoothed = gaussian(x_np, sigma=sigma, mode='wrap')
    elif x.dim() == 5:  # 3D case (e.g., [batch, channels, depth, height, width])
        x_smoothed = gaussian(x_np, sigma=sigma, mode='wrap')
    else:
        raise ValueError("Input tensor must have 3, 4, or 5 dimensions.")
   
    # Convert back to PyTorch tensor
    return torch.tensor(x_smoothed, device=x.device, dtype=x.dtype)

################################################################
# Low-Pass Filter Function
################################################################

def low_pass_filter(x_ht, cutoff):
    """
    Applies a low-pass filter to the spectral coefficients (DHT output).
    Frequencies higher than `cutoff` are dampened.

    Args:
        x_ht (torch.Tensor): Spectral coefficients.
        cutoff (float): Cutoff frequency (as a fraction of the Nyquist frequency).

    Returns:
        torch.Tensor: Filtered spectral coefficients.
    """
    size = x_ht.shape[-1]  # Get the last dimension (frequency axis)
    frequencies = torch.fft.fftfreq(size, d=1.0).to(x_ht.device)  # Compute frequency bins
    filter_mask = torch.abs(frequencies) <= cutoff  # Mask for low frequencies
    # Expand mask to match the dimensions of x_ht
    for _ in range(x_ht.dim() - 1):
        filter_mask = filter_mask.unsqueeze(0)
    return x_ht * filter_mask  # Apply mask

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
    X = torch.fft.fftn(x, dim=transform_dims)
    X = X.real - X.imag
    return X

def dht_2d(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the 2D Discrete Hartley Transform (DHT) of the input tensor.

    Args:
        x (torch.Tensor): Input tensor with shape [batch, channels, height, width].

    Returns:
        torch.Tensor: DHT of the input tensor.
    """
    transform_dims = [2, 3]  # Height and Width dimensions
    X = torch.fft.fftn(x, dim=transform_dims)
    X = X.real - X.imag
    return X

def dht_3d(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the 3D Discrete Hartley Transform (DHT) of the input tensor.

    Args:
        x (torch.Tensor): Input tensor with shape [batch, channels, depth, height, width].

    Returns:
        torch.Tensor: DHT of the input tensor.
    """
    transform_dims = [2, 3, 4]  # Depth, Height, and Width dimensions
    X = torch.fft.fftn(x, dim=transform_dims)
    X = X.real - X.imag
    return X

################################################################
# Inverse Discrete Hartley Transforms (IDHT)
################################################################

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

def compl_mul1d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    X1_H_k = x1
    X2_H_k = x2
    X1_H_neg_k = flip_periodic_1d(x1)
    X2_H_neg_k = flip_periodic_1d(x2)

    result = 0.5 * (
        torch.einsum('bix,iox->box', X1_H_k, X2_H_k) -
        torch.einsum('bix,iox->box', X1_H_neg_k, X2_H_neg_k) +
        torch.einsum('bix,iox->box', X1_H_k, X2_H_neg_k) +
        torch.einsum('bix,iox->box', X1_H_neg_k, X2_H_k)
    )

    return result

def compl_mul2d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    X1_H_k = x1
    X2_H_k = x2
    X1_H_neg_k = flip_periodic_2d(x1)
    X2_H_neg_k = flip_periodic_2d(x2)
    
    result = 0.5 * (
        torch.einsum('bixy,ioxy->boxy', X1_H_k, X2_H_k) -
        torch.einsum('bixy,ioxy->boxy', X1_H_neg_k, X2_H_neg_k) +
        torch.einsum('bixy,ioxy->boxy', X1_H_k, X2_H_neg_k) +
        torch.einsum('bixy,ioxy->boxy', X1_H_neg_k, X2_H_k)
    )
    
    return result

def compl_mul3d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    X1_H_k = x1
    X2_H_k = x2
    X1_H_neg_k = flip_periodic_3d(x1)
    X2_H_neg_k = flip_periodic_3d(x2)

    result = 0.5 * (
        torch.einsum('bixyz,ioxyz->boxyz', X1_H_k, X2_H_k) -
        torch.einsum('bixyz,ioxyz->boxyz', X1_H_neg_k, X2_H_neg_k) +
        torch.einsum('bixyz,ioxyz->boxyz', X1_H_k, X2_H_neg_k) +
        torch.einsum('bixyz,ioxyz->boxyz', X1_H_neg_k, X2_H_k)
    )

    return result

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
        out_ht[:, :, :self.modes1] = compl_mul1d(
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
        out_ht[:, :, :self.modes1, :self.modes2] = compl_mul2d(
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
        out_ht[:, :, :self.modes1, :self.modes2, :self.modes3] = compl_mul3d(
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
        self.speconv = SpectralConv3d(in_channels, out_channels, modes1, modes2, modes3)
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
        x1 = self.speconv(x)
        x2 = self.linear(x.view(x.shape[0], self.in_channel, -1))
        x2 = x2.view(x.shape[0], self.out_channel, x.shape[2], x.shape[3], x.shape[4])
        out = x1 + x2
        if self.activation is not None:
            out = self.activation(out)
        return out

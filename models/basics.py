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
    1D Discrete Hartley Transform (DHT).

    Args:
        x (torch.Tensor): Input tensor with shape (..., N).

    Returns:
        torch.Tensor: DHT-transformed tensor with shape (..., N).
    """
    return torch.fft.fft(x, dim=-1).real - torch.fft.fft(x, dim=-1).imag

def dht_2d(x: torch.Tensor) -> torch.Tensor:
    """
    2D Discrete Hartley Transform (DHT).

    Args:
        x (torch.Tensor): Input tensor with shape (..., H, W).

    Returns:
        torch.Tensor: DHT-transformed tensor with shape (..., H, W).
    """
    return torch.fft.fftn(x, dim=(-2, -1)).real - torch.fft.fftn(x, dim=(-2, -1)).imag

def dht_3d(x: torch.Tensor) -> torch.Tensor:
    """
    3D Discrete Hartley Transform (DHT).

    Args:
        x (torch.Tensor): Input tensor with shape (..., D, H, W).

    Returns:
        torch.Tensor: DHT-transformed tensor with shape (..., D, H, W).
    """
    return torch.fft.fftn(x, dim=(-3, -2, -1)).real - torch.fft.fftn(x, dim=(-3, -2, -1)).imag

################################################################
# Inverse Discrete Hartley Transforms (IDHT)
################################################################

def idht_1d(X: torch.Tensor) -> torch.Tensor:
    """
    1D Inverse Discrete Hartley Transform (IDHT).

    Since DHT is self-inverse up to a scaling factor, IDHT is equivalent to DHT scaled appropriately.

    Args:
        X (torch.Tensor): 1D DHT-transformed data with shape (..., N).

    Returns:
        torch.Tensor: The original 1D data after applying IDHT.
    """
    N = X.shape[-1]
    return dht_1d(X) / (2 * N)

def idht_2d(X: torch.Tensor) -> torch.Tensor:
    """
    2D Inverse Discrete Hartley Transform (IDHT).

    Args:
        X (torch.Tensor): 2D DHT-transformed data with shape (..., H, W).

    Returns:
        torch.Tensor: The original 2D data after applying IDHT.
    """
    H, W = X.shape[-2], X.shape[-1]
    return dht_2d(X) / (2 * H * W)

def idht_3d(X: torch.Tensor) -> torch.Tensor:
    """
    3D Inverse Discrete Hartley Transform (IDHT).

    Args:
        X (torch.Tensor): 3D DHT-transformed data with shape (..., D, H, W).

    Returns:
        torch.Tensor: The original 3D data after applying IDHT.
    """
    D, H, W = X.shape[-3], X.shape[-2], X.shape[-1]
    return dht_3d(X) / (2 * D * H * W)

################################################################
# Convolutions
################################################################

def compl_mul1d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Performs element-wise multiplication for 1D spectral data.

    Args:
        x1 (torch.Tensor): Tensor of shape [batch, in_channels, modes1].
        x2 (torch.Tensor): Tensor of shape [in_channels, out_channels, modes1].

    Returns:
        torch.Tensor: Tensor of shape [batch, out_channels, modes1].
    """
    # Element-wise multiplication and summation over in_channels
    # Using broadcasting to multiply x1 and x2
    return torch.einsum('bik, iok -> bok', x1, x2)

def compl_mul2d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Performs element-wise multiplication for 2D spectral data.

    Args:
        x1 (torch.Tensor): Tensor of shape [batch, in_channels, modes1, modes2].
        x2 (torch.Tensor): Tensor of shape [in_channels, out_channels, modes1, modes2].

    Returns:
        torch.Tensor: Tensor of shape [batch, out_channels, modes1, modes2].
    """
    return torch.einsum('bijk, oijk -> bojk', x1, x2)

def compl_mul3d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Performs element-wise multiplication for 3D spectral data.

    Args:
        x1 (torch.Tensor): Tensor of shape [batch, in_channels, modes1, modes2, modes3].
        x2 (torch.Tensor): Tensor of shape [in_channels, out_channels, modes1, modes2, modes3].

    Returns:
        torch.Tensor: Tensor of shape [batch, out_channels, modes1, modes2, modes3].
    """
    return torch.einsum('bijkm, oijkm -> bojkm', x1, x2)

################################################################
# Spectral Convolution Layers
################################################################

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = (1 / (in_channels * out_channels))
        # Initialize weights for real-valued DHT coefficients
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1)
        )  # Shape: (in_channels, out_channels, modes1)

    def forward(self, x):
        batchsize = x.shape[0]
        # Perform Discrete Hartley Transform
        x_ht = dht_1d(x)  # Shape: [batch, in_channels, length]
        
        # Apply low-pass filter if needed (optional)
        # x_ht = low_pass_filter(x_ht, cutoff=0.1)

        # Slice the relevant modes
        x_ht_slice = x_ht[:, :, :self.modes1]  # Shape: [batch, in_channels, modes1]
        
        # Perform element-wise multiplication and sum over in_channels
        out_ht = compl_mul1d(x_ht_slice, self.weights1)  # Shape: [batch, out_channels, modes1]
        
        # Create an output tensor with the same shape as x_ht
        out_ht_full = torch.zeros_like(x_ht)
        out_ht_full[:, :, :self.modes1] = out_ht  # Assign the multiplied modes
        
        # Perform Inverse Discrete Hartley Transform
        x_out = idht_1d(out_ht_full)  # Shape: [batch, out_channels, length]
        
        return x_out


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        # Initialize weights for real-valued DHT coefficients
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2)
        )  # Shape: (in_channels, out_channels, modes1, modes2)

    def forward(self, x):
        batchsize = x.shape[0]
        size1, size2 = x.shape[-2], x.shape[-1]
        # Perform Discrete Hartley Transform
        x_ht = dht_2d(x)  # Shape: [batch, in_channels, height, width]
        
        # Apply low-pass filter if needed (optional)
        # x_ht = low_pass_filter(x_ht, cutoff=0.1)

        # Slice the relevant modes
        x_ht_slice = x_ht[:, :, :self.modes1, :self.modes2]  # Shape: [batch, in_channels, modes1, modes2]
        
        # Perform element-wise multiplication and sum over in_channels
        out_ht = compl_mul2d(x_ht_slice, self.weights1)  # Shape: [batch, out_channels, modes1, modes2]
        
        # Create an output tensor with the same shape as x_ht
        out_ht_full = torch.zeros_like(x_ht)
        out_ht_full[:, :, :self.modes1, :self.modes2] = out_ht  # Assign the multiplied modes
        
        # Perform Inverse Discrete Hartley Transform
        x_out = idht_2d(out_ht_full)  # Shape: [batch, out_channels, height, width]
        
        return x_out


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.scale = (1 / (in_channels * out_channels))
        # Initialize weights for real-valued DHT coefficients
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3)
        )  # Shape: (in_channels, out_channels, modes1, modes2, modes3)

    def forward(self, x):
        batchsize = x.shape[0]
        size1, size2, size3 = x.shape[-3], x.shape[-2], x.shape[-1]
        # Perform Discrete Hartley Transform
        x_ht = dht_3d(x)  # Shape: [batch, in_channels, depth, height, width]
        
        # Apply low-pass filter if needed (optional)
        # x_ht = low_pass_filter(x_ht, cutoff=0.1)

        # Slice the relevant modes
        x_ht_slice = x_ht[:, :, :self.modes1, :self.modes2, :self.modes3]  # Shape: [batch, in_channels, modes1, modes2, modes3]
        
        # Perform element-wise multiplication and sum over in_channels
        out_ht = compl_mul3d(x_ht_slice, self.weights1)  # Shape: [batch, out_channels, modes1, modes2, modes3]
        
        # Create an output tensor with the same shape as x_ht
        out_ht_full = torch.zeros_like(x_ht)
        out_ht_full[:, :, :self.modes1, :self.modes2, :self.modes3] = out_ht  # Assign the multiplied modes
        
        # Perform Inverse Discrete Hartley Transform
        x_out = idht_3d(out_ht_full)  # Shape: [batch, out_channels, depth, height, width]
        
        return x_out

################################################################
# FourierBlock
################################################################

class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2=None, modes3=None, activation='tanh'):
        super(FourierBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        if modes2 is None and modes3 is None:
            # 1D Fourier Block
            self.speconv = SpectralConv1d(in_channels, out_channels, modes1)
        elif modes3 is None:
            # 2D Fourier Block
            self.speconv = SpectralConv2d(in_channels, out_channels, modes1, modes2)
        else:
            # 3D Fourier Block
            self.speconv = SpectralConv3d(in_channels, out_channels, modes1, modes2, modes3)
        
        # Linear layer to handle the non-spectral part
        if self.speconv.__class__.__name__ == 'SpectralConv1d':
            self.linear = nn.Conv1d(in_channels, out_channels, 1)
        elif self.speconv.__class__.__name__ == 'SpectralConv2d':
            self.linear = nn.Conv2d(in_channels, out_channels, 1)
        elif self.speconv.__class__.__name__ == 'SpectralConv3d':
            self.linear = nn.Conv3d(in_channels, out_channels, 1)
        else:
            raise ValueError("Unsupported SpectralConv layer.")

        # Define activation
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
        # Spectral convolution
        x1 = self.speconv(x)
        
        # Non-spectral convolution
        if self.speconv.__class__.__name__ == 'SpectralConv1d':
            x2 = self.linear(x)
        elif self.speconv.__class__.__name__ == 'SpectralConv2d':
            x2 = self.linear(x)
        elif self.speconv.__class__.__name__ == 'SpectralConv3d':
            x2 = self.linear(x)
        else:
            raise ValueError("Unsupported SpectralConv layer.")

        # Combine
        out = x1 + x2

        # Apply activation
        if self.activation is not None:
            out = self.activation(out)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch

def get_transform_dims(x: torch.Tensor):
    """
    Determine the transform dimensions based on the input tensor's dimensionality.
    
    - 3D tensor: [batch, channel, length] -> dim=[2]
    - 4D tensor: [batch, channel, height, width] -> dim=[2, 3]
    - 5D tensor: [batch, channel, depth, height, width] -> dim=[2, 3, 4]
    
    Raises:
        ValueError: If tensor dimensionality is not 3, 4, or 5.
    
    Returns:
        List[int]: List of dimensions to perform the transform on.
    """
    if x.dim() == 3:
        return [2]
    elif x.dim() == 4:
        return [2, 3]
    elif x.dim() == 5:
        return [2, 3, 4]
    else:
        raise ValueError(f"Unsupported tensor dimension: {x.dim()}. Supported dimensions are 3, 4, or 5.")

def compl_mul(x1: torch.Tensor, x2: torch.Tensor, num_transform_dims: int) -> torch.Tensor:
    """
    Generalized convolution using torch.einsum for 1D, 2D, and 3D cases.
    
    Args:
        x1 (torch.Tensor): Input tensor with shape [batch, in_channels, ...]
        x2 (torch.Tensor): Kernel tensor with shape [in_channels, out_channels, ...]
        num_transform_dims (int): Number of dimensions to transform (1, 2, or 3)
    
    Returns:
        torch.Tensor: Convolved tensor with shape [batch, out_channels, ...]
    """
    # Define letters for transform dimensions
    letters = ['x', 'y', 'z']
    if num_transform_dims > len(letters):
        raise ValueError(f"Number of transform dimensions {num_transform_dims} exceeds supported letters {letters}.")
    
    transform_letters = ''.join(letters[:num_transform_dims])
    
    # Construct einsum equation
    # x1: 'b i' + transform_letters
    # x2: 'i o' + transform_letters
    # output: 'b o' + transform_letters
    x1_subscript = 'bi' + transform_letters
    x2_subscript = 'io' + transform_letters
    output_subscript = 'bo' + transform_letters
    equation = f'{x1_subscript},{x2_subscript}->{output_subscript}'
    
    return torch.einsum(equation, x1, x2)

def flip_periodic(x: torch.Tensor) -> torch.Tensor:
    """
    Perform a periodic flip of the tensor along the transform dimensions.
    
    For each set of transform dimensions, concatenate the first element with the flipped remaining elements.
    
    Args:
        x (torch.Tensor): Input tensor.
        
    Returns:
        torch.Tensor: Periodically flipped tensor.
    """
    transform_dims = get_transform_dims(x)
    num_transform_dims = len(transform_dims)
    
    # Ensure transform dimensions are the last N dimensions
    expected_transform_dims = list(range(x.dim() - num_transform_dims, x.dim()))
    if transform_dims != expected_transform_dims:
        raise NotImplementedError("Transform dimensions must be the last N dimensions of the tensor.")
    
    # Compute the total number of elements along transform dims
    # Flatten the transform dims into one dimension
    batch_shape = x.shape[:-num_transform_dims]
    flattened_dim = 1
    for dim in transform_dims:
        flattened_dim *= x.shape[dim]
    x_flat = x.view(*batch_shape, flattened_dim)
    
    # Split the first element and the remaining
    first = x_flat[..., :1]  # Shape: (..., 1)
    remaining = x_flat[..., 1:]  # Shape: (..., flattened_dim - 1)
    
    # Flip the remaining elements
    remaining_flipped = torch.flip(remaining, dims=[-1])
    
    # Concatenate first and flipped remaining
    Z_flat = torch.cat([first, remaining_flipped], dim=-1)
    
    # Reshape back to original shape
    Z = Z_flat.view_as(x)
    return Z

def dht(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Discrete Hartley Transform (DHT) of the input tensor.
    
    Args:
        x (torch.Tensor): Input tensor.
        
    Returns:
        torch.Tensor: DHT of the input tensor.
    """
    transform_dims = get_transform_dims(x)
    X = torch.fft.fftn(x, dim=transform_dims)
    X = X.real - X.imag
    return X

def idht(X: torch.Tensor) -> torch.Tensor:
    """
    Compute the Inverse Discrete Hartley Transform (IDHT) of the input tensor.
    
    Since the DHT is involutory, IDHT(x) = (1/n) * DHT(DHT(x))
    
    Args:
        X (torch.Tensor): Input tensor in the DHT domain.
        
    Returns:
        torch.Tensor: Inverse DHT of the input tensor.
    """
    transform_dims = get_transform_dims(X)
    # Compute the product of sizes along transform dims
    n = 1
    for dim in transform_dims:
        n *= X.shape[dim]
    x = dht(X)
    x = x / n
    return x

def dht_conv(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the DHT of the convolution of two tensors using the convolution theorem and torch.einsum.
    
    Args:
        x (torch.Tensor): First input tensor with shape [batch, in_channels, ...]
        y (torch.Tensor): Second input tensor with shape [in_channels, out_channels, ...]
        
    Returns:
        torch.Tensor: DHT of the convolution of x and y.
        
    Raises:
        AssertionError: If x and y do not have the same shape except for the out_channels dimension.
    """
    # Ensure x and y have compatible shapes
    # x: [batch, in_channels, ...]
    # y: [in_channels, out_channels, ...]
    assert x.dim() == y.dim(), "x and y must have the same number of dimensions."
    assert y.shape[0] == x.shape[1], "y's in_channels must match x's in_channels."
    num_transform_dims = x.dim() - 2  # Exclude batch and channel dimensions
    
    # Compute DHTs
    X = dht(x)  # [batch, in_channels, ...]
    Y = dht(y)  # [in_channels, out_channels, ...]
    
    # Compute flipped versions
    Xflip = flip_periodic(X)
    Yflip = flip_periodic(Y)
    
    # Compute even and odd components
    Yeven = 0.5 * (Y + Yflip)
    Yodd  = 0.5 * (Y - Yflip)
    
    # Perform convolution using the generalized compl_mul with torch.einsum
    # Z = X * Yeven + Xflip * Yodd
    # We'll use compl_mul for the tensor contractions
    
    # First term: compl_mul(X, Yeven)
    term1 = compl_mul(X, Yeven, num_transform_dims)
    
    # Second term: compl_mul(Xflip, Yodd)
    term2 = compl_mul(Xflip, Yodd, num_transform_dims)
    
    # Combine terms
    Z = term1 + term2  # [batch, out_channels, ...]
    
    return Z

def conv(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the convolution of two tensors using the DHT.
    
    Args:
        x (torch.Tensor): First input tensor with shape [batch, in_channels, ...]
        y (torch.Tensor): Second input tensor with shape [in_channels, out_channels, ...]
        
    Returns:
        torch.Tensor: Convolution of x and y with shape [batch, out_channels, ...]
    """
    Z = dht_conv(x, y)
    z = idht(Z)
    return z

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
        out_ht[:, :, :self.modes1] = conv(x_ht[:, :, :self.modes1], self.weights1)

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
        out_ht[:, :, :self.modes1, :self.modes2] = conv(
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
        out_ht[:, :, :self.modes1, :self.modes2, :self.modes3] = conv(
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

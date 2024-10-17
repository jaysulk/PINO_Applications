import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################
# 1D Functions
################################################################

def compl_mul_1d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    1D convolution using torch.einsum.

    Args:
        x1 (torch.Tensor): Input tensor with shape [batch, in_channels, length]
        x2 (torch.Tensor): Kernel tensor with shape [in_channels, out_channels, modes1]

    Returns:
        torch.Tensor: Convolved tensor with shape [batch, out_channels, length]
    """
    # Define the einsum equation for 1D
    equation = 'bix,iox->box'
    return torch.einsum(equation, x1, x2)

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

    # Prepare slicing indices
    idx_first = [slice(None)] * Z.dim()
    idx_first[dim] = 0  # Select the first element along 'dim'

    idx_remaining = [slice(None)] * Z.dim()
    idx_remaining[dim] = slice(1, None)  # Select elements from index 1 onwards

    # Extract the first element
    first = Z[tuple(idx_first)].unsqueeze(dim)  # Shape: [batch, channels, 1]

    if Z.size(dim) > 1:
        # Select all elements from index 1 onwards and flip them
        remaining = Z[tuple(idx_remaining)].flip(dims=[dim])
        # Concatenate first and flipped remaining along the current dimension
        Z = torch.cat([first, remaining], dim=dim)
    else:
        # If there's only one element, no flipping needed
        Z = first

    return Z

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

def idht_1d(X: torch.Tensor) -> torch.Tensor:
    """
    Compute the Inverse 1D Discrete Hartley Transform (IDHT) of the input tensor.

    Since the DHT is involutory, IDHT(x) = (1/n) * DHT(DHT(x))

    Args:
        X (torch.Tensor): Input tensor in the DHT domain with shape [batch, channels, length].

    Returns:
        torch.Tensor: Inverse DHT of the input tensor.
    """
    transform_dims = [2]  # Length dimension
    n = X.shape[2]  # Length
    x = dht_1d(X)
    x = x / n
    return x

def dht_conv_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the DHT of the convolution of two 1D tensors using the convolution theorem.

    Args:
        x (torch.Tensor): First input tensor with shape [batch, in_channels, length]
        y (torch.Tensor): Second input tensor with shape [in_channels, out_channels, modes1]

    Returns:
        torch.Tensor: DHT of the convolution of x and y.
    """
    # Ensure x and y have compatible shapes
    assert x.dim() == y.dim() + 1, "For 1D, x should have one more dimension than y."
    assert y.shape[0] == x.shape[1], "y's in_channels must match x's in_channels."

    # Compute DHTs
    X = dht_1d(x)  # [batch, in_channels, length]
    Y = dht_1d(y)  # [in_channels, out_channels, length]

    # Compute flipped versions
    Xflip = flip_periodic_1d(X)
    Yflip = flip_periodic_1d(Y)

    # Compute even and odd components
    Yeven = 0.5 * (Y + Yflip)
    Yodd  = 0.5 * (Y - Yflip)

    # Perform convolution using compl_mul
    term1 = compl_mul_1d(X, Yeven)
    term2 = compl_mul_1d(Xflip, Yodd)

    # Combine terms
    Z = term1 + term2  # [batch, out_channels, length]

    return Z

def conv_1d(x_ht: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Perform 1D convolution in the Hartley domain.

    Args:
        x_ht (torch.Tensor): Hartley-transformed input tensor [batch, in_channels, modes1]
        weights (torch.Tensor): Hartley-transformed weights [in_channels, out_channels, modes1]

    Returns:
        torch.Tensor: Convolved tensor in the Hartley domain [batch, out_channels, modes1]
    """
    return compl_mul_1d(x_ht, weights)

################################################################
# 2D Functions
################################################################

def compl_mul_2d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    2D convolution using torch.einsum.

    Args:
        x1 (torch.Tensor): Input tensor with shape [batch, in_channels, height, width]
        x2 (torch.Tensor): Kernel tensor with shape [in_channels, out_channels, modes1, modes2]

    Returns:
        torch.Tensor: Convolved tensor with shape [batch, out_channels, height, width]
    """
    # Define the einsum equation for 2D
    equation = 'bixy,ioxy->boxy'
    return torch.einsum(equation, x1, x2)

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

        # Prepare slicing indices
        idx_first = [slice(None)] * Z.dim()
        idx_first[dim] = 0  # Select the first element along 'dim'

        idx_remaining = [slice(None)] * Z.dim()
        idx_remaining[dim] = slice(1, None)  # Select elements from index 1 onwards

        # Extract the first element
        first = Z[tuple(idx_first)].unsqueeze(dim)  # Shape: same as x with dim size=1

        if Z.size(dim) > 1:
            # Select all elements from index 1 onwards and flip them
            remaining = Z[tuple(idx_remaining)].flip(dims=[dim])
            # Concatenate first and flipped remaining along the current dimension
            Z = torch.cat([first, remaining], dim=dim)
        else:
            # If there's only one element, no flipping needed
            Z = first

    return Z

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

def idht_2d(X: torch.Tensor) -> torch.Tensor:
    """
    Compute the Inverse 2D Discrete Hartley Transform (IDHT) of the input tensor.

    Since the DHT is involutory, IDHT(x) = (1/n) * DHT(DHT(x))

    Args:
        X (torch.Tensor): Input tensor in the DHT domain with shape [batch, channels, height, width].

    Returns:
        torch.Tensor: Inverse DHT of the input tensor.
    """
    transform_dims = [2, 3]  # Height and Width dimensions
    n = X.shape[2] * X.shape[3]  # Height * Width
    x = dht_2d(X)
    x = x / n
    return x

def dht_conv_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the DHT of the convolution of two 2D tensors using the convolution theorem.

    Args:
        x (torch.Tensor): First input tensor with shape [batch, in_channels, height, width]
        y (torch.Tensor): Second input tensor with shape [in_channels, out_channels, modes1, modes2]

    Returns:
        torch.Tensor: DHT of the convolution of x and y.
    """
    # Ensure x and y have compatible shapes
    assert x.dim() == y.dim(), "x and y must have the same number of dimensions for 2D."
    assert y.shape[0] == x.shape[1], "y's in_channels must match x's in_channels."

    # Compute DHTs
    X = dht_2d(x)  # [batch, in_channels, height, width]
    Y = dht_2d(y)  # [in_channels, out_channels, height, width]

    # Compute flipped versions
    Xflip = flip_periodic_2d(X)
    Yflip = flip_periodic_2d(Y)

    # Compute even and odd components
    Yeven = 0.5 * (Y + Yflip)
    Yodd  = 0.5 * (Y - Yflip)

    # Perform convolution using compl_mul
    term1 = compl_mul_2d(X, Yeven)
    term2 = compl_mul_2d(Xflip, Yodd)

    # Combine terms
    Z = term1 + term2  # [batch, out_channels, height, width]

    return Z

def conv_2d(x_ht: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Perform 2D convolution in the Hartley domain.

    Args:
        x_ht (torch.Tensor): Hartley-transformed input tensor [batch, in_channels, modes1, modes2]
        weights (torch.Tensor): Hartley-transformed weights [in_channels, out_channels, modes1, modes2]

    Returns:
        torch.Tensor: Convolved tensor in the Hartley domain [batch, out_channels, modes1, modes2]
    """
    return compl_mul_2d(x_ht, weights)

################################################################
# 3D Functions
################################################################

def compl_mul_3d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    3D convolution using torch.einsum.

    Args:
        x1 (torch.Tensor): Input tensor with shape [batch, in_channels, depth, height, width]
        x2 (torch.Tensor): Kernel tensor with shape [in_channels, out_channels, modes1, modes2, modes3]

    Returns:
        torch.Tensor: Convolved tensor with shape [batch, out_channels, depth, height, width]
    """
    # Define the einsum equation for 3D
    equation = 'bixyz,ioxyz->boxyz'
    return torch.einsum(equation, x1, x2)

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

        # Prepare slicing indices
        idx_first = [slice(None)] * Z.dim()
        idx_first[dim] = 0  # Select the first element along 'dim'

        idx_remaining = [slice(None)] * Z.dim()
        idx_remaining[dim] = slice(1, None)  # Select elements from index 1 onwards

        # Extract the first element
        first = Z[tuple(idx_first)].unsqueeze(dim)  # Shape: same as x with dim size=1

        if Z.size(dim) > 1:
            # Select all elements from index 1 onwards and flip them
            remaining = Z[tuple(idx_remaining)].flip(dims=[dim])
            # Concatenate first and flipped remaining along the current dimension
            Z = torch.cat([first, remaining], dim=dim)
        else:
            # If there's only one element, no flipping needed
            Z = first

    return Z

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

def idht_3d(X: torch.Tensor) -> torch.Tensor:
    """
    Compute the Inverse 3D Discrete Hartley Transform (IDHT) of the input tensor.

    Since the DHT is involutory, IDHT(x) = (1/n) * DHT(DHT(x))

    Args:
        X (torch.Tensor): Input tensor in the DHT domain with shape [batch, channels, depth, height, width].

    Returns:
        torch.Tensor: Inverse DHT of the input tensor.
    """
    transform_dims = [2, 3, 4]  # Depth, Height, and Width dimensions
    n = X.shape[2] * X.shape[3] * X.shape[4]  # Depth * Height * Width
    x = dht_3d(X)
    x = x / n
    return x

def dht_conv_3d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the DHT of the convolution of two 3D tensors using the convolution theorem.

    Args:
        x (torch.Tensor): First input tensor with shape [batch, in_channels, depth, height, width]
        y (torch.Tensor): Second input tensor with shape [in_channels, out_channels, modes1, modes2, modes3]

    Returns:
        torch.Tensor: DHT of the convolution of x and y.
    """
    # Ensure x and y have compatible shapes
    assert x.dim() == y.dim(), "x and y must have the same number of dimensions for 3D."
    assert y.shape[0] == x.shape[1], "y's in_channels must match x's in_channels."

    # Compute DHTs
    X = dht_3d(x)  # [batch, in_channels, depth, height, width]
    Y = dht_3d(y)  # [in_channels, out_channels, depth, height, width]

    # Compute flipped versions
    Xflip = flip_periodic_3d(X)
    Yflip = flip_periodic_3d(Y)

    # Compute even and odd components
    Yeven = 0.5 * (Y + Yflip)
    Yodd  = 0.5 * (Y - Yflip)

    # Perform convolution using compl_mul
    term1 = compl_mul_3d(X, Yeven)
    term2 = compl_mul_3d(Xflip, Yodd)

    # Combine terms
    Z = term1 + term2  # [batch, out_channels, depth, height, width]

    return Z

def conv_3d(x_ht: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Perform 3D convolution in the Hartley domain.

    Args:
        x_ht (torch.Tensor): Hartley-transformed input tensor [batch, in_channels, modes1, modes2, modes3]
        weights (torch.Tensor): Hartley-transformed weights [in_channels, out_channels, modes1, modes2, modes3]

    Returns:
        torch.Tensor: Convolved tensor in the Hartley domain [batch, out_channels, modes1, modes2, modes3]
    """
    return compl_mul_3d(x_ht, weights)

################################################################
# Spectral Convolution Layers with Adaptive Basis Refinement
################################################################

################################################################
# 1D Hartley convolution layer with adaptive basis refinement
################################################################

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Hartley layer with adaptive basis refinement. It does DHT, linear transform,
        computes reconstruction error, and applies additional transforms in regions with high error.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Ensure in_channels equals out_channels for error computation
        assert self.in_channels == self.out_channels, "For adaptive refinement, in_channels must equal out_channels in SpectralConv1d."

        # Number of Hartley modes to multiply
        self.modes1 = modes1

        # Additional modes for refinement
        self.refine_modes1 = modes1  # Hardcoded as the same number; can be changed if needed

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1))
        
        # Weights for refined modes
        self.refine_weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.refine_modes1))
        
        # Error threshold for refinement (hardcoded)
        self.error_threshold = 0.1  # Adjust as needed

    def forward(self, x):
        batchsize = x.shape[0]

        # Compute Hartley coefficients
        x_ht = dht_1d(x)  # [batch, in_channels, length]

        # Multiply relevant Hartley modes
        out_ht = torch.zeros(batchsize, self.out_channels, x.size(-1), device=x.device, dtype=x.dtype)
        out_ht[:, :, :self.modes1] = conv_1d(x_ht[:, :, :self.modes1], self.weights1)

        # Inverse Hartley to get back to physical space
        x_reconstructed = idht_1d(out_ht)  # [batch, out_channels, length]

        # Compute reconstruction error
        # Since in_channels == out_channels, shapes match
        error = torch.abs(x - x_reconstructed)  # [batch, out_channels, length]

        # Create a mask where error exceeds the threshold
        mask = (error > self.error_threshold).float()  # [batch, out_channels, length]

        # Apply mask to the input for refinement
        x_refine = x * mask  # [batch, in_channels, length]

        # Compute Hartley coefficients for refined regions
        x_refine_ht = dht_1d(x_refine)  # [batch, in_channels, length]

        # Multiply relevant Hartley modes for refinement
        refine_out_ht = torch.zeros(batchsize, self.out_channels, x.size(-1), device=x.device, dtype=x.dtype)
        refine_out_ht[:, :, :self.refine_modes1] = conv_1d(x_refine_ht[:, :, :self.refine_modes1], self.refine_weights1)

        # Combine the refined output with the initial output
        out_ht += refine_out_ht

        # Return to physical space
        x_final = idht_1d(out_ht)  # [batch, out_channels, length]

        return x_final

################################################################
# 2D Hartley convolution layer with adaptive basis refinement
################################################################

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # Ensure in_channels equals out_channels for error computation
        assert self.in_channels == self.out_channels, "For adaptive refinement, in_channels must equal out_channels in SpectralConv2d."

        # Additional modes for refinement
        self.refine_modes1 = modes1  # Hardcoded as the same number; can be changed if needed
        self.refine_modes2 = modes2  # Hardcoded as the same number; can be changed if needed

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        
        # Weights for refined modes
        self.refine_weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.refine_modes1, self.refine_modes2))
        
        # Error threshold for refinement (hardcoded)
        self.error_threshold = 0.1  # Adjust as needed

    def forward(self, x):
        batchsize = x.shape[0]
        size1 = x.shape[-2]
        size2 = x.shape[-1]

        # Compute Hartley coefficients
        x_ht = dht_2d(x)  # [batch, in_channels, height, width]

        # Multiply relevant Hartley modes
        out_ht = torch.zeros(batchsize, self.out_channels, size1, size2, device=x.device, dtype=x.dtype)
        out_ht[:, :, :self.modes1, :self.modes2] = conv_2d(
            x_ht[:, :, :self.modes1, :self.modes2], self.weights1)

        # Inverse Hartley to get back to physical space
        x_reconstructed = idht_2d(out_ht)  # [batch, out_channels, height, width]

        # Compute reconstruction error
        # Since in_channels == out_channels, shapes match
        error = torch.abs(x - x_reconstructed)  # [batch, out_channels, height, width]

        # Create a mask where error exceeds the threshold
        mask = (error > self.error_threshold).float()  # [batch, out_channels, height, width]

        # Apply mask to the input for refinement
        x_refine = x * mask  # [batch, in_channels, height, width]

        # Compute Hartley coefficients for refined regions
        x_refine_ht = dht_2d(x_refine)  # [batch, in_channels, height, width]

        # Multiply relevant Hartley modes for refinement
        refine_out_ht = torch.zeros(batchsize, self.out_channels, size1, size2, device=x.device, dtype=x.dtype)
        refine_out_ht[:, :, :self.refine_modes1, :self.refine_modes2] = conv_2d(
            x_refine_ht[:, :, :self.refine_modes1, :self.refine_modes2], self.refine_weights1)

        # Combine the refined output with the initial output
        out_ht += refine_out_ht

        # Return to physical space
        x_final = idht_2d(out_ht)  # [batch, out_channels, height, width]

        return x_final

################################################################
# 3D Hartley convolution layer with adaptive basis refinement
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Hartley modes to multiply
        self.modes2 = modes2
        self.modes3 = modes3

        # Ensure in_channels equals out_channels for error computation
        assert self.in_channels == self.out_channels, "For adaptive refinement, in_channels must equal out_channels in SpectralConv3d."

        # Additional modes for refinement
        self.refine_modes1 = modes1  # Hardcoded as the same number; can be changed if needed
        self.refine_modes2 = modes2
        self.refine_modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(
            in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        
        # Weights for refined modes
        self.refine_weights1 = nn.Parameter(self.scale * torch.rand(
            in_channels, out_channels, self.refine_modes1, self.refine_modes2, self.refine_modes3))
        
        # Error threshold for refinement (hardcoded)
        self.error_threshold = 0.1  # Adjust as needed

    def forward(self, x):
        batchsize = x.shape[0]
        size1 = x.shape[-3]
        size2 = x.shape[-2]
        size3 = x.shape[-1]

        # Compute Hartley coefficients
        x_ht = dht_3d(x)  # [batch, in_channels, depth, height, width]

        # Multiply relevant Hartley modes
        out_ht = torch.zeros(batchsize, self.out_channels, size1, size2, size3, device=x.device, dtype=x.dtype)
        out_ht[:, :, :self.modes1, :self.modes2, :self.modes3] = conv_3d(
            x_ht[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)

        # Inverse Hartley to get back to physical space
        x_reconstructed = idht_3d(out_ht)  # [batch, out_channels, depth, height, width]

        # Compute reconstruction error
        # Since in_channels == out_channels, shapes match
        error = torch.abs(x - x_reconstructed)  # [batch, out_channels, depth, height, width]

        # Create a mask where error exceeds the threshold
        mask = (error > self.error_threshold).float()  # [batch, out_channels, depth, height, width]

        # Apply mask to the input for refinement
        x_refine = x * mask  # [batch, in_channels, depth, height, width]

        # Compute Hartley coefficients for refined regions
        x_refine_ht = dht_3d(x_refine)  # [batch, in_channels, depth, height, width]

        # Multiply relevant Hartley modes for refinement
        refine_out_ht = torch.zeros(batchsize, self.out_channels, size1, size2, size3, device=x.device, dtype=x.dtype)
        refine_out_ht[:, :, :self.refine_modes1, :self.refine_modes2, :self.refine_modes3] = conv_3d(
            x_refine_ht[:, :, :self.refine_modes1, :self.refine_modes2, :self.refine_modes3], self.refine_weights1)

        # Combine the refined output with the initial output
        out_ht += refine_out_ht

        # Return to physical space
        x_final = idht_3d(out_ht)  # [batch, out_channels, depth, height, width]

        return x_final

################################################################
# FourierBlock (Using SpectralConv1d, SpectralConv2d, SpectralConv3d)
################################################################

class FourierBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, activation='tanh'):
        super(FourierBlock1d, self).__init__()

        # Ensure in_channels equals out_channels for adaptive refinement
        assert in_channels == out_channels, "For adaptive refinement, in_channels must equal out_channels in FourierBlock1d."

        # Spectral convolution layer (using 1D Hartley transform with adaptive refinement)
        self.speconv = SpectralConv1d(in_channels, out_channels, modes1)

        # Linear layer applied across the channel dimension
        self.linear = nn.Conv1d(in_channels, out_channels, kernel_size=1)

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
        Input x: (batchsize, in_channels, length)
        '''
        x1 = self.speconv(x)  # [batch, out_channels, length]
        x2 = self.linear(x)   # [batch, out_channels, length]
        x = x1 + x2           # [batch, out_channels, length]

        if self.activation is not None:
            x = self.activation(x)

        return x

class FourierBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, activation='tanh'):
        super(FourierBlock2d, self).__init__()

        # Ensure in_channels equals out_channels for adaptive refinement
        assert in_channels == out_channels, "For adaptive refinement, in_channels must equal out_channels in FourierBlock2d."

        # Spectral convolution layer (using 2D Hartley transform with adaptive refinement)
        self.speconv = SpectralConv2d(in_channels, out_channels, modes1, modes2)

        # Linear layer applied across the channel dimension
        self.linear = nn.Conv2d(in_channels, out_channels, kernel_size=1)

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
        Input x: (batchsize, in_channels, height, width)
        '''
        x1 = self.speconv(x)  # [batch, out_channels, height, width]
        x2 = self.linear(x)   # [batch, out_channels, height, width]
        x = x1 + x2           # [batch, out_channels, height, width]

        if self.activation is not None:
            x = self.activation(x)

        return x

class FourierBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, activation='tanh'):
        super(FourierBlock3d, self).__init__()

        # Ensure in_channels equals out_channels for adaptive refinement
        assert in_channels == out_channels, "For adaptive refinement, in_channels must equal out_channels in FourierBlock3d."

        # Spectral convolution layer (using 3D Hartley transform with adaptive refinement)
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
        Input x: (batchsize, in_channels, depth, height, width)
        '''
        x1 = self.speconv(x)  # [batch, out_channels, depth, height, width]
        x2 = self.linear(x)   # [batch, out_channels, depth, height, width]
        x = x1 + x2           # [batch, out_channels, depth, height, width]

        if self.activation is not None:
            x = self.activation(x)

        return x

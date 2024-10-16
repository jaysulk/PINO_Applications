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
# Spectral Convolution Layers with Adaptive Refinement
################################################################

################################################################
# 1D Hartley convolution layer with Adaptive Refinement
################################################################

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, refine_threshold=0.1, refine_factor=2):
        super(SpectralConv1d, self).__init__()

        """
        1D Hartley layer with adaptive basis refinement. It does DHT, linear transform, and Inverse DHT.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            modes1 (int): Number of Hartley modes to multiply.
            refine_threshold (float): Error threshold to trigger refinement.
            refine_factor (int): Factor by which to increase the number of modes upon refinement.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.refine_threshold = refine_threshold
        self.refine_factor = refine_factor

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1))

        # Additional spectral weights for refinement
        self.refined_modes1 = self.modes1 * self.refine_factor
        self.weights_refined = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.refined_modes1))

        # Linear layer applied across the channel dimension
        self.linear = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, target=None):
        """
        Forward pass with optional adaptive basis refinement.

        Args:
            x (torch.Tensor): Input tensor [batch, in_channels, length]
            target (torch.Tensor, optional): Target tensor for error computation [batch, out_channels, length]

        Returns:
            torch.Tensor: Output tensor [batch, out_channels, length]
        """
        batchsize = x.shape[0]
        length = x.shape[-1]

        # Compute Hartley coefficients
        x_ht = dht_1d(x)  # [batch, in_channels, length]

        # Perform initial spectral convolution
        out_ht = torch.zeros(batchsize, self.out_channels, length, device=x.device, dtype=x.dtype)
        out_ht[:, :, :self.modes1] = conv_1d(x_ht[:, :, :self.modes1], self.weights1)

        # Adaptive Refinement
        if target is not None:
            # Transform back to physical space
            x_out = idht_1d(out_ht)  # [batch, out_channels, length]

            # Compute error
            error = torch.abs(target - x_out)  # Absolute error
            error_mean = error.mean(dim=1, keepdim=True)  # Mean error per sample

            # Identify high-error regions
            high_error_mask = (error_mean > self.refine_threshold).float()  # [batch, 1, length]

            # Check if any high-error regions exist
            if high_error_mask.sum() > 0:
                # Perform refined spectral convolution
                refined_out_ht = torch.zeros(batchsize, self.out_channels, length, device=x.device, dtype=x.dtype)
                # Ensure we don't exceed the length when modes are refined
                refined_modes = min(self.refined_modes1, length)
                refined_out_ht[:, :, :refined_modes] = conv_1d(
                    x_ht[:, :, :refined_modes], self.weights_refined[:, :, :refined_modes])

                # Apply mask to combine refined output
                out_ht = out_ht + refined_out_ht * high_error_mask

        # Transform back to physical space
        x_out = idht_1d(out_ht)  # [batch, out_channels, length]

        # Apply linear residual connection
        x_linear = self.linear(x)
        x = x_out + x_linear  # [batch, out_channels, length]

        return x

################################################################
# 2D Hartley convolution layer with Adaptive Refinement
################################################################

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, refine_threshold=0.1, refine_factor=2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.refine_threshold = refine_threshold
        self.refine_factor = refine_factor

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))

        # Additional spectral weights for refinement
        self.refined_modes1 = self.modes1 * self.refine_factor
        self.refined_modes2 = self.modes2 * self.refine_factor
        self.weights_refined = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.refined_modes1, self.refined_modes2))

        # Linear layer applied across the channel dimension
        self.linear = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, target=None):
        """
        Forward pass with optional adaptive basis refinement.

        Args:
            x (torch.Tensor): Input tensor [batch, in_channels, height, width]
            target (torch.Tensor, optional): Target tensor for error computation [batch, out_channels, height, width]

        Returns:
            torch.Tensor: Output tensor [batch, out_channels, height, width]
        """
        batchsize = x.shape[0]
        height = x.shape[-2]
        width = x.shape[-1]

        # Compute Hartley coefficients
        x_ht = dht_2d(x)  # [batch, in_channels, height, width]

        # Perform initial spectral convolution
        out_ht = torch.zeros(batchsize, self.out_channels, height, width, device=x.device, dtype=x.dtype)
        out_ht[:, :, :self.modes1, :self.modes2] = conv_2d(x_ht[:, :, :self.modes1, :self.modes2], self.weights1)

        # Adaptive Refinement
        if target is not None:
            # Transform back to physical space
            x_out = idht_2d(out_ht)  # [batch, out_channels, height, width]

            # Compute error
            error = torch.abs(target - x_out)  # Absolute error
            error_mean = error.mean(dim=1, keepdim=True)  # Mean error per sample

            # Identify high-error regions
            high_error_mask = (error_mean > self.refine_threshold).float()  # [batch, 1, height, width]

            # Check if any high-error regions exist
            if high_error_mask.sum() > 0:
                # Perform refined spectral convolution
                refined_out_ht = torch.zeros(batchsize, self.out_channels, height, width, device=x.device, dtype=x.dtype)
                # Ensure we don't exceed the dimensions when modes are refined
                refined_modes1 = min(self.refined_modes1, height)
                refined_modes2 = min(self.refined_modes2, width)
                refined_out_ht[:, :, :refined_modes1, :refined_modes2] = conv_2d(
                    x_ht[:, :, :refined_modes1, :refined_modes2], 
                    self.weights_refined[:, :, :refined_modes1, :refined_modes2])

                # Apply mask to combine refined output
                out_ht = out_ht + refined_out_ht * high_error_mask

        # Transform back to physical space
        x_out = idht_2d(out_ht)  # [batch, out_channels, height, width]

        # Apply linear residual connection
        x_linear = self.linear(x)
        x = x_out + x_linear  # [batch, out_channels, height, width]

        return x

################################################################
# 3D Hartley convolution layer with Adaptive Refinement
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, refine_threshold=0.1, refine_factor=2):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.refine_threshold = refine_threshold
        self.refine_factor = refine_factor

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))

        # Additional spectral weights for refinement
        self.refined_modes1 = self.modes1 * self.refine_factor
        self.refined_modes2 = self.modes2 * self.refine_factor
        self.refined_modes3 = self.modes3 * self.refine_factor
        self.weights_refined = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.refined_modes1, self.refined_modes2, self.refined_modes3))

        # Linear layer applied across the channel dimension
        self.linear = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, target=None):
        """
        Forward pass with optional adaptive basis refinement.

        Args:
            x (torch.Tensor): Input tensor [batch, in_channels, depth, height, width]
            target (torch.Tensor, optional): Target tensor for error computation [batch, out_channels, depth, height, width]

        Returns:
            torch.Tensor: Output tensor [batch, out_channels, depth, height, width]
        """
        batchsize = x.shape[0]
        depth = x.shape[-3]
        height = x.shape[-2]
        width = x.shape[-1]

        # Compute Hartley coefficients
        x_ht = dht_3d(x)  # [batch, in_channels, depth, height, width]

        # Perform initial spectral convolution
        out_ht = torch.zeros(batchsize, self.out_channels, depth, height, width, device=x.device, dtype=x.dtype)
        out_ht[:, :, :self.modes1, :self.modes2, :self.modes3] = conv_3d(
            x_ht[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)

        # Adaptive Refinement
        if target is not None:
            # Transform back to physical space
            x_out = idht_3d(out_ht)  # [batch, out_channels, depth, height, width]

            # Compute error
            error = torch.abs(target - x_out)  # Absolute error
            error_mean = error.mean(dim=1, keepdim=True)  # Mean error per sample

            # Identify high-error regions
            high_error_mask = (error_mean > self.refine_threshold).float()  # [batch, 1, depth, height, width]

            # Check if any high-error regions exist
            if high_error_mask.sum() > 0:
                # Perform refined spectral convolution
                refined_out_ht = torch.zeros(batchsize, self.out_channels, depth, height, width, device=x.device, dtype=x.dtype)
                # Ensure we don't exceed the dimensions when modes are refined
                refined_modes1 = min(self.refined_modes1, depth)
                refined_modes2 = min(self.refined_modes2, height)
                refined_modes3 = min(self.refined_modes3, width)
                refined_out_ht[:, :, :refined_modes1, :refined_modes2, :refined_modes3] = conv_3d(
                    x_ht[:, :, :refined_modes1, :refined_modes2, :refined_modes3], 
                    self.weights_refined[:, :, :refined_modes1, :refined_modes2, :refined_modes3])

                # Apply mask to combine refined output
                out_ht = out_ht + refined_out_ht * high_error_mask

        # Transform back to physical space
        x_out = idht_3d(out_ht)  # [batch, out_channels, depth, height, width]

        # Apply linear residual connection
        x_linear = self.linear(x)
        x = x_out + x_linear  # [batch, out_channels, depth, height, width]

        return x

################################################################
# FourierBlock (Using SpectralConv1d, SpectralConv2d, SpectralConv3d)
################################################################

class FourierBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, refine_threshold=0.1, refine_factor=2, activation='tanh'):
        super(FourierBlock1d, self).__init__()

        # Spectral convolution layer (using 1D Hartley transform with adaptive refinement)
        self.speconv = SpectralConv1d(in_channels, out_channels, modes1, refine_threshold, refine_factor)

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

    def forward(self, x, target=None):
        '''
        Input x: (batchsize, in_channels, length)
        '''
        x1 = self.speconv(x, target)
        x2 = self.linear(x)
        x = x1 + x2

        if self.activation is not None:
            x = self.activation(x)

        return x

class FourierBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, refine_threshold=0.1, refine_factor=2, activation='tanh'):
        super(FourierBlock2d, self).__init__()

        # Spectral convolution layer (using 2D Hartley transform with adaptive refinement)
        self.speconv = SpectralConv2d(in_channels, out_channels, modes1, modes2, refine_threshold, refine_factor)

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

    def forward(self, x, target=None):
        '''
        Input x: (batchsize, in_channels, height, width)
        '''
        x1 = self.speconv(x, target)
        x2 = self.linear(x)
        x = x1 + x2

        if self.activation is not None:
            x = self.activation(x)

        return x

class FourierBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, refine_threshold=0.1, refine_factor=2, activation='tanh'):
        super(FourierBlock3d, self).__init__()

        # Spectral convolution layer (using 3D Hartley transform with adaptive refinement)
        self.speconv = SpectralConv3d(in_channels, out_channels, modes1, modes2, modes3, refine_threshold, refine_factor)

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

    def forward(self, x, target=None):
        '''
        Input x: (batchsize, in_channels, depth, height, width)
        '''
        x1 = self.speconv(x, target)
        x2 = self.linear(x)
        x = x1 + x2

        if self.activation is not None:
            x = self.activation(x)

        return x

################################################################
# FourierNet (Example Neural Network using FourierBlocks)
################################################################

class FourierNet1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, refine_threshold=0.1, refine_factor=2, activation='tanh'):
        super(FourierNet1d, self).__init__()

        self.block1 = FourierBlock1d(in_channels, 16, modes1, refine_threshold, refine_factor, activation)
        self.block2 = FourierBlock1d(16, 32, modes1, refine_threshold, refine_factor, activation)
        self.block3 = FourierBlock1d(32, out_channels, modes1, refine_threshold, refine_factor, activation)

    def forward(self, x, targets=None):
        '''
        Args:
            x (torch.Tensor): Input tensor [batch, in_channels, length]
            targets (list of torch.Tensor, optional): List of target tensors corresponding to each block.

        Returns:
            torch.Tensor: Output tensor [batch, out_channels, length]
        '''
        if targets is not None and len(targets) != 3:
            raise ValueError("targets must be a list of three tensors corresponding to each FourierBlock1d")

        if targets is not None:
            x = self.block1(x, targets=targets[0])
            x = self.block2(x, targets=targets[1])
            x = self.block3(x, targets=targets[2])
        else:
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)

        return x

class FourierNet2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, refine_threshold=0.1, refine_factor=2, activation='tanh'):
        super(FourierNet2d, self).__init__()

        self.block1 = FourierBlock2d(in_channels, 16, modes1, modes2, refine_threshold, refine_factor, activation)
        self.block2 = FourierBlock2d(16, 32, modes1, modes2, refine_threshold, refine_factor, activation)
        self.block3 = FourierBlock2d(32, out_channels, modes1, modes2, refine_threshold, refine_factor, activation)

    def forward(self, x, targets=None):
        '''
        Args:
            x (torch.Tensor): Input tensor [batch, in_channels, height, width]
            targets (list of torch.Tensor, optional): List of target tensors corresponding to each block.

        Returns:
            torch.Tensor: Output tensor [batch, out_channels, height, width]
        '''
        if targets is not None and len(targets) != 3:
            raise ValueError("targets must be a list of three tensors corresponding to each FourierBlock2d")

        if targets is not None:
            x = self.block1(x, targets=targets[0])
            x = self.block2(x, targets=targets[1])
            x = self.block3(x, targets=targets[2])
        else:
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)

        return x

class FourierNet3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, refine_threshold=0.1, refine_factor=2, activation='tanh'):
        super(FourierNet3d, self).__init__()

        self.block1 = FourierBlock3d(in_channels, 16, modes1, modes2, modes3, refine_threshold, refine_factor, activation)
        self.block2 = FourierBlock3d(16, 32, modes1, modes2, modes3, refine_threshold, refine_factor, activation)
        self.block3 = FourierBlock3d(32, out_channels, modes1, modes2, modes3, refine_threshold, refine_factor, activation)

    def forward(self, x, targets=None):
        '''
        Args:
            x (torch.Tensor): Input tensor [batch, in_channels, depth, height, width]
            targets (list of torch.Tensor, optional): List of target tensors corresponding to each block.

        Returns:
            torch.Tensor: Output tensor [batch, out_channels, depth, height, width]
        '''
        if targets is not None and len(targets) != 3:
            raise ValueError("targets must be a list of three tensors corresponding to each FourierBlock3d")

        if targets is not None:
            x = self.block1(x, targets=targets[0])
            x = self.block2(x, targets=targets[1])
            x = self.block3(x, targets=targets[2])
        else:
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)

        return x

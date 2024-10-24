import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################
# Low-Pass Filter Function
################################################################

#def low_pass_filter(x_ht, cutoff):
#    """
#    Applies a low-pass filter to the spectral coefficients (DHT output).
#    Frequencies higher than `cutoff` are dampened.
#    """
#    size = x_ht.shape[-1]  # Get the last dimension (frequency axis)
#    frequencies = torch.fft.fftfreq(size, d=1.0)  # Compute frequency bins
#    filter_mask = torch.abs(frequencies) <= cutoff  # Mask for low frequencies
#    return x_ht * filter_mask.to(x_ht.device)

################################################################
# Gaussian Smoothing Function
################################################################

#def gaussian_smoothing(x, kernel_size=5, sigma=1.0):
#    """
#    Applies Gaussian smoothing to the output.
#    """
#    # Apply Gaussian blur (use 2D or 3D kernel as needed)
#    return F.gaussian_blur(x, kernel_size=[kernel_size], sigma=[sigma])

################################################################
# Data Augmentation Function
################################################################

#def augment_data(inputs, shift_range=0.1, scale_range=0.05):
#    """
#    Augment input data by applying random shifts and scaling.
#    
#    Parameters:
#    - inputs: torch.Tensor, the input data to be augmented
#    - shift_range: float, the maximum range for random shifts
#    - scale_range: float, the maximum range for random scaling
#    
#    Returns:
#    - augmented_inputs: torch.Tensor, the augmented input data
#    """
#    # Apply random shifts
#    shifts = torch.rand(inputs.size()) * shift_range
#    augmented_inputs = inputs + shifts
#    
#    # Apply random scaling
#    scales = 1 + torch.rand(inputs.size()) * scale_range
#    augmented_inputs = augmented_inputs * scales
#    
#    return augmented_inputs

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
    """
    1D convolution using torch.einsum.

    Args:
        x1 (torch.Tensor): Input tensor with shape [batch, in_channels, length]
        x2 (torch.Tensor): Kernel tensor with shape [in_channels, out_channels, modes1]

    Returns:
        torch.Tensor: Convolved tensor with shape [batch, out_channels, length]
    """
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    return torch.einsum("bix,iox->box", x1, x2)

def compl_mul2d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    2D convolution using torch.einsum.

    Args:
        x1 (torch.Tensor): Input tensor with shape [batch, in_channels, height, width]
        x2 (torch.Tensor): Kernel tensor with shape [in_channels, out_channels, modes1, modes2]

    Returns:
        torch.Tensor: Convolved tensor with shape [batch, out_channels, height, width]
    """
    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    return torch.einsum("bixy,ioxy->boxy", x1, x2)

def compl_mul3d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    3D convolution using torch.einsum.

    Args:
        x1 (torch.Tensor): Input tensor with shape [batch, in_channels, depth, height, width]
        x2 (torch.Tensor): Kernel tensor with shape [in_channels, out_channels, modes1, modes2, modes3]

    Returns:
        torch.Tensor: Convolved tensor with shape [batch, out_channels, depth, height, width]
    """
    # (batch, in_channel, x,y,z ), (in_channel, out_channel, x,y,z) -> (batch, out_channel, x,y,z)
    return torch.einsum("bixyz,ioxyz->boxyz", x1, x2)

################################################################
# Flip Periodic Functions
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

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1)
        )

        # Additional parameters for phase approximation
        self.phase_weights = nn.Parameter(torch.rand(in_channels, out_channels, self.modes1))

    def forward(self, x):
        batchsize = x.shape[0]
        length = x.shape[-1]

        # Compute Hartley coefficients
        x_ht = dht_1d(x)  # [batch, in_channels, length]

        # Prepare output tensor
        out_ht = torch.zeros(
            batchsize,
            self.out_channels,
            length,
            device=x.device,
            dtype=x.dtype
        )

        # Multiply relevant Hartley modes
        out_ht[:, :, :self.modes1] = dht_conv_1d(
            x_ht[:, :, :self.modes1],
            self.weights1
        )

        # Estimate phase information
        phase_estimation = torch.sin(self.phase_weights)  # Shape: [in_channels, out_channels, modes1]
        phase_estimation = phase_estimation.unsqueeze(0)  # Shape: [1, in_channels, out_channels, modes1]
        phase_estimation = phase_estimation.expand(batchsize, -1, -1, -1)  # Shape: [batchsize, in_channels, out_channels, modes1]

        # Apply phase adjustment
        out_ht[:, :, :self.modes1] *= phase_estimation[:, :, :, :self.modes1]

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

        # Additional parameters for phase approximation
        self.phase_weights = nn.Parameter(torch.rand(in_channels, out_channels, self.modes1, self.modes2))

    def forward(self, x):
        batchsize = x.shape[0]
        height, width = x.shape[-2], x.shape[-1]

        # Compute Hartley coefficients
        x_ht = dht_2d(x)  # [batch, in_channels, height, width]

        # Prepare output tensor
        out_ht = torch.zeros(
            batchsize,
            self.out_channels,
            height,
            width,
            device=x.device,
            dtype=x.dtype
        )

        # Multiply relevant Hartley modes
        out_ht[:, :, :self.modes1, :self.modes2] = dht_conv_2d(
            x_ht[:, :, :self.modes1, :self.modes2],
            self.weights1
        )

        # Estimate phase information
        phase_estimation = torch.sin(self.phase_weights)  # Shape: [in_channels, out_channels, modes1, modes2]
        phase_estimation = phase_estimation.unsqueeze(0)  # Shape: [1, in_channels, out_channels, modes1, modes2]
        phase_estimation = phase_estimation.expand(batchsize, -1, -1, -1, -1)  # Shape: [batchsize, in_channels, out_channels, modes1, modes2]

        # Apply phase adjustment
        out_ht[:, :, :self.modes1, :self.modes2] *= phase_estimation[:, :, :, :self.modes1, :self.modes2]

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
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, self.modes3
            )
        )

        # Additional parameters for phase approximation
        self.phase_weights = nn.Parameter(torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))

    def forward(self, x):
        batchsize = x.shape[0]
        depth, height, width = x.shape[-3], x.shape[-2], x.shape[-1]

        # Compute Hartley coefficients
        x_ht = dht_3d(x)  # [batch, in_channels, depth, height, width]

        # Prepare output tensor
        out_ht = torch.zeros(
            batchsize,
            self.out_channels,
            depth,
            height,
            width,
            device=x.device,
            dtype=x.dtype
        )

        # Multiply relevant Hartley modes
        out_ht[:, :, :self.modes1, :self.modes2, :self.modes3] = dht_conv_3d(
            x_ht[:, :, :self.modes1, :self.modes2, :self.modes3],
            self.weights1
        )

        # Estimate phase information
        phase_estimation = torch.sin(self.phase_weights)  # Shape: [in_channels, out_channels, modes1, modes2, modes3]
        phase_estimation = phase_estimation.unsqueeze(0)  # Shape: [1, in_channels, out_channels, modes1, modes2, modes3]
        phase_estimation = phase_estimation.expand(batchsize, -1, -1, -1, -1, -1)  # Shape: [batchsize, in_channels, out_channels, modes1, modes2, modes3]

        # Apply phase adjustment
        out_ht[:, :, :self.modes1, :self.modes2, :self.modes3] *= phase_estimation[:, :, :, :self.modes1, :self.modes2, :self.modes3]

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

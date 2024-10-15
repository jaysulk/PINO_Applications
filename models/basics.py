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
# Adaptive Spectral Convolution Layers
################################################################

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, max_modes, region_size, overlap=0):
        """
        Adaptive 1D Hartley convolution layer.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            max_modes (int): Maximum number of Hartley modes per region.
            region_size (int): Size of each region.
            overlap (int): Overlap size between regions.
        """
        super(SpectralConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_modes = max_modes
        self.region_size = region_size
        self.overlap = overlap

        self.scale = (1 / (in_channels * out_channels))
        # Initialize weights for maximum modes per region
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.max_modes)
        )

    def forward(self, x, error_signal):
        """
        Forward pass with adaptive mode refinement.
        
        Args:
            x (Tensor): Input tensor of shape (batch, in_channels, length).
            error_signal (Tensor): Error tensor of shape (batch, length) indicating errors per position.
        
        Returns:
            Tensor: Output tensor of shape (batch, out_channels, length).
        """
        batchsize, in_channels, length = x.shape
        device = x.device
        dtype = x.dtype

        # Determine the number of regions
        stride = self.region_size - self.overlap
        num_regions = (length - self.overlap) // stride
        if (length - self.overlap) % stride != 0:
            num_regions += 1  # Account for remaining part

        # Initialize output tensor and overlap counts for normalization
        out = torch.zeros(batchsize, self.out_channels, length, device=device, dtype=dtype)
        overlap_counts = torch.zeros(1, 1, length, device=device, dtype=dtype)

        for i in range(num_regions):
            start = i * stride
            end = start + self.region_size
            if end > length:
                end = length
                start = max(end - self.region_size, 0)

            region_x = x[:, :, start:end]  # (batch, in_channels, region_size)
            region_error = error_signal[:, start:end]  # (batch, region_size)

            # Estimate average error in the region
            avg_error = region_error.mean(dim=1)  # (batch,)

            # Determine number of modes based on error
            # Normalize error to [0, 1]
            min_error = avg_error.min()
            max_error = avg_error.max()
            if max_error - min_error > 1e-8:
                normalized_error = (avg_error - min_error) / (max_error - min_error)
            else:
                normalized_error = torch.zeros_like(avg_error)
            # Linearly scale modes between 1 and max_modes based on normalized error
            num_modes = (normalized_error * (self.max_modes - 1)).int() + 1  # At least 1 mode

            # Apply Hartley Transform
            region_ht = dht(region_x, dim=-1)  # (batch, in_channels, region_size)

            # Initialize output Hartley coefficients for the region
            out_ht = torch.zeros(batchsize, self.out_channels, self.region_size, device=device, dtype=dtype)

            for b in range(batchsize):
                nm = num_modes[b].item()
                if nm > self.max_modes:
                    nm = self.max_modes
                # Multiply relevant Hartley modes
                out_ht[b, :, :nm] = conv(region_ht[b, :, :nm], self.weights[:, :, :nm])

            # Inverse Hartley Transform
            region_out = idht(out_ht, dim=-1)  # (batch, out_channels, region_size)

            # Overlap-Add to the output
            out[:, :, start:end] += region_out
            overlap_counts[:, :, start:end] += 1

        # Avoid division by zero
        overlap_counts[overlap_counts == 0] = 1
        out /= overlap_counts

        return out

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, max_modes1, max_modes2, region_size1, region_size2, overlap1=0, overlap2=0):
        """
        Adaptive 2D Hartley convolution layer.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            max_modes1 (int): Maximum number of Hartley modes along the first dimension per region.
            max_modes2 (int): Maximum number of Hartley modes along the second dimension per region.
            region_size1 (int): Size of each region along the first dimension.
            region_size2 (int): Size of each region along the second dimension.
            overlap1 (int): Overlap size between regions along the first dimension.
            overlap2 (int): Overlap size between regions along the second dimension.
        """
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_modes1 = max_modes1
        self.max_modes2 = max_modes2
        self.region_size1 = region_size1
        self.region_size2 = region_size2
        self.overlap1 = overlap1
        self.overlap2 = overlap2

        self.scale = (1 / (in_channels * out_channels))
        # Initialize weights for maximum modes per region
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.max_modes1, self.max_modes2)
        )

    def forward(self, x, error_signal):
        """
        Forward pass with adaptive mode refinement.
        
        Args:
            x (Tensor): Input tensor of shape (batch, in_channels, height, width).
            error_signal (Tensor): Error tensor of shape (batch, height, width) indicating errors per position.
        
        Returns:
            Tensor: Output tensor of shape (batch, out_channels, height, width).
        """
        batchsize, in_channels, size1, size2 = x.shape
        device = x.device
        dtype = x.dtype

        # Determine the number of regions in each dimension
        stride1 = self.region_size1 - self.overlap1
        stride2 = self.region_size2 - self.overlap2
        num_regions1 = (size1 - self.overlap1) // stride1
        if (size1 - self.overlap1) % stride1 != 0:
            num_regions1 += 1
        num_regions2 = (size2 - self.overlap2) // stride2
        if (size2 - self.overlap2) % stride2 != 0:
            num_regions2 += 1

        # Initialize output tensor and overlap counts for normalization
        out = torch.zeros(batchsize, self.out_channels, size1, size2, device=device, dtype=dtype)
        overlap_counts = torch.zeros(1, 1, size1, size2, device=device, dtype=dtype)

        for i in range(num_regions1):
            for j in range(num_regions2):
                start1 = i * stride1
                end1 = start1 + self.region_size1
                if end1 > size1:
                    end1 = size1
                    start1 = max(end1 - self.region_size1, 0)

                start2 = j * stride2
                end2 = start2 + self.region_size2
                if end2 > size2:
                    end2 = size2
                    start2 = max(end2 - self.region_size2, 0)

                region_x = x[:, :, start1:end1, start2:end2]  # (batch, in_channels, region_size1, region_size2)
                region_error = error_signal[:, start1:end1, start2:end2]  # (batch, region_size1, region_size2)

                # Estimate average error in the region
                avg_error = region_error.mean(dim=(1, 2))  # (batch,)

                # Determine number of modes based on error
                # Normalize error to [0, 1]
                min_error = avg_error.min()
                max_error = avg_error.max()
                if max_error - min_error > 1e-8:
                    normalized_error = (avg_error - min_error) / (max_error - min_error)
                else:
                    normalized_error = torch.zeros_like(avg_error)
                # Linearly scale modes between 1 and max_modes based on normalized error
                num_modes1 = (normalized_error * (self.max_modes1 - 1)).int() + 1
                num_modes2 = (normalized_error * (self.max_modes2 - 1)).int() + 1

                # Apply 2D Hartley Transform
                region_ht = dht(dht(region_x, dim=-2), dim=-1)  # 2D Hartley Transform

                # Initialize output Hartley coefficients for the region
                out_ht = torch.zeros(batchsize, self.out_channels, self.region_size1, self.region_size2, device=device, dtype=dtype)

                for b in range(batchsize):
                    nm1 = num_modes1[b].item()
                    nm2 = num_modes2[b].item()
                    nm1 = min(nm1, self.max_modes1)
                    nm2 = min(nm2, self.max_modes2)
                    # Multiply relevant Hartley modes
                    out_ht[b, :, :nm1, :nm2] = conv(region_ht[b, :, :nm1, :nm2], self.weights[:, :, :nm1, :nm2])

                # Inverse 2D Hartley Transform
                region_out = idht(idht(out_ht, dim=-2), dim=-1)  # 2D Inverse Hartley Transform

                # Overlap-Add to the output
                out[:, :, start1:end1, start2:end2] += region_out
                overlap_counts[:, :, start1:end1, start2:end2] += 1

        # Avoid division by zero
        overlap_counts[overlap_counts == 0] = 1
        out /= overlap_counts

        return out

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, max_modes1, max_modes2, max_modes3,
                 region_size1, region_size2, region_size3, overlap1=0, overlap2=0, overlap3=0):
        """
        Adaptive 3D Hartley convolution layer.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            max_modes1 (int): Maximum number of Hartley modes along the first dimension per region.
            max_modes2 (int): Maximum number of Hartley modes along the second dimension per region.
            max_modes3 (int): Maximum number of Hartley modes along the third dimension per region.
            region_size1 (int): Size of each region along the first dimension.
            region_size2 (int): Size of each region along the second dimension.
            region_size3 (int): Size of each region along the third dimension.
            overlap1 (int): Overlap size between regions along the first dimension.
            overlap2 (int): Overlap size between regions along the second dimension.
            overlap3 (int): Overlap size between regions along the third dimension.
        """
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_modes1 = max_modes1
        self.max_modes2 = max_modes2
        self.max_modes3 = max_modes3
        self.region_size1 = region_size1
        self.region_size2 = region_size2
        self.region_size3 = region_size3
        self.overlap1 = overlap1
        self.overlap2 = overlap2
        self.overlap3 = overlap3

        self.scale = (1 / (in_channels * out_channels))
        # Initialize weights for maximum modes per region
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.max_modes1, self.max_modes2, self.max_modes3)
        )

    def forward(self, x, error_signal):
        """
        Forward pass with adaptive mode refinement.
        
        Args:
            x (Tensor): Input tensor of shape (batch, in_channels, depth, height, width).
            error_signal (Tensor): Error tensor of shape (batch, depth, height, width) indicating errors per position.
        
        Returns:
            Tensor: Output tensor of shape (batch, out_channels, depth, height, width).
        """
        batchsize, in_channels, size1, size2, size3 = x.shape
        device = x.device
        dtype = x.dtype

        # Determine the number of regions in each dimension
        stride1 = self.region_size1 - self.overlap1
        stride2 = self.region_size2 - self.overlap2
        stride3 = self.region_size3 - self.overlap3
        num_regions1 = (size1 - self.overlap1) // stride1
        if (size1 - self.overlap1) % stride1 != 0:
            num_regions1 += 1
        num_regions2 = (size2 - self.overlap2) // stride2
        if (size2 - self.overlap2) % stride2 != 0:
            num_regions2 += 1
        num_regions3 = (size3 - self.overlap3) // stride3
        if (size3 - self.overlap3) % stride3 != 0:
            num_regions3 += 1

        # Initialize output tensor and overlap counts for normalization
        out = torch.zeros(batchsize, self.out_channels, size1, size2, size3, device=device, dtype=dtype)
        overlap_counts = torch.zeros(1, 1, size1, size2, size3, device=device, dtype=dtype)

        for i in range(num_regions1):
            for j in range(num_regions2):
                for k in range(num_regions3):
                    start1 = i * stride1
                    end1 = start1 + self.region_size1
                    if end1 > size1:
                        end1 = size1
                        start1 = max(end1 - self.region_size1, 0)

                    start2 = j * stride2
                    end2 = start2 + self.region_size2
                    if end2 > size2:
                        end2 = size2
                        start2 = max(end2 - self.region_size2, 0)

                    start3 = k * stride3
                    end3 = start3 + self.region_size3
                    if end3 > size3:
                        end3 = size3
                        start3 = max(end3 - self.region_size3, 0)

                    region_x = x[:, :, start1:end1, start2:end2, start3:end3]  # (batch, in_channels, region_size1, region_size2, region_size3)
                    region_error = error_signal[:, start1:end1, start2:end2, start3:end3]  # (batch, region_size1, region_size2, region_size3)

                    # Estimate average error in the region
                    avg_error = region_error.mean(dim=(1, 2, 3))  # (batch,)

                    # Determine number of modes based on error
                    # Normalize error to [0, 1]
                    min_error = avg_error.min()
                    max_error = avg_error.max()
                    if max_error - min_error > 1e-8:
                        normalized_error = (avg_error - min_error) / (max_error - min_error)
                    else:
                        normalized_error = torch.zeros_like(avg_error)
                    # Linearly scale modes between 1 and max_modes based on normalized error
                    num_modes1 = (normalized_error * (self.max_modes1 - 1)).int() + 1
                    num_modes2 = (normalized_error * (self.max_modes2 - 1)).int() + 1
                    num_modes3 = (normalized_error * (self.max_modes3 - 1)).int() + 1

                    # Apply 3D Hartley Transform
                    region_ht = dht(dht(dht(region_x, dim=-3), dim=-2), dim=-1)  # 3D Hartley Transform

                    # Initialize output Hartley coefficients for the region
                    out_ht = torch.zeros(batchsize, self.out_channels, self.region_size1, self.region_size2, self.region_size3, device=device, dtype=dtype)

                    for b in range(batchsize):
                        nm1 = num_modes1[b].item()
                        nm2 = num_modes2[b].item()
                        nm3 = num_modes3[b].item()
                        nm1 = min(nm1, self.max_modes1)
                        nm2 = min(nm2, self.max_modes2)
                        nm3 = min(nm3, self.max_modes3)
                        # Multiply relevant Hartley modes
                        out_ht[b, :, :nm1, :nm2, :nm3] = conv(region_ht[b, :, :nm1, :nm2, :nm3], self.weights[:, :, :nm1, :nm2, :nm3])

                    # Inverse 3D Hartley Transform
                    region_out = idht(idht(idht(out_ht, dim=-3), dim=-2), dim=-1)  # 3D Inverse Hartley Transform

                    # Overlap-Add to the output
                    out[:, :, start1:end1, start2:end2, start3:end3] += region_out
                    overlap_counts[:, :, start1:end1, start2:end2, start3:end3] += 1

        # Avoid division by zero
        overlap_counts[overlap_counts == 0] = 1
        out /= overlap_counts

        return out


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

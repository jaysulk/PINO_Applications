import torch
import torch.nn as nn
import torch.nn.functional as F

def dht(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:  # 1D DHT for 3D tensors
        result = torch.fft.fftn(x, dim=[2])
    elif x.ndim == 4:  # 2D DHT for 4D tensors
        result = torch.fft.fftn(x, dim=[2, 3])
    elif x.ndim == 5:  # 3D DHT for 5D tensors
        result = torch.fft.fftn(x, dim=[2, 3, 4])
    else:
        raise ValueError("Unsupported input: Only 3D (1D DHT), 4D (2D DHT), and 5D (3D DHT) tensors are supported.")
    
    # Combine real and imaginary parts in a way that mimics DHT behavior
    return result.real - result.imag

def idht(x: torch.Tensor) -> torch.Tensor:
    # Compute the DHT
    transformed = dht(x)
    
    # Determine normalization factor based on input dimensions
    if x.ndim == 3:
        N = x.size(2)
        normalization_factor = N
    elif x.ndim == 4:
        M, N = x.size(2), x.size(3)
        normalization_factor = M * N
    elif x.ndim == 5:
        D, M, N = x.size(2), x.size(3), x.size(4)
        normalization_factor = D * M * N
    else:
        raise ValueError(f"Input tensor must be 3D, 4D, or 5D, but got {x.ndim}D with shape {x.shape}.")
    
    # Return the normalized inverse
    return transformed / normalization_factor


def compl_mul1d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#    X1_H_k = x1
#    X2_H_k = x2
#    X1_H_neg_k = torch.roll(torch.flip(x1, dims=[-1]), shifts=1, dims=[-1])
#    X2_H_neg_k = torch.roll(torch.flip(x2, dims=[-1]), shifts=1, dims=[-1])

#    result = 0.5 * (torch.einsum('bix,iox->box', X1_H_k, X2_H_k) -
#                    torch.einsum('bix,iox->box', X1_H_neg_k, X2_H_neg_k) +
#                    torch.einsum('bix,iox->box', X1_H_k, X2_H_neg_k) +
#                    torch.einsum('bix,iox->box', X1_H_neg_k, X2_H_k))

#    return result
    return torch.einsum("bi...,io...->bo...", x1, x2)

def compl_mul2d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#    X1_H_k = x1
#    X2_H_k = x2
#    X1_H_neg_k = torch.roll(torch.flip(x1, dims=[-1, -2]), shifts=(1, 1), dims=[-1, -2])
#    X2_H_neg_k = torch.roll(torch.flip(x2, dims=[-1, -2]), shifts=(1, 1), dims=[-1, -2])

#    result = 0.5 * (torch.einsum('bixy,ioxy->boxy', X1_H_k, X2_H_k) -
#                    torch.einsum('bixy,ioxy->boxy', X1_H_neg_k, X2_H_neg_k) +
#                    torch.einsum('bixy,ioxy->boxy', X1_H_k, X2_H_neg_k) +
#                    torch.einsum('bixy,ioxy->boxy', X1_H_neg_k, X2_H_k))

#    return result
    return torch.einsum("bixy...,ioxy...->boxy...", x1, x2)

def compl_mul3d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#    X1_H_k = x1
#    X2_H_k = x2
#    X1_H_neg_k = torch.roll(torch.flip(x1, dims=[-3, -2, -1]), shifts=(1, 1, 1), dims=[-3, -2, -1])
#    X2_H_neg_k = torch.roll(torch.flip(x2, dims=[-3, -2, -1]), shifts=(1, 1, 1), dims=[-3, -2, -1])

#    result = 0.5 * (torch.einsum('bixyz,ioxyz->boxyz', X1_H_k, X2_H_k) -
#                    torch.einsum('bixyz,ioxyz->boxyz', X1_H_neg_k, X2_H_neg_k) +
#                    torch.einsum('bixyz,ioxyz->boxyz', X1_H_k, X2_H_neg_k) +
#                    torch.einsum('bixyz,ioxyz->boxyz', X1_H_neg_k, X2_H_k))

#    return result
    return torch.einsum("bixyz...,ioxyz...->boxyz...", a, b)

def compl_mul1d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#    X1_H_k = x1
#    X2_H_k = x2
#    X1_H_neg_k = torch.roll(torch.flip(x1, dims=[-1]), shifts=1, dims=[-1])
#    X2_H_neg_k = torch.roll(torch.flip(x2, dims=[-1]), shifts=1, dims=[-1])

#    result = 0.5 * (torch.einsum('bix,iox->box', X1_H_k, X2_H_k) -
#                    torch.einsum('bix,iox->box', X1_H_neg_k, X2_H_neg_k) +
#                    torch.einsum('bix,iox->box', X1_H_k, X2_H_neg_k) +
#                    torch.einsum('bix,iox->box', X1_H_neg_k, X2_H_k))

#    return result
    return torch.einsum("bi...,io...->bo...", x1, x2)

def compl_mul2d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#    X1_H_k = x1
#    X2_H_k = x2
#    X1_H_neg_k = torch.roll(torch.flip(x1, dims=[-1, -2]), shifts=(1, 1), dims=[-1, -2])
#    X2_H_neg_k = torch.roll(torch.flip(x2, dims=[-1, -2]), shifts=(1, 1), dims=[-1, -2])

#    result = 0.5 * (torch.einsum('bixy,ioxy->boxy', X1_H_k, X2_H_k) -
#                    torch.einsum('bixy,ioxy->boxy', X1_H_neg_k, X2_H_neg_k) +
#                    torch.einsum('bixy,ioxy->boxy', X1_H_k, X2_H_neg_k) +
#                    torch.einsum('bixy,ioxy->boxy', X1_H_neg_k, X2_H_k))

#    return result
    return torch.einsum("bixy...,ioxy...->boxy...", x1, x2)

def compl_mul3d(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#    X1_H_k = x1
#    X2_H_k = x2
#    X1_H_neg_k = torch.roll(torch.flip(x1, dims=[-3, -2, -1]), shifts=(1, 1, 1), dims=[-3, -2, -1])
#    X2_H_neg_k = torch.roll(torch.flip(x2, dims=[-3, -2, -1]), shifts=(1, 1, 1), dims=[-3, -2, -1])

#    result = 0.5 * (torch.einsum('bixyz,ioxyz->boxyz', X1_H_k, X2_H_k) -
#                    torch.einsum('bixyz,ioxyz->boxyz', X1_H_neg_k, X2_H_neg_k) +
#                    torch.einsum('bixyz,ioxyz->boxyz', X1_H_k, X2_H_neg_k) +
#                    torch.einsum('bixyz,ioxyz->boxyz', X1_H_neg_k, X2_H_k))

#    return result
    return torch.einsum("bixyz...,ioxyz...->boxyz...", x1, x2)

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
        # Number of Hartley modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1))

    def forward(self, x):
        batchsize = x.shape[0]
        
        # Augment the input data
        #x = augment_data(x)
        
        # Compute Hartley coefficients up to factor of h^(- something constant)
        x_ht = dht(x)
        # Compute DHT of the flipped input to simulate sine part
        x_ht_flip = dht(x.flip(dims=[-1]))

        # Combine the Hartley and flipped-Hartley (cosine and sine components)
        cos_part = x_ht
        sin_part = x_ht_flip
        # Construct a complex representation from the cosine and sine components
        z = torch.complex(cos_part, sin_part)

        # Calculate phase information
        phase = torch.angle(z)  # Extract phase as arctan(imag/real)

        # Multiply relevant Hartley modes (magnitude component)
        out_ht = torch.zeros(batchsize, self.in_channels, x.size(-1)//2 + 1, device=x.device)
        out_ht[:, :, :self.modes1] = compl_mul1d(cos_part[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = idht(out_ht)

        # Combine magnitude (x) and phase into a real-valued signal
        #reconstructed_signal = x * torch.cos(phase)  # Reconstruction with magnitude and phase
        #return reconstructed_signal
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
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))

    def forward(self, x):
        batchsize = x.shape[0]
        size1 = x.shape[-2]
        size2 = x.shape[-1]
        
        # Augment the input data
        #x = augment_data(x)
        
        # Compute DHT
        x_dht = dht(x)
        # Compute DHT of the flipped input to simulate sine part
        x_dht_flip = dht(x.flip(dims=[-2, -1]))

        # Combine the Hartley and flipped-Hartley (cosine and sine components)
        cos_part = x_dht
        sin_part = x_dht_flip
        # Construct a complex representation from the cosine and sine components
        z = torch.complex(cos_part, sin_part)

        # Calculate phase information
        phase = torch.angle(z)  # Extract phase as arctan(imag/real)

        # Multiply relevant Hartley modes (magnitude component)
        out_dht = torch.zeros(batchsize, self.out_channels, size1, size2, device=x.device)
        out_dht[:, :, :self.modes1, :self.modes2] = compl_mul2d(cos_part[:, :, :self.modes1, :self.modes2], self.weights1)
        out_dht[:, :, -self.modes1:, :self.modes2] = compl_mul2d(cos_part[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = idht(out_dht)

        # Combine magnitude (x) and phase into a real-valued signal
        #reconstructed_signal = x * torch.cos(phase)  # Reconstruction with magnitude and phase
        #return reconstructed_signal
        return x


################################################################
# 3D Hartley convolution layer
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Hartley modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))

    def forward(self, x):
        batchsize = x.shape[0]

        # Augment the input data
        #x = augment_data(x)

        # Compute Hartley coefficients up to factor of h^(- something constant)
        x_ht = dht(x)
        x_ht_flip = dht(x.flip(dims=[2, 3, 4]))

        # Combine the Hartley and flipped-Hartley (cosine and sine components)
        cos_part = x_ht
        sin_part = x_ht_flip
        # Construct a complex representation from the cosine and sine components
        z = torch.complex(cos_part, sin_part)

        # Calculate phase information
        phase = torch.angle(z)  # Extract phase as arctan(imag/real)

        # Multiply relevant Hartley modes (magnitude component)
        out_ht = torch.zeros(batchsize, self.out_channels, x.size(2), x.size(3), x.size(4)//2 + 1, device=x.device)
        out_ht[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            compl_mul3d(cos_part[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ht[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            compl_mul3d(cos_part[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ht[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            compl_mul3d(cos_part[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ht[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            compl_mul3d(cos_part[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = idht(out_ht)

        # Combine magnitude (x) and phase into a real-valued signal
        #reconstructed_signal = x * torch.cos(phase)  # Reconstruction with magnitude and phase
        #return reconstructed_signal
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
        self.linear = nn.Conv1d(in_channels, out_channels, 1)
        
        # Activation function selection
        if activation == 'tanh':
            self.activation = nn.Tanh()  # Use nn.Tanh() for module (not in-place operation)
        elif activation == 'gelu':
            self.activation = nn.GELU()  # Apply GELU non-linearity
        elif activation == 'none':
            self.activation = None  # No activation
        else:
            raise ValueError(f'{activation} is not supported')

    def forward(self, x):
        '''
        Input x: (batchsize, in_channels, x_grid, y_grid, t_grid)
        '''
        # Apply spectral convolution (3D Hartley convolution)
        x1 = self.speconv(x)
        
        # Apply 1D convolution across the channel dimension
        # Flattening the last three dimensions into one while keeping the batch and channel
        x2 = self.linear(x.view(x.shape[0], self.in_channel, -1))
        
        # Reshape x2 back to match the original spatial and temporal grid structure
        x2 = x2.view(x.shape[0], self.out_channel, x.shape[2], x.shape[3], x.shape[4])
        
        # Combine spectral and linear outputs (skip connection)
        out = x1 + x2
        
        # Apply activation function (non-linearity)
        if self.activation is not None:
            out = self.activation(out)
        
        return out

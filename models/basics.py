import torch
import torch.nn as nn
import torch.nn.functional as F

import torch

def dht(x: torch.Tensor, dim=None) -> torch.Tensor:
    # Compute the N-dimensional FFT of the input tensor
    result = torch.fft.fftn(x, dim=dim)
    
    # Combine real and imaginary parts to compute the DHT
    return result.real + result.imag  # Use subtraction to match DHT definition

def idht(x: torch.Tensor, dim=None) -> torch.Tensor:
    # Compute the DHT of the input tensor
    transformed = dht(x, dim=dim)
    
    # Determine normalization factor based on the specified dimensions
    if dim is None:
        # If dim is None, use the total number of elements
        normalization_factor = x.numel()
    else:
        # Ensure dim is a list of dimensions
        if isinstance(dim, int):
            dim = [dim]
        normalization_factor = 1
        for d in dim:
            normalization_factor *= x.size(d)
    
    # Return the normalized inverse DHT
    return transformed / normalization_factor

def hilbert_transform(x: torch.Tensor, dim=None) -> torch.Tensor:
    """
    Compute the Hilbert transform of x along specified dimensions.
    
    Parameters:
    x (Tensor): Input tensor of shape (..., signal_dims)
    dim (tuple or int): Dimensions along which to compute the Hilbert transform
    
    Returns:
    Tensor: The Hilbert transform of x
    """
    # Compute the FFT along the specified dimensions
    X = torch.fft.fftn(x, dim=dim)
    
    # Generate the frequency multiplier h
    shape = x.shape
    h = torch.ones_like(x, dtype=torch.float, device=x.device)
    
    for d in dim:
        N = shape[d]
        freq = torch.arange(N, device=x.device)
        h_dim = torch.zeros(N, device=x.device)
        
        # Create the frequency multiplier for the Hilbert transform along this dimension
        if N % 2 == 0:
            # Even length
            h_dim[0] = 1
            h_dim[1:N//2] = 2
            h_dim[N//2] = 1
        else:
            # Odd length
            h_dim[0] = 1
            h_dim[1:(N+1)//2] = 2
        
        # Reshape h_dim to broadcast correctly
        shape_h = [1] * x.dim()
        shape_h[d] = N
        h_dim = h_dim.view(shape_h)
        
        # Multiply h with h_dim
        h = h * h_dim

    # Apply the frequency multiplier
    X = X * h

    # Compute the inverse FFT to get the Hilbert transform
    x_hilbert = torch.fft.ifftn(X, dim=dim).real
    return x_hilbert

def compl_mul1d(a, b):
    # a: (batch, in_channel, x)
    # b: (in_channel, out_channel, x)
    Ha = dht(a)
    Hb = dht(b)
    H_tilde_a = dht(hilbert_transform(a))
    H_tilde_b = dht(hilbert_transform(b))
    # Compute the convolution according to the Hartley convolution theorem
    term1 = torch.einsum("bix,iox->box", Ha, Hb)
    term2 = torch.einsum("bix,iox->box", H_tilde_a, H_tilde_b)
    return term1 + term2

def compl_mul2d((a, b):
    # a: (batch, in_channel, x, y)
    # b: (in_channel, out_channel, x, y)
    Ha = dht(a)
    Hb = dht(b)
    H_tilde_a = dht(hilbert_transform(a))
    H_tilde_b = dht(hilbert_transform(b))
    term1 = torch.einsum("bixy,ioxy->boxy", Ha, Hb)
    term2 = torch.einsum("bixy,ioxy->boxy", H_tilde_a, H_tilde_b)
    return term1 + term2

def compl_mul3d((a, b):
    # a: (batch, in_channel, x, y, z)
    # b: (in_channel, out_channel, x, y, z)
    Ha = dht(a)
    Hb = dht(b)
    H_tilde_a = dht(hilbert_transform(a))
    H_tilde_b = dht(hilbert_transform(b))
    term1 = torch.einsum("bixyz,ioxyz->boxyz", Ha, Hb)
    term2 = torch.einsum("bixyz,ioxyz->boxyz", H_tilde_a, H_tilde_b)
    return term1 + term2

################################################################
# 1d fourier layer
################################################################


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft =dht(x, dim=[2])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, s=[x.size(-1)], dim=[2])
        return x

################################################################
# 2d fourier layer
################################################################


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x, gridy=None):
        batchsize = x.shape[0]
        size1 = x.shape[-2]
        size2 = x.shape[-1]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2, 3])

        if gridy is None:
            # Multiply relevant Fourier modes
            out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, device=x.device,
                                 dtype=torch.cfloat)
            out_ft[:, :, :self.modes1, :self.modes2] = \
                compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
            out_ft[:, :, -self.modes1:, :self.modes2] = \
                compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

            # Return to physical space
            x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)), dim=[2, 3])
        else:
            factor1 = compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
            factor2 = compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
            x = self.ifft2d(gridy, factor1, factor2, self.modes1, self.modes2) / (size1 * size2)
        return x

    def ifft2d(self, gridy, coeff1, coeff2, k1, k2):

        # y (batch, N, 2) locations in [0,1]*[0,1]
        # coeff (batch, channels, kmax, kmax)

        batchsize = gridy.shape[0]
        N = gridy.shape[1]
        device = gridy.device
        m1 = 2 * k1
        m2 = 2 * k2 - 1

        # wavenumber (m1, m2)
        k_x1 =  torch.cat((torch.arange(start=0, end=k1, step=1), \
                            torch.arange(start=-(k1), end=0, step=1)), 0).reshape(m1,1).repeat(1,m2).to(device)
        k_x2 =  torch.cat((torch.arange(start=0, end=k2, step=1), \
                            torch.arange(start=-(k2-1), end=0, step=1)), 0).reshape(1,m2).repeat(m1,1).to(device)

        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(gridy[:,:,0].view(-1), k_x1.view(-1)).reshape(batchsize, N, m1, m2)
        K2 = torch.outer(gridy[:,:,1].view(-1), k_x2.view(-1)).reshape(batchsize, N, m1, m2)
        K = K1 + K2

        # basis (N, m1, m2)
        basis = torch.exp( 1j * 2* np.pi * K).to(device)

        # coeff (batch, channels, m1, m2)
        coeff3 = coeff1[:,:,1:,1:].flip(-1, -2).conj()
        coeff4 = torch.cat([coeff1[:,:,0:1,1:].flip(-1).conj(), coeff2[:,:,:,1:].flip(-1, -2).conj()], dim=-2)
        coeff12 = torch.cat([coeff1, coeff2], dim=-2)
        coeff43 = torch.cat([coeff4, coeff3], dim=-2)
        coeff = torch.cat([coeff12, coeff43], dim=-1)

        # Y (batch, channels, N)
        Y = torch.einsum("bcxy,bnxy->bcn", coeff, basis)
        Y = Y.real
        return Y


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2,3,4])
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(2), x.size(3), x.size(4)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(2), x.size(3), x.size(4)), dim=[2,3,4])
        return x


class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, activation='tanh'):
        super(FourierBlock, self).__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.speconv = SpectralConv3d(in_channels, out_channels, modes1, modes2, modes3)  # Assuming you have this defined elsewhere
        self.linear = nn.Conv1d(in_channels, out_channels, 1)

        if activation == 'tanh':
            self.activation = torch.tanh_
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = self.swish
        elif activation == 'none':
            self.activation = None

    @staticmethod
    def swish(x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        '''
        input x: (batchsize, channel width, x_grid, y_grid, t_grid)
        '''
        x1 = self.speconv(x)
        x2 = self.linear(x.view(x.shape[0], self.in_channel, -1))
        out = x1 + x2.view(x.shape[0], self.out_channel, x.shape[2], x.shape[3], x.shape[4])
        if self.activation is not None:
            out = self.activation(out)
        return out

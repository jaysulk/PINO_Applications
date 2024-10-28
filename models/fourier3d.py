import torch
import torch.nn as nn
import torch.nn.functional as F

from .basics import SpectralConv3d


class FNN3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width=16, fc_dim=128, layers=None, in_dim=4, out_dim=1, activation_function='sine', pad_x=0, pad_y=0, pad_z=0):
        '''
        Args:
            modes1: list of int, first dimension maximal modes for each layer
            modes2: list of int, second dimension maximal modes for each layer
            modes3: list of int, third dimension maximal modes for each layer
            layers: list of int, channels for each layer
            in_dim: int, input dimension
            out_dim: int, output dimension
            activation_function: str, activation function to use
        '''
        super(FNN3d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.padding = (0, 0, 0, pad_z, 0, pad_y, 0, pad_x)

        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(self.in_dim, self.layers[0])

        self.sp_convs = nn.ModuleList([
            SpectralConv3d(in_size, out_size, mode1_num, mode2_num, mode3_num)
            for in_size, out_size, mode1_num, mode2_num, mode3_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2, self.modes3)
        ])

        self.ws = nn.ModuleList([
            nn.Conv1d(in_size, out_size, 1)
            for in_size, out_size in zip(self.layers, self.layers[1:])
        ])

        self.fc1 = nn.Linear(self.layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, self.out_dim)
        
        # Define the activation function
        self.activation = self.get_activation_function(activation_function)

    def get_activation_function(self, activation_function):
        if activation_function == 'sine':
            return torch.sin
        elif activation_function == 'relu':
            return F.relu
        elif activation_function == 'softplus':
            return F.softplus
        elif activation_function == 'elu':
            return F.elu
        elif activation_function == 'silu':
            return F.silu  # Swish activation
        elif activation_function == 'tanh':
            return torch.tanh
        elif activation_function == 'gelu':
            return F.gelu
        else:
            raise ValueError(f"Unsupported activation function: {activation_function}")

    def forward(self, x):
        '''
        Args:
            x: (batchsize, x_grid, y_grid, z_grid, in_dim)
        Returns:
            u: (batchsize, x_grid, y_grid, z_grid, out_dim)
        '''
        length = len(self.ws)
        batchsize = x.shape[0]
        nx, ny, nz = x.shape[1], x.shape[2], x.shape[3]
        x = F.pad(x, self.padding, "constant", 0)
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)  # Change to shape [batchsize, channels, x, y, z]

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(
                batchsize, self.layers[i + 1], size_x, size_y, size_z)
            x = x1 + x2
            if i != length - 1:
                x = self.activation(x)
        x = x.permute(0, 2, 3, 4, 1)  # Change back to shape [batchsize, x, y, z, channels]
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = x.reshape(batchsize, size_x, size_y, size_z, self.out_dim)
        x = x[:, :nx, :ny, :nz, :]  # Remove padding
        return x

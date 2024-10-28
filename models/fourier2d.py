import torch
import torch.nn as nn
import torch.nn.functional as F

from .lowrank2d import LowRank2d
from .basics import SpectralConv2d


class FNN2d(nn.Module):
    def __init__(self, modes1, modes2,
                 width=64, fc_dim=128,
                 layers=None,
                 in_dim=3, out_dim=1,
                 activation_function='sine',
                 pad_x=0, pad_y=0):
        super(FNN2d, self).__init__()

        """
        The overall network. It contains multiple layers of the Fourier layer.
        1. Lift the input to the desired channel dimension by self.fc0.
        2. Layers of the integral operators u' = (W + K)(u),
           where W is defined by self.ws and K is defined by self.sp_convs.
        3. Project from the channel space to the output space by self.fc1 and self.fc2.

        Input: the solution of the coefficient function and locations (a(x, y), x, y)
        Input shape: (batchsize, x=s, y=s, c=3)
        Output: the solution
        Output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.padding = (0, 0, 0, pad_y, 0, pad_x)
        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, self.layers[0])

        self.sp_convs = nn.ModuleList([
            SpectralConv2d(in_size, out_size, mode1_num, mode2_num)
            for in_size, out_size, mode1_num, mode2_num in zip(
                self.layers, self.layers[1:], self.modes1, self.modes2)
        ])

        self.ws = nn.ModuleList([
            nn.Conv1d(in_size, out_size, 1)
            for in_size, out_size in zip(self.layers, self.layers[1:])
        ])

        self.fc1 = nn.Linear(self.layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)

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
        elif activation_function == 'swish':
            return lambda x: x * torch.sigmoid(x)
        else:
            raise ValueError(f"Unsupported activation function: {activation_function}")

    def forward(self, x):
        '''
        Args:
            - x: (batch size, x_grid, y_grid, in_dim)
        Returns:
            - x: (batch size, x_grid, y_grid, out_dim)
        '''
        length = len(self.ws)
        batchsize = x.shape[0]
        nx, ny = x.shape[1], x.shape[2]  # Original shape
        x = F.pad(x, self.padding, "constant", 0)
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # Change to shape [batchsize, channels, x, y]

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(
                batchsize, self.layers[i + 1], size_x, size_y)
            x = x1 + x2
            if i != length - 1:
                x = self.activation(x)
        x = x.permute(0, 2, 3, 1)  # Change back to shape [batchsize, x, y, channels]
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = x.reshape(batchsize, size_x, size_y, self.out_dim)
        x = x[:, :nx, :ny, :]  # Remove padding
        return x


class PINO2d(nn.Module):
    def __init__(self, modes1, modes2, width, layers=None, in_dim=3, out_dim=1, activation_function='sine'):
        '''
        Args:
            modes1: list of modes to keep in the x-direction
            modes2: list of modes to keep in the y-direction
            width: width of features
            layers: list of integers
            in_dim: input dimensionality, default: a(x), x, t
            out_dim: output dimensionality, default: u(x,t)
            activation_function: activation function to use
        '''
        super(PINO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, self.layers[0])

        self.sp_convs = nn.ModuleList([
            SpectralConv2d(in_size, out_size, mode1_num, mode2_num)
            for in_size, out_size, mode1_num, mode2_num in zip(
                self.layers, self.layers[1:], self.modes1, self.modes2)
        ])

        self.ws = nn.ModuleList([
            nn.Conv1d(in_size, out_size, 1)
            for in_size, out_size in zip(self.layers[:-1], self.layers[1:-1])
        ])
        self.ws.append(LowRank2d(self.layers[-2], self.layers[-1]))
        self.fc1 = nn.Linear(self.layers[-1], self.layers[-1] * 4)
        self.fc2 = nn.Linear(self.layers[-1] * 4, out_dim)

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
        elif activation_function == 'swish':
            return lambda x: x * torch.sigmoid(x)
        else:
            raise ValueError(f"Unsupported activation function: {activation_function}")

    def forward(self, x, y=None):
        '''
        Args:
            - x: (batch size, x_grid, y_grid, in_dim)
            - y: Additional input if needed (optional)
        Returns:
            - x: (batch size, x_grid, y_grid, out_dim)
        '''
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        length = len(self.ws)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            if i != length - 1:
                x1 = speconv(x)
                x2 = w(x.view(batchsize, self.layers[i], -1)).view(
                    batchsize, self.layers[i + 1], size_x, size_y)
                x = x1 + x2
                x = self.activation(x)
            else:
                x1 = speconv(x, y).reshape(batchsize, self.layers[-1], -1)
                x2 = w(x, y).reshape(batchsize, self.layers[-1], -1)
                x = x1 + x2
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class FNN2d_AD(nn.Module):
    def __init__(self, modes1, modes2,
                 width=64, fc_dim=128,
                 layers=None,
                 in_dim=3, out_dim=1,
                 activation_function='sine'):
        super(FNN2d_AD, self).__init__()

        """
        The overall network. It contains multiple layers of the Fourier layer.
        1. Lift the input to the desired channel dimension by self.fc0.
        2. Layers of the integral operators u' = (W + K)(u),
           where W is defined by self.ws and K is defined by self.sp_convs.
        3. Project from the channel space to the output space by self.fc1 and self.fc2.

        Input: the solution of the coefficient function and locations (a(x, y), x, y)
        Input shape: (batchsize, x=s, y=s, c=3)
        Output: the solution
        Output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        # Input channel is 3: (a(x, y), x, y)
        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, self.layers[0])

        self.sp_convs = nn.ModuleList([
            SpectralConv2d(in_size, out_size, mode1_num, mode2_num)
            for in_size, out_size, mode1_num, mode2_num in zip(
                self.layers, self.layers[1:], self.modes1, self.modes2)
        ])

        self.ws = nn.ModuleList([
            nn.Conv1d(in_size, out_size, 1)
            for in_size, out_size in zip(self.layers[:-1], self.layers[1:-1])
        ])
        self.ws.append(LowRank2d(self.layers[-2], self.layers[-1]))

        self.fc1 = nn.Linear(self.layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)

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
        elif activation_function == 'swish':
            return lambda x: x * torch.sigmoid(x)
        else:
            raise ValueError(f"Unsupported activation function: {activation_function}")

    def forward(self, x, y=None):
        '''
        Args:
            - x: (batch size, x_grid, y_grid, in_dim)
            - y: Additional input if needed (optional)
        Returns:
            - x: (batch size, x_grid, y_grid, out_dim)
        '''
        length = len(self.ws)
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            if i != length - 1:
                x1 = speconv(x)
                x2 = w(x.view(batchsize, self.layers[i], -1)).view(
                    batchsize, self.layers[i + 1], size_x, size_y)
                x = x1 + x2
                x = self.activation(x)
            else:
                x1 = speconv(x, y).reshape(batchsize, self.layers[-1], -1)
                x2 = w(x, y).reshape(batchsize, self.layers[-1], -1)
                x = x1 + x2
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

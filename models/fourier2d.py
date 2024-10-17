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
                 activation='tanh',
                 pad_x=0, pad_y=0):
        super(FNN2d, self).__init__()

        """
        The overall network. It contains multiple layers of the SpectralConv2d.
        1. Lift the input to the desired channel dimension by self.fc0.
        2. Multiple layers of the integral operators u' = (W + K)(u).
           W defined by self.ws; K defined by self.sp_convs.
        3. Project from the channel space to the output space by self.fc1 and self.fc2.

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.padding = (0, 0, 0, pad_y, 0, pad_x)
        # input channel is 3: (a(x, y), x, y)
        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, self.layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv2d(
            in_size, out_size, mode1_num, mode2_num)
            for in_size, out_size, mode1_num, mode2_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])

        self.fc1 = nn.Linear(self.layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'swish':
            self.activation = self.swish
        elif activation == 'sinc':
            self.activation = self.sinc
        else:
            raise ValueError(f'{activation} is not supported')

    @staticmethod
    def swish(x):
        return x * torch.sigmoid(x)

    @staticmethod
    def sinc(x):
        # Condition for handling the case when x is zero
        condition = torch.eq(x, 0.0)
        return torch.where(condition, torch.ones_like(x), torch.sin(x) / x)

    def forward(self, x, error_signal):
        '''
        Args:
            - x : (batch size, x_grid, y_grid, c)
            - error_signal: (batch size, x_grid, y_grid) indicating errors per position
        Returns:
            - x: (batch size, x_grid, y_grid, 1)
        '''
        length = len(self.ws)
        batchsize = x.shape[0]
        nx, ny = x.shape[1], x.shape[2]  # original shape
        x = F.pad(x, self.padding, "constant", 0)
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # (batch, channels, x, y)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            if i < length - 1:
                # Pass the corresponding region's error_signal
                # For simplicity, assuming error_signal is global; adapt as needed
                x1 = speconv(x, error_signal)
                x2 = w(x.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_x, size_y)
                x = x1 + x2
                x = self.activation(x)
            else:
                # Last layer without activation
                x1 = speconv(x, error_signal).reshape(batchsize, self.layers[-1], -1)
                x2 = w(x.view(batchsize, self.layers[-1], -1)).reshape(batchsize, self.layers[-1], size_x, size_y)
                x = x1 + x2

        x = x.permute(0, 2, 3, 1)  # (batch, x, y, channels)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = x.reshape(batchsize, size_x, size_y, self.out_dim)
        x = x[..., :nx, :ny, :]  # Remove padding
        return x


class PINO2d(nn.Module):
    def __init__(self, modes1, modes2, width, layers=None, in_dim=3, out_dim=1,
                 activation='tanh'):
        '''
        Args:
            modes1: number of modes to keep
            modes2: number of modes to keep
            width: width of features
            layers: list of integers
            in_dim: input dimensionality, default: a(x), x, y
            out_dim: output dimensionality, default: u(x,y)
            activation: activation function
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

        self.sp_convs = nn.ModuleList([SpectralConv2d(
            in_size, out_size, mode1_num, mode2_num)
            for in_size, out_size, mode1_num, mode2_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers[:-1], self.layers[1:-1])])
        self.ws.append(LowRank2d(self.layers[-2], self.layers[-1]))

        self.fc1 = nn.Linear(self.layers[-1], self.layers[-1] * 4)
        self.fc2 = nn.Linear(self.layers[-1] * 4, out_dim)

        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'swish':
            self.activation = self.swish
        elif activation == 'sinc':
            self.activation = self.sinc
        else:
            raise ValueError(f'{activation} is not supported')

    @staticmethod
    def swish(x):
        return x * torch.sigmoid(x)

    @staticmethod
    def sinc(x):
        # Condition for handling the case when x is zero
        condition = torch.eq(x, 0.0)
        return torch.where(condition, torch.ones_like(x), torch.sin(x) / x)

    def forward(self, x, error_signal, y=None):
        '''
        Args:
            - x : (batch size, x_grid, y_grid, c)
            - error_signal: (batch size, x_grid, y_grid) indicating errors per position
            - y : Additional input if needed (optional)
        Returns:
            - x: (batch size, x_grid, y_grid, 1)
        '''
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        length = len(self.ws)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # (batch, channels, x, y)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            if i < length - 1:
                # Pass the corresponding region's error_signal
                # For simplicity, assuming error_signal is global; adapt as needed
                x1 = speconv(x, error_signal)
                x2 = w(x.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_x, size_y)
                if x1.shape[3] < x2.shape[3]:
                    x1 = F.pad(x1, (0, x2.shape[3] - x1.shape[3], 0, 0, 0, 0))
                else:
                    x2 = F.pad(x2, (0, x1.shape[3] - x2.shape[3], 0, 0, 0, 0))
                x = x1 + x2
                x = self.activation(x)
            else:
                # Last layer
                x1 = speconv(x, error_signal).reshape(batchsize, self.layers[-1], -1)
                x2 = w(x.view(batchsize, self.layers[-1], -1)).reshape(batchsize, self.layers[-1], size_x, size_y)
                if x1.shape[3] < x2.shape[3]:
                    x1 = F.pad(x1, (0, x2.shape[3] - x1.shape[3], 0, 0, 0, 0))
                else:
                    x2 = F.pad(x2, (0, x1.shape[3] - x2.shape[3], 0, 0, 0, 0))
                x = x1 + x2

        x = x.permute(0, 2, 1)  # Adjust dimensions as needed
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class FNN2d_AD(nn.Module):
    def __init__(self, modes1, modes2,
                 width=64, fc_dim=128,
                 layers=None,
                 in_dim=3, out_dim=1,
                 activation='tanh',
                 pad_x=0, pad_y=0):
        super(FNN2d_AD, self).__init__()

        """
        The overall network with Adaptive Spectral Convolution. It contains multiple layers of the SpectralConv2d.
        1. Lift the input to the desired channel dimension by self.fc0.
        2. Multiple layers of the integral operators u' = (W + K)(u).
           W defined by self.ws; K defined by self.sp_convs.
        3. Project from the channel space to the output space by self.fc1 and self.fc2.

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = (0, 0, 0, pad_y, 0, pad_x)
        # input channel is 3: (a(x, y), x, y)
        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, self.layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv2d(
            in_size, out_size, mode1_num, mode2_num)
            for in_size, out_size, mode1_num, mode2_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers[:-1], self.layers[1:-1])])
        self.ws.append(LowRank2d(self.layers[-2], self.layers[-1]))

        self.fc1 = nn.Linear(self.layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)

        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'swish':
            self.activation = self.swish
        elif activation == 'sinc':
            self.activation = self.sinc
        else:
            raise ValueError(f'{activation} is not supported')

    @staticmethod
    def swish(x):
        return x * torch.sigmoid(x)

    @staticmethod
    def sinc(x):
        # Condition for handling the case when x is zero
        condition = torch.eq(x, 0.0)
        return torch.where(condition, torch.ones_like(x), torch.sin(x) / x)

    def forward(self, x, error_signal, y=None):
        '''
        Args:
            - x : (batch size, x_grid, y_grid, c)
            - error_signal: (batch size, x_grid, y_grid) indicating errors per position
            - y : Additional input if needed (optional)
        Returns:
            - x: (batch size, x_grid, y_grid, 1)
        '''
        length = len(self.ws)
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # (batch, channels, x, y)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            if i < length - 1:
                # Pass the corresponding region's error_signal
                # For simplicity, assuming error_signal is global; adapt as needed
                x1 = speconv(x, error_signal)
                x2 = w(x.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_x, size_y)
                x = x1 + x2
                x = self.activation(x)
            else:
                # Last layer
                x1 = speconv(x, error_signal).reshape(batchsize, self.layers[-1], -1)
                x2 = w(x.view(batchsize, self.layers[-1], -1)).reshape(batchsize, self.layers[-1], size_x, size_y)
                x = x1 + x2

        # x = x.permute(0, 2, 3, 1)  # Uncomment if needed
        x = x.permute(0, 2, 1)  # Adjust dimensions as needed
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


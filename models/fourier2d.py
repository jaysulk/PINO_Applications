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
                 pad_x=0, pad_y=0,
                 num_iterations=5):  # Number of von Neumann iterations
        super(FNN2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.padding = (0, 0, 0, pad_y, 0, pad_x)
        self.num_iterations = num_iterations

        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv2d(
            in_size, out_size, mode1_num, mode2_num)
            for in_size, out_size, mode1_num, mode2_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])

        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)

        if activation =='tanh':
            self.activation = F.tanh
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation == F.relu
        elif activation == 'swish':
            self.activation = self.swish
        else:
            raise ValueError(f'{activation} is not supported')

    @staticmethod
    def swish(x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        length = len(self.ws)
        batchsize = x.shape[0]
        nx, ny = x.shape[1], x.shape[2]
        x = F.pad(x, self.padding, "constant", 0)
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        # von Neumann iterations
        u_prev = x.clone()  # Initialize with input

        for _ in range(self.num_iterations):
            for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
                x1 = speconv(u_prev)
                x2 = w(u_prev.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_x, size_y)
                u_next = x1 + x2  # von Neumann update: u^{(n+1)} = \mathcal{O}(u^{(n)}) + u^{(n)}
                if i != length - 1:
                    u_next = self.activation(u_next)
                u_prev = u_next  # Update the previous state
        x = u_next.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = x.reshape(batchsize, size_x, size_y, self.out_dim)
        x = x[..., :nx, :ny, :]
        return x

class PINO2d(nn.Module):
    def __init__(self, modes1, modes2, width, layers=None, in_dim=3, out_dim=1, num_iterations=5):
        super(PINO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_iterations = num_iterations

        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv2d(
            in_size, out_size, mode1_num, mode2_num)
            for in_size, out_size, mode1_num, mode2_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers[:-1], self.layers[1:-1])])
        self.ws.append(LowRank2d(self.layers[-2], self.layers[-1]))

        self.fc1 = nn.Linear(layers[-1], layers[-1] * 4)
        self.fc2 = nn.Linear(layers[-1] * 4, out_dim)

    def forward(self, x, y=None):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        length = len(self.ws)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        # von Neumann iterations
        u_prev = x.clone()

        for _ in range(self.num_iterations):
            for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
                if i != length - 1:
                    x1 = speconv(u_prev)
                    x2 = w(u_prev.view(batchsize, self.layers[i], -1))\
                        .view(batchsize, self.layers[i+1], size_x, size_y)
                    u_next = x1 + x2
                    u_next = F.selu(u_next)
                else:
                    x1 = speconv(u_prev, y).reshape(batchsize, self.layers[-1], -1)
                    x2 = w(u_prev, y).reshape(batchsize, self.layers[-1], -1)
                    u_next = x1 + x2
                u_prev = u_next
        x = u_next.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.selu(x)
        x = self.fc2(x)
        return x

class FNN2d_AD(nn.Module):
    def __init__(self, modes1, modes2,
                 width=64, fc_dim=128,
                 layers=None,
                 in_dim=3, out_dim=1,
                 activation='tanh',
                 num_iterations=5):
        super(FNN2d_AD, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_iterations = num_iterations

        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv2d(
            in_size, out_size, mode1_num, mode2_num)
            for in_size, out_size, mode1_num, mode2_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers[:-1], self.layers[1:-1])])
        self.ws.append(LowRank2d(self.layers[-2], self.layers[-1]))

        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)

        if activation =='tanh':
            self.activation = F.tanh
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'swish':
            self.activation = self.swish
        else:
            raise ValueError(f'{activation} is not supported')

    @staticmethod
    def swish(x):
        return x * torch.sigmoid(x)

    def forward(self, x, y=None):
        length = len(self.ws)
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        # von Neumann iterations
        u_prev = x.clone()

        for _ in range(self.num_iterations):
            for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
                if i != length - 1:
                    x1 = speconv(u_prev)
                    x2 = w(u_prev.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_x, size_y)
                    u_next = x1 + x2
                    u_next = self.activation(u_next)
                else:
                    x1 = speconv(u_prev, y).reshape(batchsize, self.layers[-1], -1)
                    x2 = w(u_prev, y).reshape(batchsize, self.layers[-1], -1)
                    u_next = x1 + x2
                u_prev = u_next

        x = u_next.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

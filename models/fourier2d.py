import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
from functools import partial

from .basics import SpectralConv1d


class SineActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class FNN1d(nn.Module):
    def __init__(self, modes, width, layers=None, activation_function='sine'):
        super(FNN1d, self).__init__()

        """
        The overall network. It contains several layers of the Fourier layer.
        1. Lift the input to the desired channel dimension by self.fc0.
        2. Multiple layers of the integral operators u' = (W + K)(u),
           where W is defined by self.ws and K is defined by self.sp_convs.
        3. Project from the channel space to the output space by self.fc1 and self.fc2.
        
        Input: the solution of the initial condition and location (a(x), x)
        Input shape: (batchsize, x=s, c=2)
        Output: the solution at a later timestep
        Output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        if layers is None:
            layers = [width] * 4

        self.fc0 = nn.Linear(2, layers[0])  # Input channel is 2: (a(x), x)

        self.sp_convs = nn.ModuleList([
            SpectralConv1d(in_size, out_size, self.modes1)
            for in_size, out_size in zip(layers, layers[1:])
        ])

        self.ws = nn.ModuleList([
            nn.Conv1d(in_size, out_size, 1)
            for in_size, out_size in zip(layers, layers[1:])
        ])

        self.fc1 = nn.Linear(layers[-1], 128)
        self.fc2 = nn.Linear(128, 1)

        # Define the activation function
        self.activation = self.get_activation_function(activation_function)

    def get_activation_function(self, activation_function):
        if activation_function == 'sine':
            return SineActivation()
        elif activation_function == 'relu':
            return nn.ReLU()
        elif activation_function == 'softplus':
            return nn.Softplus()
        elif activation_function == 'elu':
            return nn.ELU()
        elif activation_function == 'silu':
            return nn.SiLU()  # Swish activation
        elif activation_function == 'tanh':
            return nn.Tanh()
        elif activation_function == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_function}")

    def forward(self, x):
        length = len(self.ws)

        x = self.fc0(x)
        x = x.permute(0, 2, 1)  # Change to shape [batchsize, channels, x]

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x)
            x = x1 + x2
            if i != length - 1:
                x = self.activation(x)

        x = x.permute(0, 2, 1)  # Change back to shape [batchsize, x, channels]
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

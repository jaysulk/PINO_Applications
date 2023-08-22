from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import os
# import jax
# import jax.numpy as jnp
import numpy as np
import torch
# from jax import random, grad, vmap, jit, hessian, value_and_grad
# from jax.experimental import optimizers
# from jax.experimental.optimizers import adam, exponential_decay
# from jax.experimental.ode import odeint
# from jax.nn import relu, elu, softplus
# from jax.config import config
# # from jax.ops import index_update, index
# from jax import lax
# from jax.lax import while_loop, scan, cond, fori_loop
# from jax.flatten_util import ravel_pytree

import itertools
from functools import partial
from torch.utils import data
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import scipy
import scipy.io
from scipy.io import loadmat

import sys
import h5py


class Thermo1D():
    def __init__(self,
                xmin=0,
                xmax=1,
                Nx=100,
                alpha=0.01,    # this might be unnecessary if it isn't used elsewhere, or could be the diffusion coefficient if present
                dt=1e-3,
                tend=1.0,
                device=None,
                dtype=torch.float64,
                u_velocity=1.0,  # Assuming constant advection velocity for simplicity
                Q=1.0           # Assuming constant Q for simplicity
                ):
        self.xmin = xmin
        self.xmax = xmax
        self.Nx = Nx
        x = torch.linspace(xmin, xmax, Nx + 1, device=device, dtype=dtype)[:-1]
        self.x = x
        self.dx = x[1] - x[0]
        self.alpha = alpha
        self.u = torch.zeros_like(x, device=device)   # this represents T now, might be more clear to rename it to self.T
        self.u0 = torch.zeros_like(self.u, device=device)  # this represents T0 now, might be more clear to rename it to self.T0
        self.dt = dt
        self.tend = tend
        self.t = 0
        self.it = 0
        self.U = []  # Represents temperature evolution, might be more clear to rename to self.T_evolution or similar
        self.T = []  # Represents time points, it's fine to keep as is
        self.device = device
        self.u_velocity = u_velocity
        self.Q = Q




    # All Central Differencing Functions are 4th order.  These are used to compute ann inputs.
    def CD_i(self, data, axis, dx):
        data_m2 = torch.roll(data,shifts=2,dims=axis)
        data_m1 = torch.roll(data,shifts=1,dims=axis)
        data_p1 = torch.roll(data,shifts=-1,dims=axis)
        data_p2 = torch.roll(data,shifts=-2,dims=axis)
        data_diff_i = (data_m2 - 8.0*data_m1 + 8.0*data_p1 - data_p2)/(12.0*dx)
        return data_diff_i

    def CD_ij(self, data, axis_i, axis_j, dx, dy):
        data_diff_i = self.CD_i(data,axis_i,dx)
        data_diff_ij = self.CD_i(data_diff_i,axis_j,dy)
        return data_diff_ij

    def CD_ii(self, data, axis, dx):
        data_m2 = torch.roll(data,shifts=2,dims=axis)
        data_m1 = torch.roll(data,shifts=1,dims=axis)
        data_p1 = torch.roll(data,shifts=-1,dims=axis)
        data_p2 = torch.roll(data,shifts=-2,dims=axis)
        data_diff_ii = (-data_m2 + 16.0*data_m1 - 30.0*data + 16.0*data_p1 -data_p2)/(12.0*dx**2)
        return data_diff_ii

    def Dx(self, data):
        data_dx = self.CD_i(data=data, axis=0, dx=self.dx)
        return data_dx
    
    # def Dy(self, data):
    #     data_dy = self.CD_i(data=data, axis=1, dx=self.dy)
    #     return data_dy

    def Dxx(self, data):
        data_dxx = self.CD_ii(data, axis=0, dx=self.dx)
        return data_dxx

    # def Dyy(self, data):
    #     data_dyy = self.CD_ii(data, axis=1, dx=self.dy)
    #     return data_dyy
    

    


    def thermo_calc_RHS(self, T):
        T_x = self.Dx(T)
        T_RHS = -self.u_velocity * T_x + self.Q
        return T_RHS
        
    def update_field(self, field, RHS, step_frac):
        field_new = field + self.dt * step_frac * RHS
        return field_new
        

    def rk4_merge_RHS(self, field, RHS1, RHS2, RHS3, RHS4):
        field_new = field + self.dt / 6.0 * (RHS1 + 2 * RHS2 + 2.0 * RHS3 + RHS4)
        return field_new

    def thermo_rk4(self, T, t=0):
        T_RHS1 = self.thermo_calc_RHS(T)
        t1 = t + 0.5 * self.dt
        T1 = self.update_field(T, T_RHS1, step_frac=0.5)
        
        T_RHS2 = self.thermo_calc_RHS(T1)
        t2 = t + 0.5 * self.dt
        T2 = self.update_field(T, T_RHS2, step_frac=0.5)
        
        T_RHS3 = self.thermo_calc_RHS(T2)
        t3 = t + self.dt
        T3 = self.update_field(T, T_RHS3, step_frac=1.0)
        
        T_RHS4 = self.thermo_calc_RHS(T3)
        
        t_new = t + self.dt
        T_new = self.rk4_merge_RHS(T, T_RHS1, T_RHS2, T_RHS3, T_RHS4)
        
        return T_new, t_new

    
    def plot_data(self, cmap='jet', vmin=None, vmax=None, fig_num=0, title='', xlabel='', ylabel=''):
        plt.ion()
        fig = plt.figure(fig_num)
        plt.cla()
        plt.clf()
        plt.plot(self.x, self.u)
        # c = plt.pcolormesh(self.X, self.Y, self.u, cmap=cmap, vmin=vmin, vmax=vmax, shading='gouraud')
        # fig.colorbar(c)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.axis('equal')
        # plt.axis('square')
        plt.draw() 
        plt.pause(1e-17)
        plt.show()

        
    def thermo_driver(self, T0, save_interval=10, plot_interval=0):
        self.u0 = T0[:self.Nx]
        self.u = self.u0
        self.t = 0
        self.it = 0
        self.T = []
        self.U = []
        
        if plot_interval != 0 and self.it % plot_interval == 0:
            self.plot_data(vmin=-1,vmax=1,title=r'\{T}')
        if save_interval != 0 and self.it % save_interval == 0:
            self.U.append(self.u)
            self.T.append(self.t)
            
        while self.t < self.tend:
            self.u, self.t = self.thermo_rk4(self.u, self.t)
            
            self.it += 1
            if plot_interval != 0 and self.it % plot_interval == 0:
                self.plot_data(vmin=-1,vmax=1,title=r'\{T}')
            if save_interval != 0 and self.it % save_interval == 0:
                self.U.append(self.u)
                self.T.append(self.t)

        return torch.stack(self.U)
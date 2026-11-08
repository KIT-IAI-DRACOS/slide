# SPDX-FileCopyrightText: 2019-2025 Gerben I. Beintema & Maarten Schoukens
# SPDX-License-Identifier: BSD-3-Clause
#
# This file is based on code from deepSI:
# https://github.com/MaartenSchoukens/deepSI

import torch
import torch.nn as nn

import numpy as np

##############################
#### Discrete-Time SUBNET ####
##############################
# see: https://proceedings.mlr.press/v144/beintema21a/beintema21a.pdf or 
# Beintema, Gerben, Roland Toth, and Maarten Schoukens. "Nonlinear state-space identification using deep encoder networks." Learning for dynamics and control. PMLR, 2021.
# see: https://www.sciencedirect.com/science/article/pii/S0005109823003710
# Beintema, Gerben I., Maarten Schoukens, and Roland Tóth. "Deep subspace encoders for nonlinear system identification." Automatica 156 (2023): 111210.

class SUBNET(nn.Module):
    def __init__(
            self, 
            nu:int|str, 
            ny:int|str, 
            # norm : Norm, 
            nx:int=10, 
            nb:int=20, 
            na:int=20, 
            feedthrough=False,
            # validate=True
        ) -> None:
        super().__init__()
        # self.nu, self.ny, self.norm, self.nx, self.nb, self.na, self.feedthrough = nu, ny, norm, nx, nb, na, feedthrough
        # self.f = f if f is not None else norm.f(MLP_res_net(input_size = [nx , nu], output_size = nx))
        # self.h = h if h is not None else norm.h(MLP_res_net(input_size = [nx , nu] if feedthrough else nx, output_size = ny))
        # self.encoder = encoder if encoder is not None else norm.encoder(MLP_res_net(input_size = [(nb,nu) , (na,ny)], output_size = nx))
        self.nu, self.ny, self.nx, self.nb, self.na, self.feedthrough = nu, ny, nx, nb, na, feedthrough
        self.f = MLP_res_net(input_size = [nx , nu], output_size = nx)
        self.h = MLP_res_net(input_size = [nx , nu] if feedthrough else nx, output_size = ny)
        self.encoder = MLP_res_net(input_size = [(nb,nu) , (na,ny)], output_size = nx)
        
        # if validate:
        #     validate_SUBNET_structure(self)

    # def create_arrays(self, data: Input_output_data | list, T : int=50, stride: int=1):
    #     return past_future_arrays(data, self.na, self.nb, T=T, stride=stride)

    # def forward_simple(self, upast: torch.Tensor, ypast: torch.Tensor, ufuture: torch.Tensor, yfuture: torch.Tensor=None):
    #     #is a lot simplier but also about 50% slower
    #     yfuture_sim = []
    #     x = self.encoder(upast, ypast)
    #     for u in ufuture.swapaxes(0,1):
    #         y = self.h(x,u) if self.feedthrough else self.h(x)
    #         yfuture_sim.append(y)
    #         x = self.f(x,u)
    #     return torch.stack(yfuture_sim, dim=1)

    def forward(self, upast: torch.Tensor, ypast: torch.Tensor, ufuture: torch.Tensor, yfuture: torch.Tensor=None):
        B, T = ufuture.shape[:2]
        x = self.encoder(upast, ypast)
        xfuture = []
        for u in ufuture.swapaxes(0,1): #unroll over time dim
            xfuture.append(x)
            x = self.f(x,u)
        xfuture = torch.stack(xfuture,dim=1) #has shape (Nbatch, Ntime=T, nx)

        #compute output at all the future time indicies at the same time by combining the time and batch dim.
        fl = lambda ar: torch.flatten(ar, start_dim=0, end_dim=1) #conbine batch dim and time dim (Nbatch, Ntime, ...) -> (Nbatch*Ntim, ...)
        yfuture_sim_flat = self.h(fl(xfuture), fl(ufuture)) if self.feedthrough else self.h(fl(xfuture)) #compute the output for all time and and batches in one go
        return torch.unflatten(yfuture_sim_flat, dim=0, sizes=(B,T)) #(Nbatch*T, ...) -> (Nbatch, T, ...)

    # def simulate(self, data: Input_output_data | list):
    #     if isinstance(data, (list, tuple)):
    #         return [self.simulate(d) for d in data]
    #     if data.sampling_time!=self.norm.sampling_time:
    #         warn('It seems that the model is being simulated at a different sampling time as it was trained on.')
    #     ysim = self(*past_future_arrays(data, self.na, self.nb, T='sim', add_sampling_time=False)[0])[0].detach().numpy()
    #     return Input_output_data(u=data.u, y=np.concatenate([data.y[:max(self.na, self.nb)],ysim],axis=0), state_initialization_window_length=max(self.na, self.nb))

    # def f_unbached(self, x, u):
    #     return self.f(x[None],u[None])[0]
    # def h_unbached(self, x, u=None):
    #     return self.h(x[None], u[None])[0] if self.feedthrough else self.h(x[None])[0]
    # def encoder_unbached(self, upast, ypast):
    #     return self.encoder(upast[None],ypast[None])[0]
    
################################
#### Continuous-Time SUBNET ####
################################
# see: https://arxiv.org/abs/2204.09405
#  Beintema, G. I., Schoukens, M., & Tóth, R. (2022). Continuous-time identification  of dynamic state-space models by deep subspace encoding. Presented at the 11th International Conference on Learning Representations (ICLR)

class SUBNET_CT(nn.Module):
    #both norm, base_sampling_time have a sample time 
    # def __init__(self, nu, ny, norm:Norm, nx=10, nb=20, na=20, f_CT=None, h=None, encoder=None, integrator=None, feedthrough=False, validate=True) -> None:
    def __init__(self, nu, ny, nx=10, nb=20, na=20, f_CT=None, feedthrough=False) -> None:
        super().__init__()
        self.nu, self.ny, self.nx, self.nb, self.na, self.feedthrough = nu, ny, nx, nb, na, feedthrough
        self.f_CT = MLP_res_net(input_size = [nx , nu], output_size = nx, n_hidden_nodes=nx*2)
        # self.f_CT = MLP_res_net(input_size = [nx , nu], output_size = nx)
        self.h = MLP_res_net(input_size = [nx , nu] if feedthrough else nx, output_size = ny, n_hidden_nodes=nx*2)
        # self.h = MLP_res_net(input_size = [nx , nu] if feedthrough else nx, output_size = ny)
        self.encoder = MLP_res_net(input_size = [(nb,nu) , (na,ny)], output_size = nx, n_hidden_nodes=nx*2)
        # self.encoder = MLP_res_net(input_size = [(nb,nu) , (na,ny)], output_size = nx)
        self.integrator = rk4_integrator
        # if validate:
        #     validate_SUBNET_structure(self)

    # def create_arrays(self, data: Input_output_data | list, T : int=50, stride: int=1):
    #     return past_future_arrays(data, self.na, self.nb, T=T, stride=stride, add_sampling_time=True)

    def forward(self, upast: torch.Tensor, ypast: torch.Tensor, ufuture: torch.Tensor, sampling_time : float | torch.Tensor, yfuture: torch.Tensor=None):
        B, T = ufuture.shape[:2]
        x = self.encoder(upast, ypast)
        xfuture = []
        for u in ufuture.swapaxes(0,1):
            xfuture.append(x)
            x = self.integrator(self.f_CT, x, u, sampling_time)
        xfuture = torch.stack(xfuture,dim=1) #has shape (Nbatch, Ntime=T, nx)

        #compute output at all the future time indicies at the same time by combining the time and batch dim.
        fl = lambda ar: torch.flatten(ar, start_dim=0, end_dim=1) #conbine batch dim and time dim 
        yfuture_sim_flat = self.h(fl(xfuture), fl(ufuture)) if self.feedthrough else self.h(fl(xfuture)) #compute the output for all time and and batches in one go
        return torch.unflatten(yfuture_sim_flat, dim=0, sizes=(B,T)) #(Nbatch*T) -> (Nbatch, T)
    
    # def simulate(self, data: Input_output_data | list):
    #     if isinstance(data, (list, tuple)):
    #         return [self.simulate(d) for d in data]
    #     if data.sampling_time!=self.norm.sampling_time:
    #         warn('It seems that the model is being simulated at a different sampling time as it was trained on. The encoder currently assumes that the sampling_time is kept constant')
    #     ysim = self(*past_future_arrays(data, self.na, self.nb, T='sim', add_sampling_time=True)[0])[0].detach().numpy()
    #     return Input_output_data(u=data.u, y=np.concatenate([data.y[:max(self.na, self.nb)],ysim],axis=0), state_initialization_window_length=max(self.na, self.nb))

    # def f_CT_unbached(self, x, u):
    #     return self.f_CT(x[None],u[None])[0]
    # def integrator_unbached(self, f_CT, x, u, sampling_time):
    #     return self.integrator(f_CT, x[None], u[None], sampling_time[None])[0]
    # def h_unbached(self, x, u=None):
    #     return self.h(x[None], u[None])[0] if self.feedthrough else self.h(x[None])[0]
    # def encoder_unbached(self, upast, ypast):
    #     return self.encoder(upast[None],ypast[None])[0]


##############################
####### Koopman SUBNET #######
##############################
# see: https://ieeexplore.ieee.org/abstract/document/9682946
# Iacob, Lucian Cristian, et al. "Deep identification of nonlinear systems in Koopman form." 2021 60th IEEE Conference on Decision and Control (CDC). IEEE, 2021.

# class Koopman_SUBNET(Custom_SUBNET):
class Koopman_SUBNET(nn.Module):
    '''Implements the following structure
    x_next = A@x + B(x)@(u - umean)/ustd
    y = (C@x) * ystd + ymean

    if feedthrough: y = (C@x) * ystd + ymean + (D@(u - umean)/ustd) * ystd + ymean 
    if B_depends_on_u: x_next = A@x + B(x, u)@(u - umean)/ustd
        
    '''
    # def __init__(self, nu, ny, norm : Norm, nx, nb, na, encoder=None, A=None, Bnet=None, C=None, D=None, B_depends_on_u=False, feedthrough=False):
    def __init__(self, nu, ny, nx, nb, na, B_depends_on_u=False, feedthrough=False):
        super().__init__()
        # from deepSI.networks import Matrix_converter, MLP_res_net
        self.nu, self.ny, self.nx, self.na, self.nb, self.feedthrough, self.B_depends_on_u = nu, ny, nx, na, nb, feedthrough, B_depends_on_u
        self.nu_vals = 1 if nu=='scalar' else nu
        self.ny_vals = 1 if ny=='scalar' else ny
        self.encoder = MLP_res_net(input_size = [(nb,nu) , (na,ny)], output_size = nx, n_hidden_nodes=nx*2)
        # self.encoder = MLP_res_net(input_size = [(nb,nu) , (na,ny)], output_size = nx)

        self.A = nn.Parameter(torch.randn((nx,nx))/(2*nx**0.5))
        self.Bnet = Matrix_converter(MLP_res_net([nx, nu] if B_depends_on_u else nx, nx*self.nu_vals, n_hidden_nodes=nx*2), nrows=nx, ncols=self.nu_vals)
        # self.Bnet = Matrix_converter(MLP_res_net([nx, nu] if B_depends_on_u else nx, nx*self.nu_vals), nrows=nx, ncols=self.nu_vals)
        self.C = nn.Parameter(torch.randn((self.ny_vals,nx))/(2*nx**0.5))
        if feedthrough:
            self.D = nn.Parameter(torch.randn((self.ny_vals,self.nu_vals))/(2*self.nu_vals**0.5))
        else:
            self.D = None

    def forward(self, upast: torch.Tensor, ypast: torch.Tensor, ufuture: torch.Tensor, yfuture: torch.Tensor=None):
        mv = lambda A, x: torch.bmm(A, x[:, :, None])[:,:,0] #batched matrix vector multiply
        yfuture_sim = []
        x = self.encoder(upast, ypast) #initial state
        Nbatch = upast.shape[0]
        # ufuture = (ufuture - self.norm.umean)/self.norm.ustd # Normalize input
        ufuture = ufuture.view(Nbatch, ufuture.shape[1], -1) # Convert all the u from scalars to vectors if needed
        # Add batch dimension to matrixes
        A = torch.broadcast_to(self.A, (Nbatch, self.nx, self.nx)) 
        C = torch.broadcast_to(self.C, (Nbatch, self.ny_vals, self.nx))
        D = None if self.feedthrough==False else torch.broadcast_to(self.D, (Nbatch, self.ny_vals, self.nu_vals))
        for u in ufuture.swapaxes(0,1): #iterate over time
            y = mv(C, x) + (0 if self.feedthrough==False else mv(D, u))
            yfuture_sim.append(y)
            B = self.Bnet(x) if self.B_depends_on_u==False else self.Bnet(x, u[:,0] if self.nu=='scalar' else u) #removes the vector dim if it is scalar
            x = mv(A,x) + mv(B, u)
        yfuture_sim = torch.stack(yfuture_sim, dim=1)
        if self.ny=='scalar':
            yfuture_sim = yfuture_sim[:,:,0]
        # return yfuture_sim*self.norm.ystd + self.norm.ymean
        return yfuture_sim



###########################################################
#### Multi layer peceptron/feed forward neural network ####
###########################################################

class MLP_res_net(nn.Module):
    '''Multi-Layer Perceptron with Residual Connection (MLP_res_net) as follows:
              y_pred = net(input) = net_MLP(input) + A * input
              where net_MLP(input) is a simple Multi-Layer Perceptron, e.g.:
                h_1 = input
                h-2 = activation(A_1 h_1 + b_1) #A_1.shape = n_hidden_nodes x input_size
                h_3 = activation(A_2 h_2 + b_2) #A_2.shape = n_hidden_nodes x n_hidden_nodes
                ...
                h_n_hidden_layers = activation(A_n-1 h_n-1 + b_n-1)
                return h_n_hidden_layers
    '''
    def __init__(self, input_size: str | int | list, output_size: str | int | list, n_hidden_layers = 2, n_hidden_nodes = 64, \
                 activation=nn.Tanh, zero_bias=True):
        self.input_size = input_size
        self.output_size = output_size
        super().__init__()
        self.scalar_output = output_size=='scalar'
        #convert input shape:
        def to_num(s):
            if isinstance(s, int):
                return s
            if s=='scalar':
                return 1
            a = 1
            for si in s:
                a = a*(1 if si=='scalar' else si)
            return a
        if isinstance(input_size, list):
            input_size = sum(to_num(s) for s in input_size)
        
        output_size = 1 if self.scalar_output else output_size
        self.net_res = nn.Linear(input_size, output_size)

        seq = [nn.Linear(input_size,n_hidden_nodes),activation()]
        for i in range(n_hidden_layers-1):
            seq.append(nn.Linear(n_hidden_nodes,n_hidden_nodes))
            seq.append(activation())
        seq.append(nn.Linear(n_hidden_nodes,output_size))
        self.net_nonlin = nn.Sequential(*seq)

        if zero_bias:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, val=0) #bias
        
    def forward(self, *ars):
        if len(ars)==1:
            net_in = ars[0]
            net_in = net_in.view(net_in.shape[0], -1) #adds a dim when needed
        else:
            net_in = torch.cat([a.view(a.shape[0], -1) for a in ars],dim=1) #flattens everything
        out = self.net_nonlin(net_in) + self.net_res(net_in)
        return out[:,0] if self.scalar_output else out

def rk4_integrator(f, x, u, dt, n_steps=1):
    tau=dt*50
    dtp = (dt/n_steps)
    # [:,None]
    for _ in range(n_steps): #f(x,u) has shape (nbatch, nx)
        k1 = dtp * (f(x,u)/tau)
        k2 = dtp * (f(x+k1*0.5,u)/tau)
        k3 = dtp * (f(x+k2*0.5,u)/tau)
        k4 = dtp * (f(x+k3,u)/tau)
        x = x + (k1+2*k2+2*k3+k4)/6
    return x

class Matrix_converter(nn.Module):
    '''
    Converts a net(x) vector to a matrix using a reshape
    '''
    def __init__(self, net, nrows, ncols, norm='auto'):
        super().__init__()
        self.net = net
        self.norm = norm
        self.nrows = nrows
        self.ncols = ncols

    def forward(self, *x):
        A = self.net(*x).view(x[0].shape[0], self.nrows, self.ncols)
        if self.norm=='auto':
            A = A/(self.ncols**0.5) #this can be improved with some additional math 
        else:
            A = A/self.norm #this can be improved with some additional math 
        return A

# C = lambda x: torch.as_tensor(x, dtype=torch.float32) if x is not None else None
# class IO_normalization_f(torch.nn.Module):
#     def __init__(self, fun, umean, ustd):
#         super().__init__()
#         self.fun, self.umean, self.ustd = fun, C(umean), C(ustd)    
#     def forward(self, x, u):
#         return self.fun(x, (u-self.umean)/self.ustd)

# class IO_normalization_f_CT(torch.nn.Module):
#     def __init__(self, fun, umean, ustd, tau):
#         super().__init__()
#         self.fun, self.umean, self.ustd, self.tau = fun, C(umean), C(ustd), C(tau)
#     def forward(self, x, u):
#         return self.fun(x, (u-self.umean)/self.ustd)/self.tau

# class IO_normalization_h(torch.nn.Module):
#     def __init__(self, fun, umean, ustd, ymean, ystd):
#         super().__init__()
#         self.fun, self.umean, self.ustd, self.ymean, self.ystd = fun, C(umean), C(ustd), C(ymean), C(ystd)
#     def forward(self, x, u=None):
#         if u is None:
#             y_normed = self.fun(x)
#         else:
#             y_normed = self.fun(x, (u-self.umean)/self.ustd)
#         return y_normed*self.ystd + self.ymean

# class IO_normalization_encoder(torch.nn.Module):
#     def __init__(self, fun, umean, ustd, ymean, ystd):
#         super().__init__()
#         self.fun, self.umean, self.ustd, self.ymean, self.ystd = fun, C(umean), C(ustd), C(ymean), C(ystd)
#     def forward(self, upast, ypast):
#         return self.fun((upast-self.umean)/self.ustd, (ypast-self.ymean)/self.ystd)

# class Norm:
#     def __init__(self, umean, ustd, ymean, ystd, sampling_time=1):
#         self.umean, self.ustd, self.ymean, self.ystd = C(umean), C(ustd), C(ymean), C(ystd)
#         self.sampling_time = C(sampling_time)
    
#     def f(self, fun):
#         return IO_normalization_f(fun, self.umean, self.ustd)
#     def h(self, fun):
#         return IO_normalization_h(fun, self.umean, self.ustd, self.ymean, self.ystd)
#     def encoder(self, fun):
#         return IO_normalization_encoder(fun, self.umean, self.ustd, self.ymean, self.ystd)
#     def f_CT(self, fun, tau):
#         return IO_normalization_f_CT(fun, self.umean, self.ustd, tau)

#     def transform(self, dataset : nlb.Input_output_data | list):
#         if isinstance(dataset, (list, tuple)):
#             return [self.transform(d) for d in dataset]
#         u = (dataset.u - self.umean.numpy())/self.ustd.numpy()
#         y = (dataset.y - self.ymean.numpy())/self.ystd.numpy()
#         sampling_time = None if dataset.sampling_time is None else dataset.sampling_time/self.sampling_time.item()
#         return nlb.Input_output_data(u, y, sampling_time=sampling_time, name=f'{dataset.name}-normed', \
#                                      state_initialization_window_length=dataset.state_initialization_window_length)

#     def __repr__(self):
#         return (f"Norm(umean={self.umean.numpy()}, ustd={self.ustd.numpy()}, "
#                 f"ymean={self.ymean.numpy()}, ystd={self.ystd.numpy()}, "
#                 f"sampling_time={self.sampling_time.numpy()})")
# SPDX-FileCopyrightText: 2025 Zhaoyang Li, Minghao Han, Xunyuan Yin
# SPDX-License-Identifier: CC-BY-4.0
#
# This file is based on code from the supplementary material of:
# Li, Han, and Yin, "MamKO: Mamba-based Koopman operator for
# modeling and predictive control", ICLR 2025.
# https://openreview.net/forum?id=hNjCVVm0EQ

import torch
import numpy as np

import torch.nn as nn
from einops import rearrange, repeat, einsum
import torch.nn.functional as F
import torch.nn.init as init

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.normal_(m.bias, mean=0,std=0.1)
  

class Mamba(nn.Module):
    """
    use some concept of Mamba, design this block
    """
    def __init__(self, args):
        """A mamba model for LTV system"""
        super().__init__()
        self.L = args['pred_horizon']
        self.O = args['old_horizon']
        self.N = args['latent_dim']
        self.S = args['state_dim']
        self.d_conv = args['d_conv']
        self.U = args['act_dim']
        self.delta = args['delta']

        # self.device = args['device']
        #---------------------------------without initilization---------------------------#
        # self.A = torch.abs(torch.randn(self.N, 1)) # seems only on the diagnoal => connection between each state???]

        self.A = torch.randn(self.N, 1)/20# TODO: This initial seems better
        self.A = nn.Parameter(self.A) # Stable seems like
        # self.A_ = self.P_inv*torch.diag(self.A.squeeze())*self.P_
        #---------------------------------without initilization---------------------------#
        self.x_project  = nn.Linear(self.S+self.U,self.N) # x from original d to N dimension koopman
        # self.x_project  = nn.Linear(self.S,self.N)
        self.x_project_ = nn.Linear(self.O-1,self.L)
        self.x_generate        = nn.Linear(self.N+self.U,self.delta+self.N*(self.U)+self.N*self.S)
        self.x_generate_deleta = nn.Linear(self.N,self.delta)
        self.dt_project = nn.Linear(self.delta,self.N)
         
        self.conv1d_all = nn.Conv1d(
            in_channels=self.N+self.U,
            out_channels=self.N+self.U,
            bias=True,
            kernel_size=self.d_conv,
            groups=self.N+self.U,
            padding=self.d_conv - 1, #TODO: test
        )# conv on the time sequence for states and input

        self.conv1d_states = nn.Conv1d(
        in_channels=self.N,
        out_channels=self.N,
        bias=True,
        kernel_size=self.d_conv,
        groups=self.N,
        padding=self.d_conv - 1,
        )# conv on the time sequence for all the states

        self.u_project   = nn.Linear(self.U,self.U)
        self.x_project_0 = nn.Linear(self.S,args['hidden_dim'])
        self.x_project_1 = nn.Linear(args['hidden_dim'],self.N)

        

        init.kaiming_uniform_(self.conv1d_states.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv1d_all.weight, mode='fan_in', nonlinearity='relu')

    def forward(self,x, u):
        """
        seq2seq structure for training
        old data x(0:9) u(0:9) p(0:9)             -> determine A B C delta for time 1:10      
        current data                              -> project to Z
        new data (for control) u(10:19) p(1:10)   -> project to g with linear layer and elu
        x(B,L,D_x) u(B,L*2,D_u) p(B,L*2,D)
        """
        u = u[:,:,:self.U] # TODO: FOR CONVENIENCE

        z_0   = self.project_x(x[:,self.O-1,:])
        delta, B, C = self.generate_matrices(x,u,z_0)

        deltaA,deltaB,C = self.discretion_matrices(self.A, B, C, delta)

        u_0 = self.project_u(u[:,self.O-1:,:])

        return self.selective_scan(z_0, u_0, deltaA, deltaB, C, x)

    def generate_matrices(self,x,u,z_0):
        """
        Generate delta B C via old data
        input:  old data x(0:9) u(0:9) p(0:9)
        output: delta B C
        """
        if u.dim() ==2:
            u = u.unsqueeze(2)

        x_all = self.project_x(x[:,:self.O-1,:])
        u_all = self.project_u(u[:,:self.O-1,:])
        old_data = torch.cat((x_all,u_all),dim=2)
        old_data = F.silu(old_data)
        old_data = rearrange(old_data, 'B L D -> B D L')
        old_data = self.conv1d_all(old_data)[:,:,-self.L:]
        old_data = rearrange(old_data, 'B D L -> B L D')
        # old_data = F.silu(old_data)
        old_data = F.silu(old_data)
        
        old_data = self.x_generate(old_data) #  B L D -> B L delta+2*N
        (delta, B, C) = old_data.split(split_size=[self.delta, self.N*(self.U), self.N*self.S], dim=-1)

        delta = F.softplus(self.dt_project(delta))# B L N
        self.delta_save = delta
        batch, length , _ = B.shape
        B =  B.reshape(batch, length , self.N, self.U)
        C =  C.reshape(batch, length , self.N, self.S)

        return delta, B, C



    def selective_scan(self,z0, u, deltaA, deltaB, C,x):
        """
        z(k+1) = deltaA(k) z(k) + deltaB(k) u(k)
        x(k)   = C z(k)
        L  generate L+1
        """
        ys = []    
        # loss = nn.MSELoss()  
        # loss_ = 0
        for i in range(self.L): # The same as in the controller
            z0 = torch.mul(z0,deltaA[:,i,:])+einsum(deltaB[:,i,:],u[:,i,:],'B N L, B L -> B N')
            y = einsum(C[:,i,:,:],z0,'B L N, B L -> B N')
            # loss_+= loss(y,x[:,self.O+i,:])
            # ys.append(y.detach().cpu().numpy())
            ys.append(y)

        # return loss_/self.L, np.array(ys)
        # return np.array(ys)
        return torch.stack(ys, dim=1)
            


    def discretion_matrices(self, A, B, C, delta):
        """
        Generate A B after discretion_matrices
        delta
        """
        # A = -torch.exp(A)
        A = -F.celu(A)
        # A = -F.silu(A)
        # A = -F.celu(A)
        # A = -F.elu(A)
        # A = -F.elu(A) #TODO unstable system
        # print(A)
        deltaA = torch.exp(einsum(delta, A.squeeze(), 'B L N, N -> B L N')) # delta: B L N
        deltaA_cum = einsum((deltaA-1),(1/A).squeeze(),'B L N, N->B L N')
        deltaB     = einsum(deltaA_cum, B, 'B L N, B L N C -> B L N C')
        return deltaA, deltaB, C
        

    def project_x(self, x_0):
        """
        Project the state x into high-dimensional space
        """
        # return F.silu(self.x_project_0(x_0))
        # return F.relu(self.x_project_0(x_0))
        # return self.x_project_1(F.silu(self.x_project_0(x_0)))
        return self.x_project_1(F.relu(self.x_project_0(x_0)))

    def project_u(self, u):
        """
        Project the input u into high-dimensional space
        """
        # return F.elu(self.u_project(u))
        return u

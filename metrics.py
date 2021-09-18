import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx
from scipy.stats import gaussian_kde


def ade(predAll,targetAll,count_):
    All = len(predAll)
    sum_all = 0 
    for s in range(All):
        pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)
        target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)
        
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0 
        for i in range(N):
            for t in range(T):
                sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
        sum_all += sum_/(N*T)
        
    return sum_all/All


def fde(predAll,targetAll,count_):
    All = len(predAll)
    sum_all = 0 
    for s in range(All):
        pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)
        target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0 
        for i in range(N):
            for t in range(T-1,T):
                sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
        sum_all += sum_/(N)

    return sum_all/All


def seq_to_nodes(seq_):
    max_nodes = seq_.shape[1] #number of pedestrians in the graph
    seq_ = seq_.squeeze()
    seq_len = seq_.shape[2]
    
    V = np.zeros((seq_len,max_nodes,2))
    for s in range(seq_len):
        step_ = seq_[:,:,s]
        for h in range(len(step_)): 
            V[s,h,:] = step_[h]
            
    return V.squeeze()

def nodes_rel_to_nodes_abs(nodes,init_node):
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s,ped,:] = np.sum(nodes[:s+1,ped,:],axis=0) + init_node[ped,:]

    return nodes_.squeeze()

def closer_to_zero(current,new_v):
    dec =  min([(abs(current),current),(abs(new_v),new_v)])[1]
    if dec != current:
        return True
    else: 
        return False
        
def bivariate_loss(V_pred,V_trgt):
    #mux, muy, sx, sy, corr
    #assert V_pred.shape == V_trgt.shape
    normx = V_trgt[:,:,0]- V_pred[:,:,0]
    normy = V_trgt[:,:,1]- V_pred[:,:,1]

    sx = torch.exp(V_pred[:,:,2]) #sx
    sy = torch.exp(V_pred[:,:,3]) #sy
    corr = torch.tanh(V_pred[:,:,4]) #corr
    
    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.mean(result)
    
    return result


def kde_nll(pred, gt, log_pdf_lower_bound=-20, mode='raw', ignore_if_fail=False):
    """
    Credits: https://github.com/vita-epfl/trajnetplusplustools and https://github.com/StanfordASL/Trajectron.
    """
    pred_len = gt.shape[0]
    nll_all_ped_list = []
    for p in range(gt.shape[1]):
        ll = 0.0
        same_pred = 0
        # all predictions are the same, which can happen for constant velocity with 0 speed
        for timestep in range(gt.shape[0]):
            curr_gt, curr_pred = gt[timestep, p], pred[timestep, :, p]
            if np.all(curr_pred[1:] == curr_pred[:-1]):
                same_pred += 1
                continue  # Identical prediction at particular time-step, skip
            try:
                scipy_kde = gaussian_kde(curr_pred.T)
                # We need [0] because it's a (1,)-shaped tensor
                log_pdf = np.clip(scipy_kde.logpdf(curr_gt.T), a_min=log_pdf_lower_bound, a_max=None)[0]
                if np.isnan(log_pdf) or np.isinf(log_pdf) or log_pdf > 100:
                    same_pred += 1  # Difficulties in computing Gaussian_KDE
                    continue
                ll += log_pdf
            except Exception as e:
                same_pred += 1  # Difficulties in computing Gaussian_KDE

        if same_pred == pred_len:
            if ignore_if_fail:
                continue  # simply not being considered for computation
            else:
                raise Exception('Failed to compute KDE-NLL for one or more trajectory. To ignore the trajectories that '
                                f'result in computation failure, supply --ignore_if_kde_nll_fails.{os.linesep}WARNING! '
                                'This will mean that some samples will be ignored, which may be unfair when comparing '
                                'with other methods whose samples do not result in error.')

        ll = ll / (pred_len - same_pred)
        nll_all_ped_list.append(ll)
    nll_all_ped = np.array(nll_all_ped_list)
    return nll_all_ped if mode == 'raw' else np.sum(nll_all_ped)

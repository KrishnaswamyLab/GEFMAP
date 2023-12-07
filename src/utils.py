import os
import sys
import pickle as pk
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy
import torch
import torch_geometric


def pk_save(obj, fname):
    with open(fname, 'wb') as handle:
        pk.dump(obj, handle, protocol=pk.HIGHEST_PROTOCOL)


def pk_load(fname):
    with open(fname, 'rb') as handle:
        return pk.load(handle)
    
def check_device():
    import torch
    if torch.cuda.is_available():
        print("CUDA is available.")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    else:
        print("CUDA is not available.")

##############################################################################
##############################################################################

def min_max(X):
    maxval = torch.max(X)
    minval = torch.min(X)
    X = (X-minval)/(maxval+1e-6-minval) #1e-6 to account for zero vals
    return X

def G2edgeindex(G):
    Glist = []
    for i, edge in enumerate(list(G.edges)):
        Glist.append([edge[0],edge[1]])
    return Glist

def to_sparse_mx(edge_index, N):
    row, col = edge_index
    edge_attr = torch.ones(row.size(0))
    out = sp.coo_matrix(
        (edge_attr.numpy(), (row.numpy(), col.numpy())), (N, N))
    return out

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx  = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data) 
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



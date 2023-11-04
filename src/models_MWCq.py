from utils import *
import numpy as np

import networkx as nx
import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn import Linear
from torch import tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import sparse
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.data import Data
import torch.optim as optim
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import time
import random

#############################################
############################################# #utils


def exploss(graph, output_dis, penalty_coefficient, w_exp):
    #W = graph.W
    #W_complement = graph.W_complement
    N = graph.N
    edge_index = graph.edge_index
    adjmatrix = to_sparse_mx(edge_index, N)
    adjmatrix = sparse_mx_to_torch_sparse_tensor(adjmatrix)
    I_n = sp.eye(N)
    I_n = sparse_mx_to_torch_sparse_tensor(I_n)
    Fullm = torch.ones(I_n.size(0),I_n.size(1))- I_n


    #W = adjmatrix + I_n
    #W_complement = 1_{n x n} - (I + W)
    diffusionprob = torch.mm(Fullm - adjmatrix,output_dis)
    elewiseloss = output_dis * diffusionprob
    lossComplE = penalty_coefficient * torch.sum(elewiseloss) 
    #weighted
    ouput_wt = min_max(w_exp) * output_dis
    lossE = torch.sum(ouput_wt*torch.mm(adjmatrix,output_dis))
    loss = -lossE + lossComplE
    retdict = {}
    retdict["loss"] = [loss,lossComplE] #final loss
    return retdict


def train(model, epoch, train_loader, optimizer, penalty_coefficient, verbose = True):
    model.train()
    print('Epoch:-----%d'%epoch)
    for i, batch in enumerate(train_loader):
        batchloss = 0.0
        for j in range(len(batch)): # len(batch) len of the batch
            features = torch.FloatTensor(batch[j].x)
            w_exp = torch.tensor([feat[2].item() for feat in features]) 
            output = model(features)
            retdict = exploss(model.graph, output, penalty_coefficient, w_exp = w_exp)
            batchloss += retdict["loss"][0]
        if verbose is True:
            print("Length of batch:%d"%len(batch))
            print('Loss: %.5f'%batchloss)
        else:
            pass
        optimizer.zero_grad()
        batchloss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        optimizer.step()




def plot_support(model, batch, ax = None):
    import matplotlib.pyplot as plt
    #fig, ax = plt.subplots()
    for j in range(len(batch)):
        features = torch.FloatTensor(batch[j].x)
        output = model(features)
        outs = output.detach().numpy().reshape(-1,)
        outs = np.sort(outs)

        ax.scatter(x = np.arange(0, len(outs)), y = outs, s= 10)
        ax.set_ylabel('p*')
        ax.set_xlabel('node_id')

#############################################
############################################# #layers



def GCN_diffusion(A_gcn, A_gcn_feature, order =3):
    '''
    graph: base_graph object
    '''

    degrees = torch.sparse.sum(A_gcn,0)
    D = degrees.to_dense()
    D = torch.pow(D, -0.5)
    D = D.unsqueeze(dim=1)
    gcn_diffusion_list = []

    for i in range(order):
        A_gcn_feature = torch.mul(A_gcn_feature,D) #
        A_gcn_feature = torch.spmm(A_gcn,A_gcn_feature)
        A_gcn_feature = torch.mul(A_gcn_feature,D)
        gcn_diffusion_list += [A_gcn_feature,]
    return gcn_diffusion_list



def scattering_diffusion(adj, feature, order =3):
    degrees = torch.sparse.sum(adj,0)
    D = degrees
    D= D.to_dense()
    D = torch.pow(D, -1)
    D = D.unsqueeze(dim=1)

    iteration = 2**(order-1)
    feature_p = feature
    sct_diffusion_list = []

    for i in range(iteration):
        D_inv_x = D*feature_p
        W_D_inv_x = torch.spmm(adj,D_inv_x)
        feature_p = 0.5*feature_p + 0.5*W_D_inv_x
        sct_diffusion_list += [feature_p,]

    sct_feature1 = sct_diffusion_list[0]-sct_diffusion_list[1]
    sct_feature2 = sct_diffusion_list[1]-sct_diffusion_list[2]
    sct_feature3 = sct_diffusion_list[2]-sct_diffusion_list[3]
    return sct_feature1,sct_feature2,sct_feature3





#############################################
############################################# #models


class SCTConv(torch.nn.Module):
    def __init__(self, base_graph, hidden_dim):
        super().__init__()
        self.graph = base_graph
        self.hid = hidden_dim
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.a = Parameter(torch.zeros(size=(2*hidden_dim, 1)))

    def forward(self,X,moment = 1):
        '''
        input: X [batch x nodes x features]
        returns: X' [batch x nodes x features]
        '''
        support0 = X
        N = self.graph.N
        adj = self.graph.W
        '''
        Creating a normalized adjacency matrix with self loops.
        https://arxiv.org/pdf/1609.02907.pdf
        '''
        A_gcn = self.graph.W + self.graph.I_n
        h = support0

        #low pass
        gcn_diffusion_list = GCN_diffusion(A_gcn,support0)
        h_A =  gcn_diffusion_list[0]
        h_A2 =  gcn_diffusion_list[1]
        h_A3 =  gcn_diffusion_list[2]
        
        h_A = nn.LeakyReLU()(h_A)
        h_A2 = nn.LeakyReLU()(h_A2)
        h_A3 = nn.LeakyReLU()(h_A3)

        #bandpass filters
        h_sct1,h_sct2,h_sct3 = scattering_diffusion(adj,support0)
        h_sct1 = torch.abs(h_sct1)**moment
        h_sct2 = torch.abs(h_sct2)**moment
        h_sct3 = torch.abs(h_sct3)**moment

        #concatenate
        a_input_A = torch.hstack((h, h_A)).unsqueeze(1)
        a_input_A2 = torch.hstack((h, h_A2)).unsqueeze(1)
        a_input_A3 = torch.hstack((h, h_A3)).unsqueeze(1)
        a_input_sct1 = torch.hstack((h, h_sct1)).unsqueeze(1)
        a_input_sct2 = torch.hstack((h, h_sct2)).unsqueeze(1)
        a_input_sct3 = torch.hstack((h, h_sct3)).unsqueeze(1)

        a_input =  torch.cat((a_input_A,a_input_A2,a_input_A3,a_input_sct1,a_input_sct2,a_input_sct3),1).view(N,6,-1)
        #GATV2
        e = torch.matmul(torch.nn.functional.relu(a_input),self.a).squeeze(2)
        attention = F.softmax(e, dim=1).view(N, 6, -1)

        h_all = torch.cat((h_A.unsqueeze(dim=1), h_A2.unsqueeze(dim=1),h_A3.unsqueeze(dim=1), h_sct1.unsqueeze(dim=1), h_sct2.unsqueeze(dim=1), h_sct3.unsqueeze(dim=1)),dim=1)
        h_prime = torch.mul(attention, h_all) # element wise product
        h_prime = torch.mean(h_prime,1)

        X = self.linear1(h_prime)
        X = F.leaky_relu(X)
        X = self.linear2(X)
        X = F.leaky_relu(X)
        return X







class scattering_GNN(nn.Module):
    def __init__(self, base_graph, input_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        self.graph = base_graph
        self.adj = self.graph.W
        self.in_proj = torch.nn.Linear(input_dim, hidden_dim)
        self.convs = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(SCTConv(self.graph, hidden_dim=hidden_dim))
        self.mlp1 = torch.nn.Linear(hidden_dim*(1+n_layers), hidden_dim) ##check
        self.mlp2 = torch.nn.Linear(hidden_dim,output_dim)

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'init model; total number of parameters: {total_params}')

    def forward(self, X):## 
        numnodes = self.graph.N
        scale = np.sqrt(numnodes)
        X = X/scale
        X = self.in_proj(X)
        hidden_states = X

        for layer in self.convs:
               X = layer(X) ## 
               X = X/scale # normalize
               hidden_states = torch.cat([hidden_states, X], dim=1)

        X = hidden_states 
        X = self.mlp1(X)
        X = F.leaky_relu(X)
        X = self.mlp2(X) 
        maxval = torch.max(X)
        minval = torch.min(X)
        X = (X-minval)/(maxval+1e-6-minval)
        return X



#############################################
############################################# #decoder

import time

def test(model, graph, loader,  walkerstart=0,  Numofwalkers = 10,thresholdloopnodes = None):
    N = graph.N
    if thresholdloopnodes is None:
        thresholdloopnodes = N

    index = 0
    clilist = []
    timelist = []
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            for j in range(len(batch)): 
                t_0 = time.time()
                features = torch.FloatTensor(batch[j].x)
                output_dis = model(features)

                predC = []
                for walkerS in range(0,min(Numofwalkers,N)):
                    p_c = decoder(graph, output_dis, 
                                  walkerstart=walkerstart,  Numofwalkers = Numofwalkers,thresholdloopnodes=thresholdloopnodes)
                    predC += [p_c]
                
                cliques = max(predC)
                clilist += [cliques]
                t_pred = time.time() - t_0 
                timelist += [t_pred]
    return clilist,timelist




def decoder(graph, output_dis, walkerstart=0, Numofwalkers = 10, thresholdloopnodes=None):
    N = graph.N
    if thresholdloopnodes is None:
        thresholdloopnodes = N
    edge_index = graph.edge_index
    adjmatrix = to_sparse_mx(edge_index, N)

    _sorted, indices = torch.sort(output_dis.squeeze(),descending=True)
    initiaprd = 0.*indices  # torch zeros
    initiaprd = initiaprd.numpy() 
    for walker in range(min(Numofwalkers,N)):
        if walker < walkerstart:
            initiaprd[indices[walker]] = 0.
        else:
            pass
        
        initiaprd[indices[walkerstart]] = 1. 
        for clq in range(walkerstart+1,min(thresholdloopnodes,N)):
            initiaprd[indices[clq]] = 1.
            binary_vec = np.reshape(initiaprd, (-1,1))
            ZorO = np.sum(binary_vec)**2  - np.sum(binary_vec) - np.sum(binary_vec*(adjmatrix.dot(binary_vec)))
            if ZorO < 0.0001: # same as ZorO == 0
                pass
            else:
                initiaprd[indices[clq]] = 0.
        return np.sum(initiaprd)
        














#############################################
############################################# #archive/QC code

'''
train_test = [b for i, b in enumerate(train_loader)]
features = torch.FloatTensor(train_test[0][0].x)

#test_loss
output = model(features)
w_exp = torch.tensor([feat[2].item() for feat in features]) 
retdict = exploss(model.graph, output, penalty_coefficient, w_exp = w_exp)
retdict
#test_loss
train(model, 0, train_loader, optimizer, penalty_coefficient, verbose=False)
#test_loss
output = model(features)
w_exp = torch.tensor([feat[2].item() for feat in features]) 
retdict = exploss(model.graph, output, penalty_coefficient, w_exp = w_exp)
retdict
'''



def getclicnum(adjmatrix,dis,walkerstart = 0,thresholdloopnodes = 50):
    '''
    ComplementedgeM: complement matrix of adj 
    dis: distribution on the nodes, higher ->better
    cpu: cpu is usually better for small model
    '''
    _sorted, indices = torch.sort(dis.squeeze(),descending=True)#flatten, elements are sorted in descending order by value.
    initiaprd = 0.*indices  # torch zeros
    initiaprd = initiaprd.cpu().numpy() 
    for walker in range(min(thresholdloopnodes,adjmatrix.get_shape()[0])):
        if walker < walkerstart:
            initiaprd[indices[walker]] = 0.
        else:
            pass
    initiaprd[indices[walkerstart]] = 1. # the one with walkerstart'th largest prob is in the clique, start with walkerstart
    for clq in range(walkerstart+1,min(thresholdloopnodes,adjmatrix.get_shape()[0])): # loop the 50 high prob nodes
        initiaprd[indices[clq]] = 1.
        binary_vec = np.reshape(initiaprd, (-1,1)) 
        ZorO = np.sum(binary_vec)**2  - np.sum(binary_vec) - np.sum(binary_vec*(adjmatrix.dot(binary_vec)))
        if ZorO < 0.0001: # same as ZorO == 0
            pass
        else:
            initiaprd[indices[clq]] = 0.
    return np.sum(initiaprd)
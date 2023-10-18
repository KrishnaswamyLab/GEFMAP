from utils import *
import networkx as nx

import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import Linear
from torch import tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import sparse
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp
import time
torch.manual_seed(1)
np.random.seed(2)

############# utils #######################################

def G2edgeindex(G):
    Glist = []
    for i, edge in enumerate(list(G.edges)):
        #print(i, edge)
        Glist.append([edge[0],edge[1]])
    return Glist

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

############# prepross #######################################



def get_psd_features(g, verbose = True):
    if verbose == True:
        print(f'is connected: {nx.is_connected(g)}......connected subgraphs: {len([g.subgraph(c).copy() for c in nx.connected_components(g)])}')
    n_nodes = len(g)
    feature_vector = []
    contg = contg = nx.is_connected(g)

    if contg:
        for j in g.nodes:
            try:
                _eccen = nx.eccentricity(g,j)
                _deg = nx.degree(g,j)
                #_deg = np.log(_deg)
                _deg = np.sqrt(_deg)
                _cluter = nx.clustering(g,j)
                feature_vector.append([_eccen,_deg,_cluter])
            except:
                print("{j}th is Wrong")
                feature_vector.append([0,0,0])

    elif not contg:
        graphs = [g.subgraph(c).copy() for c in nx.connected_components(g)]
        node_vector = [] #for ordering features
        for gsub in graphs:
            node_vector.extend(list(gsub.nodes))
            for j in gsub.nodes:
                try:
                    _eccen = nx.eccentricity(gsub,j)
                    _deg = nx.degree(gsub,j)
                    #_deg = np.log(_deg)
                    _deg = np.sqrt(_deg)
                    _cluter = nx.clustering(gsub,j)
                    feature_vector.append([_eccen,_deg,_cluter])
                except:
                    print("{j}th is Wrong")
                    feature_vector.append([0,0,0])

        #reorder features
        ordered_nodes = list(np.arange(0,len(g.nodes)))
        indices_to_reorder = np.argsort([ordered_nodes.index(item) for item in node_vector])
        feature_vector = [feature_vector[idx] for idx in indices_to_reorder]

    return np.array(feature_vector)




def normalize_adjacency_matrix(A, I):
    """
    Creating a normalized adjacency matrix with self loops.
    :param A: Sparse adjacency matrix.
    :param I: Identity matrix.
    :return A_tile_hat: Normalized adjacency matrix.
    """
    A_tilde = A + I
    degrees = A_tilde.sum(axis=0)[0].tolist()
    D = sp.diags(degrees, [0])
    D = D.power(-0.5)
    A_tilde_hat = D.dot(A_tilde).dot(D)
    return A_tilde_hat


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalizemx(mx):
    degrees = mx.sum(axis=0)[0].tolist()
#    print(degrees)
    D = sp.diags(degrees, [0])
    D = D.power(-1)
    mx = mx.dot(D)
    return mx


############### layers ############################################################


def exploss(edge_index,output_dis,penalty_coefficient=0.005,device = 'cuda'):
    adjmatrix = to_scipy_sparse_matrix(edge_index)
    adjmatrix = sparse_mx_to_torch_sparse_tensor(adjmatrix).to(device) 
    I_n = sp.eye(adjmatrix.size(0))
    I_n = sparse_mx_to_torch_sparse_tensor(I_n).cuda()
    Fullm = torch.ones(I_n.size(0),I_n.size(1)).cuda() - I_n #(N,N) 
    diffusionprob = torch.mm(Fullm - adjmatrix,output_dis)
    elewiseloss = output_dis * diffusionprob
    lossComplE = penalty_coefficient * torch.sum(elewiseloss) # loss on compl of Edges
    lossE = torch.sum(output_dis*torch.mm(adjmatrix,output_dis))
    loss = -lossE + lossComplE
    retdict = {}
    retdict["loss"] = [loss,lossComplE] #final loss
    return retdict


def GCN_diffusion(sptensor, feature, order = 3):
    """
    Creating a normalized adjacency matrix with self loops.
    sptensor = W
    https://arxiv.org/pdf/1609.02907.pdf
    """
    I_n = sp.eye(sptensor.size(0))
    I_n = sparse_mx_to_torch_sparse_tensor(I_n)
    A_gcn = sptensor +  I_n
    degrees = torch.sparse.sum(A_gcn,0)
    D = degrees.to_dense()
    D = torch.pow(D, -0.5)
    D = D.unsqueeze(dim=1)
    gcn_diffusion_list = []
    A_gcn_feature = feature #x
    for i in range(order):
        A_gcn_feature = torch.mul(A_gcn_feature,D) #
        A_gcn_feature = torch.spmm(A_gcn,A_gcn_feature)
        A_gcn_feature = torch.mul(A_gcn_feature,D)
        gcn_diffusion_list += [A_gcn_feature,]
    return gcn_diffusion_list



def scattering_diffusion(sptensor, feature, order =3):
    '''
    sptensor = W
    '''
    degrees = torch.sparse.sum(sptensor,0)
    D = degrees
    D= D.to_dense()
    D = torch.pow(D, -1)
    D = D.unsqueeze(dim=1)
    
    iteration = 2**(order-1)
    feature_p = feature
    sct_diffusion_list = []

    for i in range(iteration):
        D_inv_x = D*feature_p
        W_D_inv_x = torch.spmm(sptensor,D_inv_x)
        feature_p = 0.5*feature_p + 0.5*W_D_inv_x
        sct_diffusion_list += [feature_p,]
    
    sct_feature1 = sct_diffusion_list[0]-sct_diffusion_list[1]
    sct_feature2 = sct_diffusion_list[1]-sct_diffusion_list[2]
    sct_feature3 = sct_diffusion_list[2]-sct_diffusion_list[3]
    return sct_feature1,sct_feature2,sct_feature3


############### models ############################################################

class SCTConv(Module):
    def __init__(self, hidden_dim,):
        super().__init__()
        self.hid = hidden_dim
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.a = Parameter(torch.zeros(size=(2*hidden_dim, 1)))

    def forward(self,X,adj,moment = 1):
        """
        Params
        ------
        adj [batch x nodes x nodes]: adjacency matrix
        X [batch x nodes x features]: node features matrix
        Returns
        -------
        X' [batch x nodes x features]: updated node features matrix
        """
        support0 = X 
        N = support0.size()[0]
        h = support0
        #low pass filters
        gcn_diffusion_list = GCN_diffusion(adj,support0)
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
    



class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        #embedding module (input projection)
        self.in_proj = torch.nn.Linear(input_dim, hidden_dim)
        #diffusion module
        self.convs = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(SCTConv(hidden_dim))
        self.mlp1 = Linear(hidden_dim*(1+n_layers), hidden_dim)
        self.mlp2 = Linear(hidden_dim,output_dim)

    def forward(self, X, adj, moment=1):
        numnodes = X.size(0)
        scale = np.sqrt(numnodes) # for graph norm
        X = X/scale
        # embedding module
        X = self.in_proj(X)
        hidden_states = X
        for layer in self.convs:
            X = layer(X,adj,moment = 1)
            # normalize
            X = X/scale
            hidden_states = torch.cat([hidden_states, X], dim=1)
        X = hidden_states 
        X = self.mlp1(X)
        X = F.leaky_relu(X)
        X = self.mlp2(X) 
        maxval = torch.max(X)
        minval = torch.min(X)
        X = (X-minval)/(maxval+1e-6-minval)
        
        return X
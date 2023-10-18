from utils import *
import networkx as nx
import torch
import torch.nn as nn
#from torch.nn import Module
from torch.nn.modules.module import Module
from torch.nn import Linear
from torch import tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import sparse
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp
from sklearn.preprocessing import normalize

import time
#torch.manual_seed(1)
#np.random.seed(2)

############################################################ utils
def G2edgeindex(G):
    Glist = []
    for i, edge in enumerate(list(G.edges)):
        Glist.append([edge[0],edge[1]])
    return Glist

def to_sparse_mx(edge_index, N):
    row, col = edge_index
    #NO EDGE ATTR
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



def min_max(X):
    maxval = torch.max(X)
    minval = torch.min(X)
    X = (X-minval)/(maxval+1e-6-minval) #1e-6 to account for zero vals
    return X


def exploss(graph, w_exp, output_dis, penalty_coefficient):
    W = graph.W
    W_complement = graph.W_complement

    diffusionprob = torch.mm(W_complement - W,output_dis)
    lossE = torch.sum((w_exp*output_dis)*torch.mm(W,output_dis))
    elewiseloss = output_dis * diffusionprob
    lossComplE = penalty_coefficient * torch.sum(elewiseloss)
    ##
    loss = -lossE + lossComplE
    retdict = {}
    retdict["loss"] = [loss,lossComplE] #final loss
    return retdict


############################################################# graph construction /base metabolic graph

class base_graph():
    # base graph G unweighted, undirected
    def __init__(self, G):
        self.G = G
        self.N = len(G.nodes)
        self.edge_list = G2edgeindex(G)
        self.edge_index = torch.transpose(torch.tensor(self.edge_list,dtype=torch.long),0,1)
        adjmatrix = to_sparse_mx(self.edge_index, self.N)
        self.W = sparse_mx_to_torch_sparse_tensor(adjmatrix)

        I_n =sp.eye(self.W.size(0))
        self.I_n = sparse_mx_to_torch_sparse_tensor(I_n)
        self.Fullm = torch.ones(self.I_n.size(0),self.I_n.size(1)) - self.I_n 
        self.W_complement = self.Fullm - self.W
        self.A_gcn = self.W + self.I_n


    def get_universal_features(self):
        graphs = {}
        contg = nx.is_connected(self.G)
        _eccen = []
        _cluter = []

        if contg:
            for j in self.G.nodes:
                try:
                    _eccen += [nx.eccentricity(self.G,j)]
                    _cluter += [nx.clustering(self.G,j)]
                except:
                    print("{j}th is Wrong")
                    _eccen +=[0]
                    _cluter += [0]
            self._eccen = _eccen
            self._cluter = _cluter
        
        elif not contg:
            G_ = [self.G.subgraph(c).copy() for c in nx.connected_components(self.G)]
            node_vector = []
            for gsub in G_:
                node_vector.extend(list(gsub.nodes))
                for j in gsub.nodes:
                    try:
                        _eccen += [nx.eccentricity(gsub,j)]
                        _cluter += [nx.clustering(gsub,j)]
                    except:
                        print("{j}th is Wrong")
                        _eccen +=[0]
                        _cluter += [0]

            ordered_nodes = list(np.arange(0,len(self.G.nodes)))
            indices_to_reorder = np.argsort([ordered_nodes.index(item) for item in node_vector])
            self._eccen = _eccen
            self._cluter = _cluter
            self.indices_to_reorder = indices_to_reorder

    def get_input_features(self, srm):
        self.srm = srm
        self.total_samples = srm.shape[0]
        f = []
        self.get_universal_features()
        self.node_rxn_labels = {n:srm.columns[i] for i, n in enumerate(self.G.nodes)}
        contg = nx.is_connected(self.G)
        if contg:
            for s, samp in enumerate(srm.index):
                feature_vector = []
                g = self.G.copy()
                for j in g.nodes:
                    try:
                        _exp = srm.iloc[s, j]
                        neighbors = [e[1] for e in self.edge_list if e[0] == j]
                        neighbor_ids = [self.node_rxn_labels[n] for n in neighbors]
                        _deg = np.sum(srm[neighbor_ids].iloc[s,:])
                        _deg = np.sqrt(_deg)
                    except:
                        _exp = 0
                        _deg = 0
                    feature_vector.append([self._eccen[j],self._cluter[j], _exp, _deg])
                f += [feature_vector]
        elif not contg:
            G_ = [self.G.subgraph(c).copy() for c in nx.connected_components(self.G)]
            for s, samp in enumerate(srm.index):
                g = G_.copy()
                feature_vector = [] 
                for gsub in G_:
                    edge_list = G2edgeindex(gsub)
                    for j in gsub.nodes:
                        try:
                            _exp = srm.iloc[s, j]
                            neighbors = [e[1] for e in edge_list if e[0] == j]
                            neighbor_ids = [self.node_rxn_labels[n] for n in neighbors]
                            _deg = np.sum(srm[neighbor_ids].iloc[s,:])
                            _deg = np.sqrt(_deg)
                        except:
                            _exp = 0
                            _deg = 0
                        feature_vector.append([self._eccen[j],self._cluter[j], _exp, _deg])
                feature_vector = [feature_vector[i] for i in self.indices_to_reorder]
                f += [feature_vector]
        self.f = f
        return self.f






############################################################ layers


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


############################################################ models


class SCTConv(torch.nn.Module):
    def __init__(self, base_graph, hidden_dim, smooth,Withgres=False):
        super().__init__()
        self.graph = base_graph
        self.hid = hidden_dim
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.a = Parameter(torch.zeros(size=(2*hidden_dim, 1)))
        self.smoothlayer = Withgres #turn on graph residual layer or not
        self.gres = GC_withres(self.graph, hidden_dim,hidden_dim,smooth = smooth)


    def forward(self,X,moment = 1):
        '''
        input: X [batch x nodes x features]
        returns: X' [batch x nodes x features]
        '''
        support0 = X
        N = self.graph.N
        adj = self.graph.W
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
        if self.smoothlayer:
            h_prime = self.gres(h_prime)
        else:
            pass

        X = self.linear1(h_prime)
        X = F.leaky_relu(X)
        X = self.linear2(X)
        X = F.leaky_relu(X)
        return X





class GC_withres(Module):
    """
    res conv
    """
    def __init__(self, base_graph, in_features, out_features,smooth):
        super(GC_withres, self).__init__()
        self.base_graph = base_graph
        self.A_gcn = base_graph.A_gcn
        self.in_features = in_features
        self.out_features = out_features
        self.smooth = smooth
        self.mlp = nn.Linear(in_features, out_features)
    def forward(self, input):
        support = self.mlp(input)
        A_gcn = self.A_gcn
        degrees = torch.sparse.sum(A_gcn,0)
        D = degrees
        D = D.to_dense() # transfer D from sparse tensor to normal torch tensor
        D = torch.pow(D, -0.5)
        D = D.unsqueeze(dim=1)

        A_gcn_feature = support
        A_gcn_feature = torch.mul(A_gcn_feature,D)
        A_gcn_feature = torch.spmm(A_gcn,A_gcn_feature)
        A_gcn_feature = torch.mul(A_gcn_feature,D)
        output = A_gcn_feature * self.smooth + support
        output = output/(1+self.smooth)
        return output






class scattering_GNN(nn.Module):
    def __init__(self, base_graph, input_dim, hidden_dim, output_dim, n_layers,Withgres=False, smooth=0):
        super().__init__()
        self.graph = base_graph
        self.adj = self.graph.W
        self.smooth = smooth
        self.in_proj = torch.nn.Linear(input_dim, hidden_dim)
        self.convs = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(SCTConv(self.graph, hidden_dim,self.smooth,Withgres))
        self.mlp1 = torch.nn.Linear(hidden_dim*(1+n_layers), hidden_dim)
        self.mlp2 = torch.nn.Linear(hidden_dim,output_dim)

    def forward(self, X):## calls for adj - TODOS
        numnodes = self.graph.N
        scale = np.sqrt(numnodes)
        X = X/scale
        X = self.in_proj(X)
        hidden_states = X

        for layer in self.convs:
               X = layer(X) ## calls for adj - TODOS
               X = X/scale # normalize
               hidden_states = torch.cat([hidden_states, X], dim=1)

        X = hidden_states 
        X = self.mlp1(X)
        X = F.leaky_relu(X)
        X = self.mlp2(X) 

        #X = min_max(X)
        X = X/ torch.sum(X)
        #from sklearn.preprocessing import normalize
        #X_n = normalize(X.detach().numpy(), norm='l1', axis =0)
        #X = torch.tensor(X_n)

        X = X/ torch.sum(X)

        return X





############################################################ decoder


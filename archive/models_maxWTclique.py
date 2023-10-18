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

############# input/prepross #######################################




''' #for connected
def get_input_features(G, srm, node_rxn_labels):
    graphs = {}
    except_nodes= 0
    edge_list = G2edgeindex(G)

    #same for all graphs
    _eccen = {node:nx.eccentricity(G,node) for node in G.nodes}
    _cluter = {node:nx.clustering(G,node) for node in G.nodes}

    
    for s, samp in enumerate(srm.index):
        g = G.copy()
        _exp = {node:srm.iloc[s, i] for i, node in enumerate(g.nodes)}
        deg_vec = []
        for node in g.nodes:
            neighbors = [e[1] for e in edge_list if e[0] == node]
            neighbor_ids = [node_rxn_labels[n] for n in neighbors]
            wd = np.sum(srm[neighbor_ids].iloc[s,:])
            deg_vec += [wd]
        _deg = {node:deg_vec[i] for i, node in enumerate(g.nodes)}

        nx.set_node_attributes(g, _eccen, "_eccen")
        nx.set_node_attributes(g, _cluter, "_cluter")
        nx.set_node_attributes(g, _exp, "_exp")
        nx.set_node_attributes(g, _deg, "_deg")

        graphs[s] = g
    return graphs

'''


def get_universal_features(G):
    graphs = {}
    edge_list = G2edgeindex(G)
    n_nodes = len(G)
    contg = nx.is_connected(G)

    ## all graphs to make it faster
    _eccen = []
    _cluter = []

    if contg:
        for j in G.nodes:
            try:
                 _eccen += [nx.eccentricity(G,j)]
                 _cluter += [nx.clustering(G,j)]
            except:
                print("{j}th is Wrong")
                _eccen +=[0]
                _cluter += [0]

        return _eccen, _cluter

    elif not contg:
        G_ = [G.subgraph(c).copy() for c in nx.connected_components(G)]
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

        ordered_nodes = list(np.arange(0,len(G.nodes)))
        indices_to_reorder = np.argsort([ordered_nodes.index(item) for item in node_vector])

    
        return _eccen, _cluter, indices_to_reorder




#####

def get_input_features(G, srm, node_rxn_labels):
    f = []
    contg = nx.is_connected(G)
    
    if contg:
         indices_to_reorder = None
         _eccen, _cluter = get_universal_features(G)
         edge_list = G2edgeindex(G)
         for s, samp in enumerate(srm.index):
             g = G.copy()
             feature_vector = [] ##[gene_expression, eccentricity, clust_coeff, weighted_degree]
             for j in G.nodes:
                try:
                     _exp = srm.iloc[s, j]
                     neighbors = [e[1] for e in edge_list if e[0] == j]
                     neighbor_ids = [node_rxn_labels[n] for n in neighbors]
                     _deg = np.sum(srm[neighbor_ids].iloc[s,:])
                     _deg = np.sqrt(_deg)
                except:
                    _exp = 0
                    _deg = 0
                feature_vector.append([_eccen[j],_cluter[j], _exp, _deg])
             f += [feature_vector]
            
                #print(feature_vector)

    elif not contg:
        _eccen, _cluter, indices_to_reorder = get_universal_features(G)
        G_ = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        for s, samp in enumerate(srm.index):
            g = G.copy()
            feature_vector = [] 
            for gsub in G_:
                edge_list = G2edgeindex(gsub)
                for j in gsub.nodes:
                    try:
                        _exp = srm.iloc[s, j]
                        neighbors = [e[1] for e in edge_list if e[0] == j]
                        neighbor_ids = [node_rxn_labels[n] for n in neighbors]
                        _deg = np.sum(srm[neighbor_ids].iloc[s,:])
                        _deg = np.sqrt(_deg)
                    except:
                        _exp = 0
                        _deg = 0
                    feature_vector.append([_eccen[j],_cluter[j], _exp, _deg])
            feature_vector = [feature_vector[i] for i in indices_to_reorder]
            f += [feature_vector]
    return f

    





             





############# models #######################################
##### diffusion layers #######



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


class GC_withres(Module):
    """
    res conv
    """
    def __init__(self, in_features, out_features,smooth):
        super(GC_withres, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.smooth = smooth
        self.mlp = nn.Linear(in_features, out_features)
    def forward(self, input, adj):
        support = self.mlp(input)
        I_n = sp.eye(adj.size(0))
        I_n = sparse_mx_to_torch_sparse_tensor(I_n)
        A_gcn = adj +  I_n
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

##### model_utils ######


class SCTConv(Module):
    def __init__(self, hidden_dim, smooth, dropout,Withgres=False):
        super().__init__()
        self.hid = hidden_dim
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.a = Parameter(torch.zeros(size=(2*hidden_dim, 1)))
        self.smoothlayer = Withgres #turn on graph residual layer or not
        self.gres = GC_withres(hidden_dim,hidden_dim,smooth = smooth)
        self.dropout = dropout
        

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
        #GATV2
        e = torch.matmul(torch.nn.functional.relu(a_input),self.a).squeeze(2)
        attention = F.softmax(e, dim=1).view(N, 6, -1)

        h_all = torch.cat((h_A.unsqueeze(dim=1), h_A2.unsqueeze(dim=1),h_A3.unsqueeze(dim=1), h_sct1.unsqueeze(dim=1), h_sct2.unsqueeze(dim=1), h_sct3.unsqueeze(dim=1)),dim=1)
        h_prime = torch.mul(attention, h_all) # element wise product
        h_prime = torch.mean(h_prime,1)
        if self.smoothlayer:
            h_prime = self.gres(h_prime,adj)
        else:
            pass

        X = self.linear1(h_prime)
        X = F.leaky_relu(X)
        X = self.linear2(X)
        X = F.leaky_relu(X)
        return X
    
    ######## model ##################



class mwc_GNN(nn.Module):
     def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout, Withgres= False, smooth=0):
          super().__init__()
          self.dropout = dropout
          self.smooth = smooth
          self.in_proj = torch.nn.Linear(input_dim, hidden_dim)
          self.convs = torch.nn.ModuleList()
          for _ in range(n_layers):
               self.convs.append(SCTConv(hidden_dim,self.smooth,self.dropout,Withgres))
          self.mlp1 = torch.nn.Linear(hidden_dim*(1+n_layers), hidden_dim)
          self.mlp2 = torch.nn.Linear(hidden_dim,output_dim)

     def forward(self, X, adj, moment=1):
          numnodes = X.size(0)
          scale = np.sqrt(numnodes)
          X = X/scale
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
          X = (X-minval)/(maxval+1e-6-minval) #1e-6 to account for zero vals
          
          return X
     


########## training utils ############


'''
def exploss(edge_index,output_dis,penalty_coefficient=0.005):
    N = len(np.unique(edge_index)) #n_nodes
    adjmatrix = to_sparse_mx(edge_index, N)
    adjmatrix = sparse_mx_to_torch_sparse_tensor(adjmatrix) #W = adj
    I_n =sp.eye(adjmatrix.size(0))
    I_n = sparse_mx_to_torch_sparse_tensor(I_n)
    Fullm = torch.ones(I_n.size(0),I_n.size(1)) - I_n #(N,N) 

    
    diffusionprob = torch.mm(Fullm - adjmatrix,output_dis)
    elewiseloss = output_dis * diffusionprob
    lossComplE = penalty_coefficient * torch.sum(elewiseloss) # loss on compl of Edges
    lossE = torch.sum(output_dis*torch.mm(adjmatrix,output_dis))
    loss = -lossE + lossComplE
    retdict = {}
    retdict["loss"] = [loss,lossComplE] #final loss
    return retdict
'''



################# outs ##########################

def getclicnum(adjmatrix,dis,walkerstart = 0,thresholdloopnodes = 50):
    '''
    ComplementedgeM: complement matrix of adj 
    dis: distribution on the nodes, higher ->better
    cpu: cpu is usually better for small model
    '''
    _sorted, indices = torch.sort(dis.squeeze(),descending=True)#flatten, elements are sorted in descending order by value.
    initiaprd = 0.*indices  # torch zeros
    initiaprd = initiaprd.numpy() 
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
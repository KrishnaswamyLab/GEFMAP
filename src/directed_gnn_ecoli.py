# -*- coding: utf-8 -*-
"""modified undirected_gnn_ecoli.py to have a directed version

Yixuan's note: need to install torch_geometric_signed_directed, please follow 

https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/notes/installation.html

"""

#GNN to predict directed metabolic graphs
#input: RAG reaction adjacency graph, mapped reaction expression (node features)
#output: edge weights

#include stoichiometric penalties (S*v==0)
#objective function c is known (maximize for biomass accumulation)

import os

import torch

os.environ['TORCH'] = torch.__version__
print(torch.__version__)
import pickle
import pickle as pk
import numpy as np
import pandas as pd
from torch.nn import Linear
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
# from torch_geometric.nn import TopKPooling
# from torch_geometric.nn import global_mean_pool
from torch_geometric_signed_directed.nn import MagNetConv
"""# <u>e. coli core metabolic network</u>
- 72 metabolites
- 95 reactions
- 137 genes

> ## <u>Data Preprocessing
"""



out_path = './output/METABOLIC_DATA/ecoli_network/'

with open(out_path + 'linear_solutions/ecoli_res.pickle', 'rb') as b:
    solutions = pk.load(b)

with open(out_path + 'linear_solutions/ecoli_bounds.pickle', 'rb') as b:
    bounds = pk.load(b)

S_matrix = pd.read_csv(out_path + 'datasets/S_matrix.csv', index_col=0)

with open(out_path + '/linear_solutions/ecoli_graphs.pickle', 'rb') as b:
    graphs = pk.load(b)



S_bool = S_matrix[S_matrix == 0].fillna(1).to_numpy()
"""$RAG = \hat{S}^T \hat{S}$

 where $\hat{S}$ is boolean S_matrix
"""

RAG = np.matmul(np.transpose(S_bool), S_bool)
RAG = pd.DataFrame(RAG, index=S_matrix.columns, columns=S_matrix.columns)

edge_index = np.zeros([1, 2])
for i, row in enumerate(RAG.index):
    for f, col in enumerate(RAG.columns):
        if RAG.iloc[i, f] > 0:
            p = ([i, f])
            edge_index = np.vstack([edge_index, p])
edge_index = edge_index[2:, :]

edge_index = edge_index.reshape(2, 2204)
edge_index = torch.from_numpy(edge_index)
edge_index = edge_index.type(torch.LongTensor)


dictionary = {}
for i, curr_bounds in enumerate(bounds):

    #x = torch.from_numpy(curr_bounds.reshape(95,2)).float() #both upper and lower bounds
    x = torch.from_numpy(curr_bounds[:,
                                     0].reshape(95,
                                                1)).float()  #just upper bound
    y = torch.from_numpy(solutions[i].reshape(95, 1)).float()

    data = Data(x=x, y=y, edge_index=edge_index)
    key = "graph_" + str(i)
    dictionary[key] = data
"""# <u>GNN Implementation"""

num_of_nodes = 95
num_of_edge_types = 1


class GNN_MagNet(torch.nn.Module):

    def poly_regression():
        #Write the equation here
        pass

    #Too large output_features: Make it 2 or 4.
    def __init__(self, input_features=1, half_hidden_features=8, K=2, q=0.25):
        super(GNN_MagNet, self).__init__()
        torch.manual_seed(12345)
        #Define the layers and activation functions (and pooling if we need to downsample) for the GNN here
        #We can use TopKPooling since it seems to be most efficient theoritically
        #Can we use GCNConv here? or do we need to create our own message passing network?

        # self.conv1 = GCNConv(input_features, output_features) #8 node features
        # self.conv1 = GINConv(Sequential(Linear(input_features, output_features),
        #                BatchNorm1d(output_features), ReLU(),
        #                Linear(output_features, output_features), ReLU()))
        self.conv1 = MagNetConv(
            in_channels=input_features,
            out_channels=half_hidden_features*2,
            K=K, q=q)
        # self.pool1 = TopKPooling()
        # self.conv2 = GCNConv(output_features,output_features)
        # self.conv2 = GINConv(Sequential(Linear(output_features, output_features),
        #                BatchNorm1d(output_features), ReLU(),
        #                Linear(output_features, output_features), ReLU()))
        self.conv2 = MagNetConv(
            in_channels=half_hidden_features*2,
            out_channels=half_hidden_features*2,
            K=K, q=q)
        # self.pool2 = TopKPooling()

        # self.conv3 = GINConv(Sequential(Linear(output_features, output_features),
        #                BatchNorm1d(output_features), ReLU(),
        #                Linear(output_features, output_features), ReLU()))
        self.conv3 = MagNetConv(
            in_channels=half_hidden_features*2,
            out_channels=half_hidden_features,
            K=K, q=q)
        # self.conv3 = GCNConv(output_features,2)
        #Add MLP layers after the GCNConv layers for expressive power of GNNs
        # self.MLPLayer1 = ...
        # self.MLPLayer2 = ...

        #Node dimensionality would be the reduction in the dimension from the number of node features to whatever dimension
        #Example : if we have 8 node features it could be reduced to 4 then 2
        self.rates = Linear(16, 1)

    def forward(self, data_x, data_edge_index):
        #x = mapped reaction expression - node features
        #Perform forward pass here:

        #1. Downsample if it is a large graph (can use TopKPooling)
        # x = self.pool1(x,edge_index)

        #2. Message passing using the GCNConv - Neighborhood aggregation.
        #Obtains node embeddings by aggregating information of neighbor nodes.
        #The number of layers corresponds to the number of hops for node aggregation information.
        #In this case, 2-hop.
        x_real, x_img = self.conv1(data_x, data_x, data_edge_index)
        #Maybe try dropout()
        x_real = x_real.relu()
        x_real, x_img = self.conv2(x_real, x_img, data_edge_index)
        #Remove this relu()
        x_real = x_real.relu()
        x_real, x_img = self.conv3(x_real, x_img, data_edge_index)
        x = torch.cat([x_real, x_img], dim=-1)

        #3. Apply classifier/predictor to classify whether the node is fluxing or not. (2 classes)
        #Can we make this a regression problem for real valued quantities (amount of metabolites)
        # weight = torch.nn.Parameter(torch.FloatTensor(35, 1))
        # score = F.linear(x,weight)
        # score = F.softmax(x, dim=1)
        # score = Linear(35,35)
        output_data = self.rates(x)
        return output_data

        # return x


#loss function
#enforce S*V = 0
#distance metric of edge weights (true vs. predicted) which captures gene expression information

# def custom_loss(actual, predicted,S):
#     #MSE - Mean Squared Err
#     rmse = mean_squared_error(actual, predicted, squared=False)

torch.set_printoptions(profile="full")

# print(dictionary)
#Would this training data be of just eh subgraphs we generate?

# def train(train_data):
#     count = 0
#     #for epoch in range(epochs):
#         #Iterate over the training data
#     for key,data in train_data.dataset.items():
#         optimizer.zero_grad()
#         #output would be the classes associated with each node
#         # print(data)

#         output = model(data.x,data.edge_index)
#         # output = output.double()
#         # print("Output data")
#         # print(output)
#         # print(data.y)
#         # print(data.edge_weights.type())
#         #Compute the loss: penalize for output of the model (fluxing or not) versus ground truth
#         loss = mse_loss(output,data.y) #+ 0.1 * np.sum(output,axis=0)
#         #print("Loss:")
#         #print(loss)
#         #Derive the gradients
#         loss.backward()
#         #Update parameters
#         optimizer.step()

#         # print(torch)
#         #Clear gradient

#     print(output.shape, data.y.shape)
#     count = 0
#     for i in range(output.shape[0]):

#         if torch.eq(output[i],data.y[i]):
#             count += 1

#     print("Loss:")
#     print(loss)
#     print("Accuracy:")
#     print(count, "out of 95")
# print(count)
#return output, data.y

#train function
#loop through the epochs (for hyperparameter tuning)
#Iterate over dataset and compute the loss
model = GNN_MagNet(input_features=1, half_hidden_features=8, K=2, q=0.25)
#Change the optimizer to SGD.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#need to use nnl loss due to softmax being the final layer of the gnn
mse_loss = torch.nn.L1Loss()
#Load the data to be trained
training_data = DataLoader(dictionary)


def train(dictionary):
    for key, data in dictionary.items():
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x, data.edge_index)  # Perform a single forward pass.
        loss = mse_loss(
            out,
            data.y)  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
    count = 0
    lst = []
    for i in range(out.shape[0]):

        diff = out[i] - data.y[i]
        lst.append(diff)
        # if torch.eq(out[i],data.y[i]):
        #     count += 1
    # print("output - data.y for each data point")
    # print(lst)
    print("Loss:")
    print(loss)
    print("Accuracy:")
    print(count, "out of 95")


print(training_data.dataset)

train(dictionary)

for epoch in range(0, 30):
    train(dictionary)

#test function
testing_data = DataLoader(data, batch_size=batch)


def test():
    #for epoch in range(epochs):
    #Iterate over the training data
    for data in training_data:
        #Perform forward pass
        edge_weight = model(data.x, data.edge_index)
        #Get predictions
        predictions = edge_weight.argmax(dim=1)
        #correct count
        correct += int((predictions == data.edge_index).sum())
    return correct


"""#Data processing"""



data_folder = '../data/toy_data/'

with open(data_folder + 'toy_graphs.pickle', 'rb') as handle:
    graphs = pickle.load(handle)

with open(data_folder + 'node_features.pickle', 'rb') as handle:
    features = pickle.load(handle)

#edge index RAG
RAG = pd.read_csv(data_folder + "toy_adjacency_matrix.csv", index_col=0)

edge_index = np.zeros([1, 2])
for i, row in enumerate(RAG.index):
    for f, col in enumerate(RAG.columns):
        if RAG.iloc[i, f] > 0:
            p = ([i, f])
            edge_index = np.vstack([edge_index, p])
edge_index = edge_index[2:, :]

dictionary_features = {}

for j, graph in enumerate(list(graphs.items())):
    edge_features = np.copy(edge_index)
    for i in range(len(edge_features)):
        g = graph[1] + graph[1].transpose()  #transpose to make undirected
        edge_features[i] = g[int(edge_index[i][0]), int(edge_index[i][1])]

    dictionary_features[j] = edge_features.reshape(2, 35)[0]

edge_feature_list = list(dictionary_features.values())

"""Convert from numpy to torch tensor

"""

edge_index = edge_index.reshape(2, 35)

edge_index = torch.from_numpy(edge_index)

edge_index = edge_index.type(torch.LongTensor)



"""Data format: Data(edge_index=[2, num_edges], x=[num_nodes, feature_vector], y=[target_labels/correct_labels])"""

# len(list(graphs.keys()))
dictionary = {}
for i in range(len(list(graphs.keys()))):
    x = torch.from_numpy(list(features.values())[i])
    edge_weight = torch.from_numpy(edge_feature_list[i])
    # edge_weight = edge_weight.type(torch.LongTensor)
    data = Data(x=x, edge_index=edge_index, edge_weights=edge_weight)
    key = "graph_" + str(i)
    dictionary[key] = data

"""#Recurrent GNN
Do we need a hidden state? Can we stick to simple markov process where input state is node states and output is edge weight.

Hidden state z - Amount of transcript

Input state x - Node states

Output y_hat - Edge Weight amount of metabolites.
"""

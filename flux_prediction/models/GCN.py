from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import MLP
from torch.nn import ReLU
import torch
from torch.optim import Adam
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import pickle
import numpy as np
import pandas as pd
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import BatchNorm1d
from torch.nn import Sequential
class GCN(torch.nn.Module):
    def poly_regression():
        #Write the equation here
        pass
    #Too large output_features: Make it 2 or 4.
    def __init__(self, input_features, output_features = 4):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(in_channels = input_features, out_channels = 16)
        self.conv2 = GCNConv(16, 16)
        self.conv3 = GCNConv(16,16)

        self.rates = Linear(16, 1)

    def forward(self,data_x,data_edge_index):
        x = self.conv1(data_x,data_edge_index)
        #Maybe try dropout()
        x = x.relu()
        x = self.conv2(x,data_edge_index)
        #Remove this relu()
        x = x.relu()
        x = self.conv3(x,data_edge_index)

        output_data = self.rates(x)
        return output_data


def train_GCN(model,train_data, optimizer, mse_loss):
    for key,data in train_data.items():
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x, data.edge_index)
        data.y = data.y.to(dtype = torch.float64)
        # print(out.dtype)
        #Also run without the stoich loss
        loss = mse_loss(out.double(), data.y)

        loss.backward()  # Derive gradients.
        optimizer.step()
    
    return loss
#Need to ammend this test function
def test_GCN(model,test_data, mse_loss):
    preds = []
    labels = []
    count = 0
    loss = 0
    sum_y = 0
    for key,data in test_data.items():
        #Perform forward pass
        out = model(data.x,data.edge_index)
        actual = data.y
        preds = np.append(preds, out.detach().numpy())
        labels = np.append(labels, actual.detach().numpy())

        count += 1
        sum_y += torch.sum(data.y)
        # print(pred)
        loss += mse_loss(out, data.y)
        
    return loss/sum_y, preds, labels




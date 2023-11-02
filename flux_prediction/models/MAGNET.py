from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv
from torch_geometric_signed_directed.nn import MagNetConv

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

class MAGNET(torch.nn.Module):

    def poly_regression():
        #Write the equation here
        pass

    #Too large output_features: Make it 2 or 4.
    def __init__(self, input_features, half_hidden_features=8, K=2, q=0.25):
        super(MAGNET, self).__init__()
        torch.manual_seed(12345)
        
        self.conv1 = MagNetConv(
            in_channels=input_features,
            out_channels=half_hidden_features*2,
            K=K, q=q, trainable_q=False)

        self.conv2 = MagNetConv(
            in_channels=half_hidden_features*2,
            out_channels=half_hidden_features*2,
            K=K, q=q, trainable_q=False)

        self.conv3 = MagNetConv(
            in_channels=half_hidden_features*2,
            out_channels=half_hidden_features,
            K=K, q=q, trainable_q=False)

        self.rates = Linear(16, 1)

    def forward(self, data_x, data_edge_index):

        x_real, x_img = self.conv1(data_x, data_x, data_edge_index)
        x_real = x_real.relu()
        x_real, x_img = self.conv2(x_real, x_img, data_edge_index)
        x_real = x_real.relu()
        x_real, x_img = self.conv3(x_real, x_img, data_edge_index)
        x = torch.cat([x_real, x_img], dim=-1)
        output_data = self.rates(x)
        return output_data

def train_MAGNET(model,train_data, optimizer, mse_loss, S_bool):
    for key,data in train_data.items():
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x, data.edge_index)
        data.y = data.y.to(dtype = torch.float64)
        Smat = torch.from_numpy(S_bool).float()
        # print(out.dtype)
        stoich_ineq = torch.matmul(Smat, out.float())
        stoich_loss = mse_loss(stoich_ineq, torch.zeros(72))
        #Also run without the stoich loss
        loss = mse_loss(out.double(), data.y)
        globals()['stoich_ineq'] = stoich_ineq
        globals()['stoich_loss'] = stoich_loss
        globals()['out'] = out
        globals()['loss'] = loss

        loss.backward()  # Derive gradients.
        optimizer.step()
    
    return loss

def test_MAGNET(model,test_data, mse_loss):
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
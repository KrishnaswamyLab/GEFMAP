from sklearn.model_selection import train_test_split
import pickle as pk
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from scipy.linalg import null_space

class BaseLoader(Dataset):
    def __init__(self, X, Y):
        self.X = X.astype('float32')
        self.Y = Y.astype('float32')
    def __getitem__(self, item):
        return (self.X[item], self.Y[item])
    def __len__(self):
        return len(self.X)


def process_data_FCN(data, S_matrix, randomseed, num_workers = 2, batch = 32):
    ns = null_space(S_matrix)
    ns = ns.astype('float32')
    S_matrix = S_matrix.astype('float32')

    X = []
    Y = []
    for i, j in data.items():
        X.append(j.x)
        Y.append(j.y)


    X = [j.x.numpy() for t,j in data.items()]

    Y = [j.y.numpy() for t,j in data.items()]

    # print(X)
    # print(Y)
    X = np.array(X)
    Y = np.array(Y)

    #save in datast dict
    dataset = {}
    dataset['X'] = X
    dataset['Y'] = Y
    dataset['ns'] = ns
    dataset['S'] = S_matrix

    X_train, X_test,Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=randomseed)
    X_train, X_val, Y_train, Y_val  = train_test_split(X_train, Y_train, shuffle=True, random_state=randomseed, test_size=0.2)

    ns, train, val, test = ns, BaseLoader(X_train, Y_train), BaseLoader(X_val, Y_val), BaseLoader(X_test, Y_test)
    train_loader = DataLoader(train, num_workers=num_workers, pin_memory = True, batch_size=batch, shuffle=True, drop_last=False)
    val_loader = DataLoader(val, num_workers=num_workers, pin_memory = True, batch_size=batch, shuffle=False, drop_last=True)
    test_loader = DataLoader(test, num_workers=num_workers, pin_memory = True, batch_size=batch, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader,ns


#######################--MLP Preprocessing--#######################
def process_data_MLP(data, S_matrix, randomseed, num_workers = 2, batch = 32):
    S_matrix = S_matrix.astype('float32')

    X = []
    Y = []
    for i, j in data.items():
        X.append(j.x)
        Y.append(j.y)


    X = [j.x.numpy() for t,j in data.items()]

    Y = [j.y.numpy() for t,j in data.items()]

    # print(X)
    # print(Y)
    X = np.array(X)
    Y = np.array(Y)

    #save in datast dict
    dataset = {}
    dataset['X'] = X
    dataset['Y'] = Y
    dataset['S'] = S_matrix

    X_train, X_test,Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=randomseed)
    X_train, X_val, Y_train, Y_val  = train_test_split(X_train, Y_train, shuffle=True, random_state=randomseed, test_size=0.2)

    train, val, test = BaseLoader(X_train, Y_train), BaseLoader(X_val, Y_val), BaseLoader(X_test, Y_test)
    train_loader = DataLoader(train, num_workers=num_workers, pin_memory = True, batch_size=batch, shuffle=True, drop_last=False)
    val_loader = DataLoader(val, num_workers=num_workers, pin_memory = True, batch_size=batch, shuffle=False, drop_last=True)
    test_loader = DataLoader(test, num_workers=num_workers, pin_memory = True, batch_size=batch, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader

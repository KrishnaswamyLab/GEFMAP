import os, math, time
import numpy as np
import pickle as pk
import pandas as pd
import networkx as nx
import matplotlib.pylab as plt
from scipy.linalg import null_space
from scipy.optimize import linprog

# load file
f = pd.read_csv('./e_coli_S_matrix.csv')  
S = f.iloc[:,1:].to_numpy()

# target reaction
c = np.zeros(S.shape[1])
c[24] = -1

ns = null_space(S)

# list all simulated pickle files
root ='./ecoli/'
files = os.listdir(root)

# store data
X = []
bound = [] 
for f in files:
    if f[-3:] == '.pk' and f != 'data.pk':
        data = pk.load(open(root + f, 'rb'))
        X.extend(data['X'])
        bound.extend(data['bound'])

print(np.array(X).shape, np.array(bound).shape)

data_X = []
data_bound = []
for x, b in zip(X, bound):
    
    # skip nan and values out of bounds
    if np.sum(np.isnan(x)): continue
    if np.sum(x < b[:,0]) or np.sum(x > b[:,1]): 
        continue
    
    data_X.append(x)
    data_bound.append(b)

print(np.array(data_X).shape, np.array(data_bound).shape)

# create pickle
dataset = {}
dataset['X'] = np.array(data_bound).transpose((0,2,1)).astype('float32')
dataset['Y'] = np.array(data_X).astype('float32')
dataset['ns'] = ns.astype('float32')
dataset['S'] = S.astype('float32')
pk.dump(dataset, open('ecoli/data.pk', 'wb'))
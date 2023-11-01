from sklearn.model_selection import train_test_split
import pickle as pk
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

def load_data(out_path, bounds_file, solution = None, S_matrix_file = None, RAG_file = None):
    if S_matrix_file is not None:
        # print("going to ecoli")
        with open(out_path+ bounds_file, 'rb') as b:
            graphs = pk.load(b)
        
        bounds = graphs['bounds']
        solution = 'solutions_'+solution
        solutions = graphs[solution]
        
        
        S_matrix = pd.read_csv(out_path + S_matrix_file, index_col = 0)
        S_bool = S_matrix[S_matrix == 0].fillna(1).to_numpy()
        RAG = np.matmul(np.transpose(S_bool), S_bool)
        RAG = pd.DataFrame(RAG, index = S_matrix.columns, columns = S_matrix.columns)

    else:
        # print("going to toy")
        with open(out_path+ bounds_file, 'rb') as b:
            bounds = pk.load(b)

        with open(out_path+ solution, 'rb') as b:
            solutions = pk.load(b)

        S_matrix = np.array([[1, -1, 0, 0, 0, 0, 0, 0],
                        [0, 1, -1, -1, 0, 0, 0, 0],
                        [0, 0, 1, 0, -1, 0, 0, -1],
                        [0, 0, 0, 1, 0, -1, -1, -2],
                        [0, 0, 0, 0, 1, 1, 0, -1]])
        
        RAG = pd.read_csv(out_path + RAG_file,index_col = 0)
    
    return bounds, solutions, S_matrix, RAG

def pre_process_data(bounds, solutions, RAG, data_type):
    '''
    Processes the bounds, solutions and RAG to obtain a torch_geometric data to feed into the GNN

    Args:
        bounds: Upper and lower bounds (input x)
        solutions: FBA solutions (labels y)
        RAG: Reaction Adjacency Matrix (edge indices)
    Returns:
        dictionary: keys - graph_num, values - torch_geometric data objects
    '''
    edge_index = np.zeros([1,2])
    for i, row in enumerate(RAG.index):
        for f, col in enumerate(RAG.columns):
            if RAG.iloc[i,f] > 0:
                p = ([i,f])
                edge_index = np.vstack([edge_index, p])
    edge_index = edge_index[2:,:]
    edge_index = edge_index.reshape(edge_index.shape[1],edge_index.shape[0])
    edge_index = torch.from_numpy(edge_index)
    edge_index = edge_index.type(torch.LongTensor)
    if data_type == "ecoli":
        # print("going to ecoli")
        dictionary = {}
        for i, curr_bounds in enumerate(bounds):

            x = torch.from_numpy(curr_bounds.reshape(95,2)).float() #both upper and lower bounds
            # x = torch.from_numpy(curr_bounds[:,0].reshape(95,1)).float() #just upper bound
            y = torch.from_numpy(solutions[i].reshape(95,1)).float()

            data = Data(x=x, y=y, edge_index=edge_index)
            key = "graph_"+str(i)
            dictionary[key] = data
    elif data_type == "toy":
        # print("going to toy")
        dictionary = {}
        for i, curr_bounds in enumerate(bounds):

            x = torch.from_numpy(list(bounds.values())[i].reshape(8,1)).float()
            y = torch.from_numpy(np.array(list(solutions.values())[i]).reshape(8,1)).float()

            data = Data(x=x, y=y, edge_index=edge_index)
            key = "graph_"+str(i)
            dictionary[key] = data
    
    return dictionary
    

def split_data(data, split, random_state):
    graph_names = list(data.keys())
    data_objects = list(data.values())

    test_size = split

    # Split the indices of the Data objects into training and testing sets
    indices = list(range(len(data_objects)))
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)

    train_data = {graph_names[i]: data_objects[i] for i in train_indices}
    test_data = {graph_names[i]: data_objects[i] for i in test_indices}

    return train_data, test_data


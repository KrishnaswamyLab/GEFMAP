import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
# import seaborn as sns

from GSMM import metabolic_model
from GSMM import models
from GSMM import base_graph
from GSMM import merge_tx_model
import pandas as pd
import numpy as np
import pickle as pk
import os
import sys
import libsbml
module_path = os.path.abspath(os.path.join('../src/'))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils import *
from GSMM import *

def pre_process_human_data():

    #GSMM
    models_dir = os.path.abspath('../src/sbml_models/')
    organism = "homo_sapiens"
    sbml_file = libsbml.readSBMLFromFile(os.path.join(models_dir, models[organism]))
    gene_exp_scale = 100
    bound_max = 1000
    m_model = metabolic_model(organism, sbml_file, bound_max=bound_max)

    #matrix
    S = m_model.S
    S_matrix = m_model.S_matrix
    reactions = list(m_model.reactions.keys())
    metabolites = list(m_model.metabolites.keys())

    S_bool = np.array(S_matrix.astype('bool').astype('int'))
    S_bool = sp.csr_matrix(S_bool)
    RAG = S_bool.T.dot(S_bool)
    adj = RAG.astype('bool').astype('int') 
    adj = adj - np.identity(adj.shape[0])
    A = sp.csr_matrix(adj)
    # print(nx.__version__)
    G = nx.from_scipy_sparse_array(A)


    data_temp_path = '../data/'
    with open(os.path.join(data_temp_path, 'adata_prepross_all.pk'), 'rb') as handle:
        adata = pk.load(handle)

    with open(os.path.join(data_temp_path, 'data_magic.pk'), 'rb') as handle:
        data_magic = pk.load(handle)

    adata.var_names_make_unique()
    dg = [g.split(" (")[0] for g in adata.var_names]
    # data = pd.DataFrame(adata.X, index = adata.obs_names, columns =  dg)
    data = pd.DataFrame(data_magic, index = adata.obs_names, columns =  dg)
    #delta gene exp
    data_all = data.copy()
    t0 = [str(index).__contains__('1A') for index in data.index]
    ctrl = data[t0].mean()
    data = data - ctrl
    data.drop(data_all[t0].index, inplace=True)

    #subset to neuronal
    tL =[str(index).__contains__('T8') for index in data.index]
    df = data[tL]
    data = df
    #neuronal mask
    data = data[data['ONECUT2'] > 0.2]
    #filter names
    model_symbol = list(m_model.genes.keys())
    model_BIGG = list(m_model.genes.values())
    data_genes = data.columns
    matching_genes = [gene for gene in data_genes if gene in model_symbol]
    model_BIGGid = [m_model.genes[i] for i in matching_genes]
    data = data[matching_genes]
    # df.columns = model_BIGGid
    # data = df
    data.columns = model_BIGGid


    # data_all = data.copy()
    # t0 = [str(index).__contains__('1A') for index in data.index]
    # ctrl = data[t0].mean()
    # data = data - ctrl
    # data.drop(data_all[t0].index, inplace=True)


    rga = 0
    rnone = 0
    no_gene = []
    reaction_exp_matrix = {}
    for i in range(0,len(m_model.reactions)):
        rc = list(m_model.reactions.values())[i]
        if rc.gene_associations is None:
            rnone += 1
            pass
        else:
            gene_association = rc.gene_associations
            try:
                rc.get_reaction_expression(data.T)
                reaction_exp_matrix[rc.id] = rc.reaction_expression
            except:
                no_gene += [i]
                rga+=1
    reaction_exp_matrix = pd.DataFrame(reaction_exp_matrix)
    reaction_exp_matrix = reaction_exp_matrix.fillna(0)
    R = reaction_exp_matrix
    srm = reaction_exp_matrix*gene_exp_scale
    R, srm = merge_tx_model(m_model, R, srm)

    # node_rxn_labels = {n:srm.columns[i] for i, n in enumerate(G.nodes)}
    # graph = base_graph(G, m_model)
    G_ = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    g = G_[0]
    R_all = R.copy()
    srm_all = srm.copy()
    R = R.loc[:,np.abs(srm).sum()>0]
    srm = srm.loc[:,np.abs(srm).sum()>0]
    filt_nodes = [srm_all.columns.get_loc(i) for i in srm.columns]

    # filt_nodes = filt_nodes[:20]


    A_sub = A[filt_nodes][:,filt_nodes]
    G = nx.from_scipy_sparse_array(A_sub)
    graph = base_graph(G, m_model)
    subgraphs = [g for g in nx.connected_components(graph.G)]
    # G_ = [graph.G.subgraph(c).copy() for c in nx.connected_components(graph.G)]
    node_vector = []
    for gsub in G_:
        node_vector.extend(list(gsub.nodes))


    # g = G_[0]
    # R_all = R.copy()
    # srm_all = srm.copy()
    # mask = np.random.randint(0, srm.shape[0], 3000)
    # R = R.loc[:,np.abs(srm).sum()>0]
    # srm = srm.loc[:,np.abs(srm).sum()>0]
    # filt_nodes = [srm_all.columns.get_loc(i) for i in srm.columns]
    # # filt_nodes = filt_nodes[:20]
    # A_sub = A[filt_nodes][:,filt_nodes]
    # G = nx.from_scipy_sparse_array(A_sub)
    # graph = base_graph(G, m_model)
    graph.get_universal_features()
    f = graph.get_input_features(srm, R)
    return f

if __name__ == "__main__":
    print("Starting pre-processing..")
    features = pre_process_human_data()
    pk_save(features, 'input_features_final.pk')
    # features = pk_load('input_features_final.pk')
    # print(len(features))
    # print(features[0])
    print("DONE!")


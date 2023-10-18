from utils import *
import torch

from torch.optim import Adam
from torch.nn import ReLU
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import BatchNorm1d
from torch.nn import Sequential
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import MLP

from utils_ecoli import *
from sklearn.preprocessing import MinMaxScaler

import time, argparse
from scipy.optimize import linprog
method = 'simplex'
options = {'tol': 1e-6}

#############################################
#FBA gt
#############################################

from GSMM import *




def run_gt_solutions(G, srm, R, metabolic_model, inplace = False):
    from utils_ecoli import FBA_ecoli


    node_rxn_labels = {n:srm.columns[i] for i, n in enumerate(G.nodes)}
    graph = base_graph(G)
    f = graph.get_input_features(srm, R)
    mw_nodes, mw_weights = graph.gt_clique()

    FBA = FBA_ecoli(metabolic_model, ATP = 8.39, glucose = -10)
    bounds = FBA.get_constrained_bounds(R, bound_scale = 1)
    bio_c, clique_c = FBA.get_objective_functions(mw_nodes, with_biomass = False, b_weight = 0.8)

    RES_bio, solutions_bio, failed_bio = FBA.get_solutions(bio_c)
    RES_clique, solutions_clique, failed_clique = FBA.get_solutions(clique_c)

    failed = failed_bio + failed_clique
    if failed > 0:
        raise ValueError("An error occurred")
    return f, mw_nodes, mw_weights, bounds, bio_c, clique_c, RES_bio, RES_clique



def gt_clique(base_graph):
    mw_nodes = {}
    mw_weights = {}
    for j, samp_id in enumerate(base_graph.srm.index):
        G_ = base_graph.G.copy()
        G_exp = {}
        for i, n in enumerate(G_.nodes):
            n_exp = base_graph.srm.iloc[j,i]
            G_exp[n] = int(n_exp) 
        nx.set_node_attributes(G_, G_exp, "G_exp")
        mw_nd, mw_wt = nx.max_weight_clique(G_, weight="G_exp")
        mw_nodes[j] = mw_nd
        mw_weights[j] = mw_wt
    return mw_nodes, mw_weights



######################################################################
######################################################################


class FBA_ecoli():
    def __init__(self, model, ATP = 8.39, glucose = -10):
        self.model = model
        self.S = model.S_matrix.to_numpy()
        self.n_reaction = self.S.shape[1]
        self.ATP = ATP
        self.glucose = glucose
        #self.oxygen = oxygen
        self.ub = [x.ub for x in list(model.reactions.values())]
        self.lb = [x.lb for x in list(model.reactions.values())]

        for i, reaction in enumerate(model.reactions.values()):
            if isinstance(self.ATP, (int, float)):
                if reaction.id == 'R_ATPM':
                    self.lb[i] = self.ATP
                    print(reaction.id, 'bounds set to....... ', self.lb[51], self.ub[51])
            if isinstance(self.glucose, (int, float)):
                if reaction.id == 'R_EX_glc__D_e':
                    self.lb[i] = self.glucose
                    print(reaction.id, 'bounds set to....... ', self.lb[15], self.ub[15])
            elif reaction.is_exchange(): #
                self.lb[i] = -1000
                if reaction.reversible:
                    self.ub[i] = 1000
                else:
                    self.ub[i] = 0

    def get_constrained_bounds(self, R, bound_scale = 1):
        self.R = R
        bounds = []

        for s in range(0,R.shape[0]):
            gene_expression = R.iloc[s,:]
            ub_ = gene_expression * self.ub * bound_scale
            lb_ = gene_expression * self.lb * bound_scale
            bounds +=[np.c_[lb_,ub_]]
        self.bounds = bounds

        return self.bounds

    def get_objective_functions(self, mw_nodes, with_biomass = False, b_weight = 0.8):

        bio_c = []
        clique_c = []

        for s, b in enumerate(self.bounds):
            #biomass
            c = np.zeros(self.S.shape[1])
            c[24] = -1
            bio_c.append(c)
            #max wt clique
            c = np.zeros(self.S.shape[1]) ##
            obj_reactions = mw_nodes[s]
            for n in obj_reactions:
                c[n] = -1
            scaler = MinMaxScaler(feature_range=(-1, 0)) 
            c = scaler.fit_transform(np.array(c*self.R.iloc[s,:]).reshape(-1,1)).flatten()
            if with_biomass:
                c[24] = b_weight
            clique_c.append(c)

        self.biomass_c = bio_c 
        self.max_wt_clique_c = clique_c
        return bio_c, clique_c


    def get_solutions(self, c_list):
        failed = 0
        RES = []
        solutions = []

        for s, bounds_ in enumerate(self.bounds):

            c = c_list[s]
            try:
                res = linprog(c,A_ub=None, b_ub=None, A_eq = self.S, b_eq = np.zeros(self.S.shape[0]),
                    bounds=bounds_) #, method= method
                if res.success and res.status == 0:
                    RES.append(res)
                    solutions.append(res.x)
                else:
                    print(f'failed........{s, self.R.index[s]} with error {res.status}')
                    failed +=1
            except:
                print(f'failed to run')
        return RES, solutions, failed
        



######################################################################
###################################################################### archive_augment

#sequential commands to generate augmented (+gaussian_noise) data

'''
from utils_ecoli import *
from utils_ecoli import run_gt_solutions


target_samples = 3000
noise_std = 1.5

if 'gene_exp_scale' not in globals():
    print(f'need to define gene_exp_scale...... setting to default = 10')
    gene_exp_scale = 10

n_ = 1
success = 0
n_samples = srm.shape[0]
aug_iter =( target_samples // n_samples)

inputs_save = []
d_save = []


while success < aug_iter:
    n_ += 1
    np.random.seed(n_)
    print(n_, success)
    gene_exp_noisy = data_norm + np.random.normal(0, noise_std, data_norm.shape)
    gene_exp_noisy = gene_exp_noisy.abs()
    reaction_exp_matrix_noisy = map_reactions(model, gene_exp_noisy)
    R_, srm_ = scale_reaction_matrix(reaction_exp_matrix_noisy, gene_exp_scale, plot= False)
    R_n, srm_n = merge_tx_model(model, R_, srm_)
    try:
        f, mw_nodes, mw_weights, bounds, bio_c, clique_c, RES_bio, RES_clique = run_gt_solutions(G, srm_n, R_n, model)
        print('saving vars')
        d_save += [[f, mw_nodes, mw_weights, bounds, bio_c, clique_c, RES_bio, RES_clique]]
        inputs_save += [[gene_exp_noisy, R_n, srm_n]]
        success += 1

    except ValueError as error:
        continue


        


'''


'''
#concat
f = []
mw_nodes = []
mw_weights = []
bounds = []
bio_c = []
clique_c = []
RES_bio = []
RES_clique = []


f_og, mw_nodes_og, mw_weights_og, bounds_og, bio_c_og, clique_c_og, RES_bio_og, RES_clique_og = run_gt_solutions(G, srm, R, model)

f += f_og
mw_nodes += mw_nodes_og
mw_weights += mw_weights_og
bounds += bounds_og
bio_c += bio_c_og
clique_c += clique_c_og
RES_bio += RES_bio_og
RES_clique += RES_clique_og

for var in d_save: #f, mw_nodes, bio_c, clique_c, RES_bio, RES_clique
    f += var[0]
    mw_nodes += var[1]
    mw_weights += var[2]
    bounds += var[3]
    bio_c += var[4]
    clique_c += var[5]
    RES_bio += var[6]
    RES_clique += var[7]



gexp_aug = data_norm.copy()
R_aug = R.copy()
srm_aug = srm.copy()

for var in inputs_save: #gene_exp_noisy, R_n, srm_n
    gexp_aug= pd.concat([gexp_aug, var[0]], axis = 1)
    R_aug= pd.concat([R_aug, var[1]], axis = 0)
    srm_aug= pd.concat([srm_aug, var[2]], axis = 0)



print(len(f), 
      gexp_aug.shape, 
      R_aug.shape, 
      srm_aug.shape
      )

'''


'''
graph =  base_graph(G)
graph.srm = srm_aug
graph.R = R_aug

graph.f = f
graph.mw_nodes = mw_nodes
graph.mw_weights = mw_weights
graph.bounds = bounds
graph.bio_c  = bio_c
graph.clique_c  = clique_c
graph.res_bio  = RES_bio
graph.res_clique  = RES_clique


os.listdir(data_outs)
sub_dir = 'augmented_data_current'
f_ = 'aug_graph_obj'
fname = os.path.join(data_outs, sub_dir, f'{f_}.pk')
pk_save(graph, fname)

'''


######################################################################
###################################################################### archive

'''
from utils_ecoli import augment_genelevel_data
aug_gene_dict = augment_genelevel_data(metabolic_model, data_norm, noise_std = 0.1, new_samp_number = 1000, scales = 2)

from utils_ecoli import augment_rxnlevel_data
aug_rxn_dict = augment_rxnlevel_data(metabolic_model, data_norm, noise_std = 0.1, new_samp_number = 1000, scales = 2)

from utils_ecoli import run_gt_solutions
#testing tolerance to noise
for n in range(2):
    dd = [aug_gene_dict, aug_rxn_dict][n]
    d_curr = list(dd.values())[1]
    print(list(d_curr.keys()))
    try:
        exp, R, srm = list(d_curr.values())
        print('gene')
        run_gt_solutions(G, srm, R, metabolic_model, inplace = True)
    except:
         R, srm = list(d_curr.values())
         print('reaction')
         run_gt_solutions(G, srm, R, metabolic_model, inplace = True)
'''


def augment_genelevel_data(metabolic_model, data_norm, noise_std, new_samp_number = 3000, scales = 3):
    model = metabolic_model
    if 'gene_exp_scale' not in globals():
        print(f'need to define gene_exp_scale...... setting to default = 10')
        gene_exp_scale = 10
    else:
        gene_exp_scale = globals()['gene_exp_scale']

    reaction_exp_matrix = map_reactions(model, data_norm)
    R, srm = scale_reaction_matrix(reaction_exp_matrix, gene_exp_scale, plot= False)
    R, srm = merge_tx_model(model, R, srm)
    
    data_dict = {}
    n_samples = srm.shape[0]
    aug_iter =( new_samp_number // n_samples)-1

    noise_curr = 0
    for n in range(0,scales):
        noise_curr += noise_std
        lab = f'data_gaussNoise_{str(noise_curr)[:4]}'
        print(lab)
        ### tosave
        gene_expression = data_norm.copy()
        gene_expression.columns = gene_expression.columns.map(lambda x: x + f'_original')

        ####
        #### generate new samples and concatenate
        for n_ in range(aug_iter):
            gene_exp_noisy = data_norm + np.random.normal(0, noise_curr, data_norm.shape)
            gene_exp_noisy = gene_exp_noisy.abs()
            gene_exp_noisy.columns = gene_exp_noisy.columns.map(lambda x: x + f'_{n_}')
            gene_expression = pd.concat([gene_expression,gene_exp_noisy], axis = 1)
            
        reaction_exp_matrix_noisy = map_reactions(model, gene_expression)
        R_, srm_ = scale_reaction_matrix(reaction_exp_matrix_noisy, gene_exp_scale, plot= False)
        R_aug, srm_aug = merge_tx_model(model, R_, srm_)

        new_data = {"gene_expression":gene_expression,
                    "R_aug":R_aug,
                    "srm_aug":srm_aug}
        data_dict[lab] = new_data
    return data_dict
        



def augment_rxnlevel_data(metabolic_model, data_norm, noise_std, new_samp_number = 3000, scales = 3):
    model = metabolic_model
    if 'gene_exp_scale' not in globals():
        print(f'need to define gene_exp_scale...... setting to default = 10')
        gene_exp_scale = 10
    else:
        gene_exp_scale = globals()['gene_exp_scale']

    reaction_exp_matrix = map_reactions(model, data_norm)
    R_, srm_ = scale_reaction_matrix(reaction_exp_matrix, gene_exp_scale, plot= False)
    R, srm = merge_tx_model(model, R_, srm_)
    
    data_dict = {}
    n_samples = srm.shape[0]
    aug_iter =( new_samp_number // n_samples)-1

    noise_curr = 0
    for n in range(0,scales):
        noise_curr += noise_std
        lab = f'data_gaussNoise_{str(noise_curr)[:4]}'
        print(lab)

        srm_aug = srm.copy()
        R_aug = R.copy()

        for n_ in range(aug_iter):
            noise_add = np.random.normal(0, noise_curr, R_.shape)
            srm_noise = srm_.copy() + noise_add
            srm_noise = srm_noise.abs()
            R_noise = R_.copy() + (noise_add/10)
            R_noise = R_noise.abs()

            R_ns, srm_ns = merge_tx_model(model, R_noise, srm_noise) 
        
            srm_aug = pd.concat([srm_aug, srm_ns], axis = 0)
            R_aug = pd.concat([R_aug, R_ns], axis = 0)

        new_data = {"R_aug":R_aug,
            "srm_aug":srm_aug}
        
        data_dict[lab] = new_data
    return data_dict
from utils import *
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import linprog
method = 'simplex'
options = {'tol': 1e-6}

#############################################
############################################# #FBA gt



class FBA_ecoli():
    def __init__(self, base_graph, ATP = 8.39, glucose = -10, verbose = True):
        self.base_graph = base_graph
        self.m_model = base_graph.m_model
        self.S = base_graph.m_model.S_matrix.to_numpy()
        self.n_reaction = self.S.shape[1]
        self.ATP = ATP
        self.glucose = glucose
        #self.oxygen = oxygen
        self.R = self.base_graph.R
        self.srm = self.base_graph.srm
        self.ub = [x.ub for x in list(base_graph.m_model.reactions.values())]
        self.lb = [x.lb for x in list(base_graph.m_model.reactions.values())]

        for i, reaction in enumerate(base_graph.m_model.reactions.values()):
            if isinstance(self.ATP, (int, float)):
                if reaction.id == 'R_ATPM':
                    self.lb[i] = self.ATP
                    if verbose:
                        print(reaction.id, 'bounds set to....... ', self.lb[51], self.ub[51])
            if isinstance(self.glucose, (int, float)):
                if reaction.id == 'R_EX_glc__D_e':
                    self.lb[i] = self.glucose
                    if verbose:
                        print(reaction.id, 'bounds set to....... ', self.lb[15], self.ub[15])
            elif reaction.is_exchange(): #
                self.lb[i] = -1000
                if reaction.reversible:
                    self.ub[i] = 1000
                else:
                    self.ub[i] = 0

    def get_constrained_bounds(self, bound_scale = 1, return_bounds = True):
        sample_bounds = []
        for s, samp in enumerate(self.R.index):
            bounds = []
            for s in range(0,self.R.shape[0]):
                gene_expression = self.R.iloc[s,:]
                ub_ = gene_expression * self.ub * bound_scale
                lb_ = gene_expression * self.lb * bound_scale
                bounds += [np.c_[lb_,ub_]]
            sample_bounds += [bounds]
        self.bounds = bounds
        if return_bounds:
            return bounds
    
    def get_gt_clique(self, return_vals = True):
        mw_nodes = []
        mw_weights = []
        for j, samp_id in enumerate(self.srm.index):
            G_ = self.base_graph.G.copy()
            G_exp = {}
            for i, n in enumerate(G_.nodes):
                n_exp = self.srm.iloc[j,i]
                G_exp[n] = int(n_exp) 
            nx.set_node_attributes(G_, G_exp, "G_exp")
            mw_nd, mw_wt = nx.max_weight_clique(G_, weight="G_exp")
            mw_nodes += [mw_nd]
            mw_weights += [mw_wt]
        self.mw_nodes = mw_nodes
        self.mw_weights = mw_weights
        mw_node_len = [len(c) for c in self.mw_nodes]
        self.mw_node_len = mw_node_len
        if return_vals:
            return mw_nodes, mw_weights, mw_node_len

    
    def get_gt_obj(self, return_objectives = True, with_biomass = False, b_weight = 0.8):
        self.G = self.base_graph.G
        self.f = self.base_graph.f

        #biomass
        c = np.zeros(self.S.shape[1])
        c[24] = -1 #biomass rxn idx
        self.biomass_obj = c
        #gt clique
        cliques = []
        for s, b in enumerate(self.bounds):
            c = np.zeros(self.S.shape[1])
            obj_reactions = self.mw_nodes[s]
            for n in obj_reactions:
                c[n] = -1
            scaler = MinMaxScaler(feature_range=(-1, 0)) 
            c = scaler.fit_transform(np.array(c*self.R.iloc[s,:]).reshape(-1,1)).flatten()
            if with_biomass:
                c[24] = b_weight
            cliques += [c]
        self.gt_clique_obj = cliques
        if return_objectives:
            return self.biomass_obj, self.gt_clique_obj




    def get_lp_solutions(self, S_matrix, return_vals = True):
        from scipy.optimize import linprog

        RES_b = []
        RES_c = []
        solutions_b = []
        solutions_c = []
        failed = 0
        f_rm = []

        for h  in range(self.base_graph.N):
            bounds = self.bounds[h]
            c_b = self.biomass_obj
            c_c = self.gt_clique_obj[h]

            res_b = linprog(c_b,A_ub=None, b_ub=None, A_eq = S_matrix, b_eq = np.zeros(S_matrix.shape[0]),
                            bounds=bounds)
            
            res_c = linprog(c_c,A_ub=None, b_ub=None, A_eq = S_matrix, b_eq = np.zeros(S_matrix.shape[0]),
                            bounds=bounds)
            
            if res_b.success and res_c.success:
                RES_b.append(res_b)
                RES_c.append(res_c)
                solutions_b.append(res_b.x)
                solutions_c.append(res_c.x)
            
            else:
                # print(f'failed.......')
                failed += 1
                f_rm += [h]

        self.RES_b = RES_b
        self.RES_c = RES_c
        self.solutions_b = solutions_b
        self.solutions_c = solutions_c
        self.failed = failed
        self.f_rm = f_rm
        
        if return_vals:
            return RES_b, RES_c, solutions_b, solutions_c, failed, f_rm


        



















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
'''

import libsbml
import scipy
import networkx as nx
import numpy as np

from utils import *

##############################################################################
############################################################################## init

from tabulate import tabulate
from six import string_types


models = {['ECOLI_core', 'mus_musculus', 'homo_sapiens'][i]:org for i, org in 
          enumerate(['e_coli_core_SBML3.xml', 'mus_iMM1415.xml', 'Recon3D.xml'])}

models_ = tabulate(list(models.items()), ['ORGANISM', 'MODEL_SBML'],tablefmt="grid")



############################################################ 
############################################################ from sbml

class association(object):
    def __init__(self):
        self.type = ''
        self.gene = ''
        self.children = []

    
def map_bool(gene_association, data):
    #gene_association = reaction.gene_associations
    if gene_association.type == 'gene' and gene_association.gene in data.index:
        rexp = data.loc[gene_association.gene, :]

    elif gene_association.children is not None:
        if all(cp.type == 'gene' for cp in gene_association.children):
            gexp = data.loc[[ac.gene for ac in gene_association.children if ac.gene in data.index], :]
            if gene_association.type == 'and':
                rexp = np.min(gexp, axis = 0)
            elif gene_association.type == 'or':
                rexp = np.max(gexp, axis = 0)
    return  rexp 



class Reaction(object): 
    '''for type XML, HS_currrent
    '''
    def __init__(self, sbml_reaction, xml_params, convert = False, convert_dict = None):
        self.sbml_reaction = sbml_reaction
        self.fbc = sbml_reaction.getPlugin('fbc')
        self.id = sbml_reaction.id
        self.name = sbml_reaction.name
        self.reversible = sbml_reaction.getReversible()
        self.compartment = sbml_reaction.getCompartment()
        self.gpa = self.fbc.getGeneProductAssociation() 

        ############ ub/lb ########################
        #method accounts for reversability

        ub = self.fbc.getUpperFluxBound()
        if isinstance(ub, string_types):
            ub = xml_params[ub]
        lb = self.fbc.getLowerFluxBound()
        if isinstance(lb, string_types):
            lb = xml_params[lb]

        self.ub = ub
        self.lb = lb
        
        ############ substrate/products ############
        
        self.reactants = {}
        for r in sbml_reaction.getListOfReactants():
            metabolite = r.getSpecies()
            coeff = r.getStoichiometry()
            self.reactants.update({metabolite: coeff})

        self.products = {}
        for r in sbml_reaction.getListOfProducts():
            metabolite = r.getSpecies()
            coeff = r.getStoichiometry()
            self.products.update({metabolite: coeff})
            

        ############ gene to reaction mapping ##########################
        
        if self.gpa is None:
            self.gene_associations = None
            
        else:
            self.q = self.gpa.getAssociation()
            self.gene_associations = reaction_genes(self.q)
            
           
        
    # ex/intracellular transport
    
    def is_exchange(self):
        if len(self.products) == 0 and len(self.reactants) > 0:
            return True
        elif len(self.reactants) == 0 and len(self.products) > 0:
            return True
        else:
            return False


    def get_reaction_expression(self, data):
        
        genes = []
        
        
        if self.gene_associations is not None:
            gene_association = self.gene_associations

            try:
                self.reaction_expression  = map_bool(gene_association, data) #if no children or all genes
                genes.append(gene_association.gene)

            except:
                rexp = []
                f1 = gene_association.children #f1 reaction children
                for ac in f1:
                    if ac.children is None: 
                        gexp_f1 = map_bool(ac, data)
                        genes.append(ac.gene)
                    else: 
                        gexp_f1 = [] #f2 children of children
                        f2 = ac.children
                        for ac2 in f2:
                            if ac2.children is None:
                                gexp_f2 = map_bool(ac2, data)
                                genes.append(ac2.gene)
                            else:
                                gexp_f2 = []
                                f3 = ac2.children #f2 children of children of children
                                for ac3 in f3:
                                    gexp_f2.append(map_bool(ac3, data)) 
                                    genes.append(ac3.gene)

                                #bool gexp2 based on ac2
                                if ac2.type == 'and':
                                    gexp_f2 = np.min(gexp_f2, axis = 0)
                                elif ac2.type == 'or':
                                    gexp_f2 = np.max(gexp_f2, axis = 0)

                            gexp_f1.append(gexp_f2)

                    if ac.type == 'and':
                        gexp_f1 = np.min(gexp_f1, axis = 0)
                    elif ac.type == 'or':
                        gexp_f1 = np.max(gexp_f1, axis = 0)
                    
                    rexp.append(gexp_f1)

                if gene_association.type == 'and':
                    rexp = np.min(rexp, axis = 0)
                elif gene_association.type == 'or':
                    rexp = np.max(rexp, axis = 0) 
                
                if rexp is not None:
                    self.reaction_expression = rexp
        
        elif self.gene_associations is None:
            self.reaction_expression = None
        
        self.genes = genes

        
        
def reaction_genes(q):
    gene_associations = association()
    
    if isinstance(q, libsbml.FbcOr):
        gene_associations.type = 'or'
        gene_associations.gene = None
        gene_associations.children = [reaction_genes(ac) for ac in q.getListOfAssociations()]

    if isinstance(q, libsbml.FbcAnd):
        gene_associations.type = 'and'
        gene_associations.gene = None
        gene_associations.children = [reaction_genes(ac) for ac in q.getListOfAssociations()]

    if isinstance(q, libsbml.GeneProductRef):
        gp = q.getGeneProduct().split('_')[1].upper() #use for e.coli
        #human convert gp from entrez to symbol
        gene_associations.type = 'gene'
        gene_associations.gene = gp
        gene_associations.children = None
        

    return gene_associations
            



class metabolic_model(object):
    """Attributes
    ----------
    model
    model_name
    S
    genes
    metabolites
    reactions
    r_vector
    xml_params: cobra_default_lb, cobra_default_u
    """

    def __init__(self, model_name, sbml_Document, bound_max = None):
        self.model_name = model_name
        try:
            print(f'loading {models[model_name]}...')
        except:
            print(f'\"{model_name}\" not in model names {list(models.keys())}')

        self.level = sbml_Document.getLevel()
        if self.level != 3:
            print('warning: model not level 3')
            
        self.model = sbml_Document.model
        self.genes = {}
        self.metabolites = {}
        self.reactions = {}
        
        self.fbc = self.model.getPlugin("fbc")
        self.xml_params = {}
        for pp in self.model.getListOfParameters():
            key = pp.getId()
            val = pp.getValue()
            self.xml_params[key] = val
        
        if bound_max is not None:
            #print(f'bound max: {bound_max}')
            self.xml_params.update({'cobra_default_lb': -1*bound_max, 'cobra_default_ub': bound_max})


        
        ############ add genes ############
        
        for gp in self.fbc.getListOfGeneProducts():  # type: libsbml.GeneProduct
            key = gp.getName()
            
            v = gp.getId() #change to match gene expression data
            val = str(v).split('_')[1].upper()
            self.genes[key] = val
        
        #if self.model_name == 'homo_sapiens':
        #   print('converting gene IDs.........')
        #    self.genes = {y:x for x, y in list(self.genes.items())}
        ############ add metabolites (species) ############
        
        for mt in self.model.getListOfSpecies():
            self.metabolites[mt.id] = mt
            
        
        ############ add reactions ############
        
        for rxn in self.model.getListOfReactions():
            rc = Reaction(rxn, self.xml_params)
            self.reactions[rc.id] = rc
            
        
        ############ reversability vector ############
        self.r_vector = np.array([r.reversible for r in list(self.reactions.values())]).astype('int')
        
        
        ############ S matrix ############
        self.S = {}
        for reaction_id, rr in self.reactions.items():

            # reactants
            for metabolite, coefficient in rr.reactants.items():

                if metabolite not in self.S:
                    self.S[metabolite] = []

                self.S[metabolite].append((reaction_id, coefficient * -1))

            # products
            for metabolite, coefficient in rr.products.items():

                if metabolite not in self.S:
                    self.S[metabolite] = []

                self.S[metabolite].append((reaction_id, coefficient))

        n_ = int(len(self.reactions))
        m_ = int(len(self.metabolites))
        self.S_matrix = pd.DataFrame(np.zeros((m_,n_)),
                                     index = list(self.metabolites.keys()),
                                     columns = list(self.reactions.keys()))

        for metab, stoichs in self.S.items(): #for each metabolite (row)
            for reaction in stoichs: #for each associated reaction (column)
                self.S_matrix.loc[metab, reaction[0]] = int(reaction[1])
            
        print(f'{len(list(self.genes.values()))} genes')
        print(f'{len(list(self.metabolites.values()))} metabolites')
        print(f'{len(list(self.reactions.values()))} reactions with bound max {bound_max}')




############################################################ 
############################################################ transcript mapping
    

#map genes to reactions
def map_reactions(model, data):
    
    reaction_exp_matrix = {}
    for i in range(0,len(model.reactions)):
        rc = list(model.reactions.values())[i]
        rc.get_reaction_expression(data)
        if rc.reaction_expression is not None:
            reaction_exp_matrix[rc.id] = rc.reaction_expression

    reaction_exp_matrix = pd.DataFrame(reaction_exp_matrix)
    return reaction_exp_matrix


#scale
def scale_reaction_matrix(reaction_exp_matrix, gene_exp_scale, plot = False):
    df = reaction_exp_matrix
    nan_columns = df.columns[df.isna().any()].tolist()
    missing_gene_mask = column_indices_with_nan = [df.columns.get_loc(col) for col in nan_columns]

    reaction_exp_matrix = reaction_exp_matrix.fillna(0)
    from sklearn.preprocessing import normalize
    R = normalize(reaction_exp_matrix, norm="l1", axis=1, return_norm=False)
    R = pd.DataFrame(R, index = reaction_exp_matrix.index, columns = reaction_exp_matrix.columns)
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, gene_exp_scale))
    srm = scaler.fit_transform(R.T) #function doesn't have axis param
    srm = pd.DataFrame(srm.T, index = reaction_exp_matrix.index, columns = reaction_exp_matrix.columns)

    if plot:
        import matplotlib.pyplot as plt
        rand_mask = np.random.randint(0,50, 9)
        fig, axes = plt.subplots(1,2,  figsize =(12,3))
        for i, ax in enumerate(axes):
            data = [R, srm][i]
            title = [f'reaction exp matrix',f'scaled'][i]
            for i in rand_mask:
                ax.hist(data.iloc[:,i], bins= 50)
                ax.set_title(title)
    
    return R, srm


#add in reactions from model without gene association
#for max wt clique we use scaled values and fill NA with 0
#for FBA we use l1 norm values where $v_{i} \in [0,1$

def merge_tx_model(model, R, srm, plot = False):
    full_exp_mat = pd.DataFrame(index = list(model.reactions.keys()))
    
    R = pd.merge(full_exp_mat, R.T, 
        how = 'left', left_index=True, right_index=True).fillna(1).T
    
    srm = pd.merge(full_exp_mat, srm.T, 
        how = 'left', left_index=True, right_index=True).fillna(0).T
    

    return R, srm






############################################################ 
############################################################ storage univ graph class

class base_graph():
    def __init__(self, G, metabolic_model):
        from utils import sparse_mx_to_torch_sparse_tensor
        from utils import G2edgeindex
        from utils import to_sparse_mx
        import torch

        self.m_model = metabolic_model
        self.G = G
        self.N = len(G.nodes)
        I_n =sp.eye(self.N)
        self.I_n = sparse_mx_to_torch_sparse_tensor(I_n)
        self.edge_list = G2edgeindex(G)
        self.edge_index = torch.transpose(torch.tensor(self.edge_list,dtype=torch.long),0,1)
        adjmatrix = to_sparse_mx(self.edge_index, self.N)
        adjmatrix = sparse_mx_to_torch_sparse_tensor(adjmatrix)

        self.Fullm = torch.ones(self.I_n.size(0),self.I_n.size(1)) - self.I_n 
        self.W = adjmatrix + self.I_n
        self.W_complement = self.Fullm - self.W
    

    def get_universal_features(self):
        graphs = {}
        contg = nx.is_connected(self.G)
        _eccen = []
        _cluter = []

        if contg:
            for j in self.G.nodes:
                try:
                    _eccen += [nx.eccentricity(self.G,j)]
                    _cluter += [nx.clustering(self.G,j)]
                except:
                    print("{j}th is Wrong")
                    _eccen +=[0]
                    _cluter += [0]
            self._eccen = _eccen
            self._cluter = _cluter
        
        elif not contg:
            G_ = [self.G.subgraph(c).copy() for c in nx.connected_components(self.G)]
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

            ordered_nodes = list(np.arange(0,len(self.G.nodes)))
            indices_to_reorder = np.argsort([ordered_nodes.index(item) for item in node_vector])
            self._eccen = _eccen
            self._cluter = _cluter
            self.indices_to_reorder = indices_to_reorder

    def get_input_features(self, srm, R):
        self.R = R
        self.srm = srm
        self.total_samples = srm.shape[0]
        f = []
        self.get_universal_features()
        self.node_rxn_labels = {n:srm.columns[i] for i, n in enumerate(self.G.nodes)}
        contg = nx.is_connected(self.G)
        if contg:
            for s, samp in enumerate(srm.index):
                feature_vector = []
                g = self.G.copy()
                for j in g.nodes:
                    try:
                        _exp = srm.iloc[s, j]
                        neighbors = [e[1] for e in self.edge_list if e[0] == j]
                        neighbor_ids = [self.node_rxn_labels[n] for n in neighbors]
                        _deg = np.sum(srm[neighbor_ids].iloc[s,:])
                        _deg = np.abs(_deg)
                        _deg = np.sqrt(_deg)
                    except:
                        _exp = 0
                        _deg = 0
                    feature_vector.append([self._eccen[j],self._cluter[j], _exp, _deg])
                f += [feature_vector]
        elif not contg:
            G_ = [self.G.subgraph(c).copy() for c in nx.connected_components(self.G)]
            for s, samp in enumerate(srm.index):

                g = G_.copy()
                feature_vector = [] 
                for gsub in G_:
                    edge_list = G2edgeindex(gsub)
                    for j in gsub.nodes:
                        try:
                            _exp = np.abs(srm.iloc[s, j])
                            neighbors = [e[1] for e in edge_list if e[0] == j]
                            neighbor_ids = [self.node_rxn_labels[n] for n in neighbors]
                            _deg = np.sum(srm[neighbor_ids].iloc[s,:])
                            _deg = np.abs(_deg)
                            _deg = np.sqrt(_deg)
                        except:
                            _exp = 0
                            _deg = 0
                        feature_vector.append([self._eccen[j],self._cluter[j], _exp, _deg])
                feature_vector = [feature_vector[i] for i in self.indices_to_reorder]
                f += [feature_vector]
        self.f = f

    
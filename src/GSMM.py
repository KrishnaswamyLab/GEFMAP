from utils import *

from tqdm import tqdm
import libsbml
from six import string_types
import scipy

from tabulate import tabulate

models = {['ECOLI_core', 'mus_musculus', 'homo_sapiens'][i]:org for i, org in 
          enumerate(['e_coli_core_SBML3.xml', 'mus_iMM1415.xml', 'Recon3D.xml'])}

models_ = tabulate(list(models.items()), ['ORGANISM', 'MODEL_SBML'],tablefmt="grid")



##############################################################################
#metabolic models from sbml
##############################################################################

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
    def __init__(self, sbml_reaction, xml_params):
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
        
        if self.gene_associations is None:
            self.reaction_expression = None
        
        elif self.gene_associations is not None:
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

                self.reaction_expression = rexp
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
        gp = q.getGeneProduct().split('_')[1].upper()
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



##############################################################################
#incorporating gene expression data
##############################################################################
    

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
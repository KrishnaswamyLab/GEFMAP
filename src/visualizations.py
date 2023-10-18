from utils import *
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.colors as clr
from matplotlib.colors import LinearSegmentedColormap


def random_hex_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


'''
metadata_dict = {'sample_group': metadata['Condition ID'].astype('str'),
'carbon' : metadata['Carbon Source (g/L)'].astype('str'),
'nitrogen' : metadata['Nitrogen Source (g/L)'].astype('str'),
'electron_acc' : metadata['Electron Acceptor'].astype('str'),
'growth_rate' : metadata['Growth Rate (1/hr)']
}
'''


'''
fig, axes = plt.subplots(3,4, figsize = (9,7))
sub_samp =  np.random.randint(0, srm.shape[0], 6)
pos = nx.spring_layout(G)
cmap = 'rocket_r'
vmin = 0
vmax = gene_exp_scale

for i, ax in enumerate(axes.flatten()):
    ax.set_axis_off()
    #sns.despine()
    nx.draw_networkx_edges(G, pos=pos, alpha = 0.1, ax = ax)
    if i % 2 ==0:
        samp_curr = sub_samp[i//2]
        node_color = G_srm.iloc[samp_curr,:]
        mw_nd = mw_nodes[samp_curr]
        nx.draw_networkx_nodes(G, pos=pos, alpha = 1, node_size = 20, ax= ax, 
                       vmin = vmin, vmax = vmax,
                       node_color = node_color, cmap=cmap)
    else:
        node_color_sub = node_color[mw_nd]
        nx.draw_networkx_nodes(G, nodelist = mw_nd, pos=pos, alpha = 1, node_size = 20,  ax= ax, 
                       vmin = vmin, vmax = vmax,
                       node_color = node_color_sub, cmap=cmap)
'''


######### separate

def plot_metadata(y_feat, c_feat, metadata_dict, colors, ylabel = ''):
    empty = metadata_dict[c_feat].isna().astype('bool').sum()
    if empty > 20:
        print(f'{empty} NA values in {c_feat}')
    vars = np.unique(metadata_dict[c_feat].astype('str'))
    if len(vars) > len(colors):
        colors = random_colors = [random_hex_color() for _ in range(len(vars))]
    colors = colors[:len(vars)]
    c_dict = {v:colors[i] for i, v in enumerate(vars)}
    c_dict['none'] = '#FFFFFF'
    c = [c_dict[str(v)] for v in metadata_dict[c_feat]]
    ####
    ####
    fig, axes = plt.subplots(1,2, figsize = (10,4))
    for i, ax in enumerate(axes):
        title = ['biomass', 'max wt clique'][i]

        y=y_feat[i]
        x = np.arange(0, len(y))

        ax.scatter(x=x, y=y, c = c)
        ax.set_title(title)
        ax.set_xlabel('sample_id')
        ax.set_ylabel(ylabel)
    plt.suptitle(f'color_{str(c_feat)}')
    plt.show()


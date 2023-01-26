import time, argparse
import numpy as np
import pickle as pk
import pandas as pd
from scipy.optimize import linprog

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument('--data', type=str, default='e_coli')

args = parser.parse_args()

rng = np.random.default_rng(args.start)

if args.data == 'mouse':
    # load file
    f = pd.read_csv('data/mouse_S_matrix.csv')  
    S = f.iloc[:,1:].to_numpy()

    # target reaction
    c = np.zeros(S.shape[1])
    c[3722] = -1

    # reverse_reaction
    r = pd.read_csv('data/mouse_reversability.csv')
    reverse_reaction = r["0"].to_numpy()
    num_reverse = sum(reverse_reaction)
else:
    # load file
    f = pd.read_csv('data/e_coli_S_matrix.csv')  
    S = f.iloc[:,1:].to_numpy()

    # target reaction
    objs = pk.load(open('data/ecoli_clique_objfunctions.pkl', 'rb'))
    c = np.zeros((len(objs.keys()), S.shape[1]))
    for i, obj in enumerate(objs):
        c[i][objs[obj]] = -1

    # reverse_reaction
    r = pd.read_csv('data/ecoli_reversability.csv')
    reverse_reaction = r["0"].to_numpy() == False
    num_reverse = sum(reverse_reaction)

# simulation
for e in range(args.start, args.end):
    X = []
    bounds = []
    objs = []
    start_time = time.time()
    for _ in range(16):
        low = np.zeros(S.shape[1])
        up  = 10.0*np.ones(S.shape[1]) + rng.normal(0, 2, S.shape[1])
        low[reverse_reaction] = -10.0*np.ones(num_reverse) + rng.normal(0, 2, num_reverse)

        reverse = up < low
        tmp = up[reverse]
        up[reverse] = low[reverse]
        low[reverse]=tmp
        
        bound = np.c_[low, up]
        obj = c[rng.integers(low=0, high=len(c), size=1)[0]]
        res = linprog(obj, A_ub=None, b_ub=None, A_eq = S, b_eq = np.zeros(S.shape[0]),
                      bounds=bound)

        if res.success and res.status == 0:
            bounds.append(bound)
            X.append(res.x)
            objs.append(obj)
        
    X, bounds, objs = np.array(X), np.array(bounds), np.array(objs)
    print("Epoch %d --- %s seconds ---" % (e, time.time() - start_time))
    pk.dump({'X':X.astype('float32'),'bound':bounds.astype('float32'), 'objs':objs.astype('float32')}, 
            open('data_clique/'+args.data+'/'+str(e)+'.pk','wb'))


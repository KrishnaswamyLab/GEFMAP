# external files
import torch
import numpy as np
import pickle as pk
from numpy import inf
import torch.optim as optim
from datetime import datetime
import os, time, argparse, csv
import torch.nn.functional as F


from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import l1_loss, mse_loss
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

# internal files
from layer.FCN import FCN
from utils.save_settings import write_log

# select cuda device if available
cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="FCN")
    parser.add_argument('--log_root', type=str, default='../logs/', help='the path saving model.t7 and the training process')
    parser.add_argument('--log_path', type=str, default='test', help='the path saving model.t7 and the training process, the name of folder will be log/(current time)')
    parser.add_argument('--data_path', type=str, default='../data_clique/ecoli', help='data set folder, for default format see dataset/cora/cora.edges and cora.node_labels')
    parser.add_argument('--dataset', type=str, default='data', help='data set selection')

    parser.add_argument('--epochs', type=int, default=3000, help='Number of (maximal) training epochs.')
    parser.add_argument('--batch', type=int, default=32, help='batch size.')
    parser.add_argument('--method_name', type=str, default='FCN_3', help='method name')

    parser.add_argument('--layer', type=int, default=5, help='How many layers of gcn in the model, default 2 layers.')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout prob')

    parser.add_argument('--debug', '-D', action='store_true', help='debug mode')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--w', type=float, default=10, help='weights for loss from lb<x or x>ub')

    parser.add_argument('--num_filter', type=int, default=32, help='num of filters')
    parser.add_argument('--randomseed', type=int, default=3407, help='if set random seed in training')
    parser.add_argument('--target', type=int, default=-1, help='target reaction')
    return parser.parse_args()

class BaseLoader(Dataset):
    def __init__(self, X, Y):
        self.X = X.astype('float32')
        self.Y = Y.astype('float32')
    def __getitem__(self, item):
        return (self.X[item], self.Y[item])
    def __len__(self):
        return len(self.X)

def preproccess(path, seed = 50):
    dataset = pk.load(open(path, 'rb'))
    ns = dataset['ns'] # null space vectors
    X = dataset['X']
    Y = dataset['Y']
    S = dataset['S']

    S2m_pos = (S + np.abs(S))/2 #production
    S2m_neg = (S - np.abs(S))/2 

    X_train, X_test,Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=seed)
    X_train, X_val, Y_train, Y_val  = train_test_split(X_train, Y_train, shuffle=True, random_state=seed, test_size=0.2)
    return ns, BaseLoader(X_train, Y_train), BaseLoader(X_val, Y_val), BaseLoader(X_test, Y_test), S2m_pos, S2m_neg
    #return ns, BaseLoader(X_train, Y_train), BaseLoader(X_test, Y_test), S2m_pos, S2m_neg

def MFG(v, S2m_neg, S2m_pos):
    v2m_pos = (v + np.abs(v))/2
    v2m_neg = (v - np.abs(v))/2
    Jv = np.array([np.dot(S2m_pos, v2m_pos.T), np.dot(S2m_pos, v2m_neg.T)])
    Jv = 1.0/Jv
    Jv[Jv == inf] = 0.0

    Jv = torch.diag_embed(torch.FloatTensor(Jv).transpose(2,1)).numpy()
    v2m_pos = torch.diag_embed(torch.FloatTensor(v2m_pos)).numpy()
    v2m_neg = torch.diag_embed(torch.FloatTensor(v2m_neg)).numpy()
  
    L = np.array([np.matmul(S2m_pos, v2m_pos), np.matmul(S2m_pos, v2m_neg)])
    R = np.array([np.matmul(S2m_neg, v2m_pos), np.matmul(S2m_neg, v2m_neg)])

    M = np.matmul(L.transpose(0,1,3,2), np.matmul(Jv, R))
    M = M[0] + M[1]
    return np.abs(M)

def loss(preds, y):
    return l1_loss(preds, y)

def main(args):

    seed = 0
    if args.randomseed > 0:
        torch.manual_seed(args.randomseed)
        seed = args.randomseed

    date_time = datetime.now().strftime('%m-%d-%H:%M:%S')
    log_path = os.path.join(args.log_root, args.log_path, args.save_name, date_time)
    if os.path.isdir(log_path) == False:
        try:
            os.makedirs(log_path)
        except FileExistsError:
            print('Folder exists!')
            
    # load data
    ns, train, val, test, S2m_pos, S2m_neg = preproccess(args.data_path+'/'+args.dataset+'.pk', seed = seed)

    # torch data loader
    train_loader = DataLoader(train, num_workers=4, pin_memory = True, batch_size=args.batch, shuffle=True, drop_last=False)   
    val_loader = DataLoader(val, num_workers=4, pin_memory = True, batch_size=args.batch, shuffle=False, drop_last=True)
    test_loader = DataLoader(test, num_workers=4, pin_memory = True, batch_size=args.batch, shuffle=False, drop_last=False)   

    
    # loss function
    criterion = loss

    # initialize magnet (feature dim is 2)
    model = FCN(3, ns, label_dim=ns.shape[1], 
        layer = args.layer, num_filter = args.num_filter, dropout=args.dropout).to(device)    
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = ReduceLROnPlateau(opt, factor=0.5, patience=100)

    log_str_full = ''
    best_test_mse = 10000.0
    early_stopping = 0
    for epoch in range(args.epochs):
        loss_val, loss_train = 0.0, 0.0
        start_time = time.time()
        ####################
        # Train
        ####################
        count = 0
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            batch_size = x.shape[0]
            count += batch_size
            preds = model(x)
            # unsupervised loss
            train_loss = -torch.mean(preds*(x[:,-1]==-1)) + args.w*torch.mean(F.relu(x[:,0]-preds) + F.relu(preds-x[:,1])) #+ criterion(preds, y)
            # supervised loss + unsupervised loss
            #train_loss = -torch.mean(preds[:,args.target]) + 10.0*torch.mean( F.relu(x[:,0]-preds) + F.relu(preds-x[:,1]) ) + criterion(preds, y)
            # supervised loss
            #train_loss = criterion(preds, y)
            
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            loss_train += train_loss.item() * batch_size
        
        loss_train /= count
        outstrtrain = 'Train loss:, %.6f, ' % (loss_train)
        ####################
        # Val
        ####################
        count = 0
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                batch_size = x.shape[0]
                count += batch_size
                preds = model(x)
                val_loss = criterion(preds, y)
                loss_val += val_loss.item() * batch_size

            loss_val /= count
        outstrval = 'Val loss:, %.6f, ' % (loss_val)
        scheduler.step(loss_val)

        duration = "--- %.4f seconds ---" % (time.time() - start_time)
        log_str = ("%d, / ,%d, epoch," % (epoch, args.epochs))+outstrtrain+outstrval+duration+' lr: '+str(opt.param_groups[0]['lr'])
        log_str_full += log_str + '\n'
        print(log_str)

        ####################
        # Save weights
        ####################
        save_perform = loss_val
        if save_perform <= best_test_mse:
            early_stopping = 0
            best_test_mse = save_perform
            torch.save(model.state_dict(), log_path + '/model.t7')
        else:
            early_stopping += 1
        if early_stopping > 1000 or epoch == (args.epochs-1):
            torch.save(model.state_dict(), log_path + '/model_latest.t7')
            break

    ####################
    # Testing
    ####################
    results = {}
    results['pred']=[]
    results['label']=[]
    results['bound']=[]
    results['M_label'] = []
    results['M_pred'] = []
    model.load_state_dict(torch.load(log_path + '/model.t7'))
    model.eval()
    loss_test = 0.0
    count = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            batch_size = x.shape[0]
            count += batch_size
            preds = model(x)
            val_loss = criterion(preds, y)
            loss_test += val_loss.item() * batch_size
            results['pred'].append(preds.detach().cpu().numpy())
            results['label'].append(y.detach().cpu().numpy())
            results['bound'].append(x.detach().cpu().numpy())
            
            #M_label = MFG(y.detach().cpu().numpy(), S2m_neg, S2m_pos)
            #M_pred  = MFG(preds.detach().cpu().numpy(), S2m_neg, S2m_pos)

            #pk.dump({'M_label': M_label, 'M_pred': M_pred}, 
            #            open(args.method_name+str(i)+'.pk', 'wb'))
    
    results['pred'] = np.concatenate(results['pred'])
    results['label'] = np.concatenate(results['label'])
    #results['M_label'] = np.concatenate(results['M_label'])
    #results['M_pred'] = np.concatenate(results['M_pred'])
    results['bound'] = np.concatenate(results['bound'])

    loss_test /= count
    outstrtest = 'Test loss:, %.6f, ' % (loss_test)
    log_str_full += outstrtest
    print(outstrtest)
    with open(log_path + '/log.csv', 'w') as file:
        file.write(log_str_full)
        file.write('\n')

    # get MFG
    write_log(vars(args), log_path)
    return results

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        args.epochs = 1

    dir_name = os.path.join(os.path.dirname(os.path.realpath(
            __file__)), '../result_arrays',args.log_path,args.dataset+'/')
    args.log_path = os.path.join(args.log_path, args.method_name, args.dataset)

    if os.path.isdir(dir_name) == False:
        try:
            os.makedirs(dir_name)
        except FileExistsError:
            print('Folder exists!')

    save_name = (args.method_name + 'lr' + str(int(args.lr*1e4)) + 'num_filters' + str(int(args.num_filter)) 
                + 'layer' + str(int(args.layer)) + 'rs' + str(args.randomseed) + 'w'+str(args.w))
    args.save_name = save_name
    
    results = main(args)
    pk.dump(results, open(dir_name+save_name+'.pk', 'wb'))    

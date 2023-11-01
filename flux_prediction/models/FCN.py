import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.functional import mse_loss
import time
import torch.optim as optim
import numpy as np
class FCN(nn.Module):
    def __init__(self, in_c, out_c, nullspace, label_dim, kernel, num_filter = 2,  layer=2, dropout=False):
        super(FCN, self).__init__()

        self.layers = nn.ModuleList([nn.Conv1d(in_c, out_c, kernel), nn.ReLU(), nn.Flatten(),
                                        nn.Linear(nullspace.shape[0], num_filter, bias=True),
                                        #nn.BatchNorm1d(num_filter),
                                        nn.ReLU()])
        while layer > 3:
            self.layers.append(nn.Linear(num_filter, num_filter, bias=True))
            #self.layers.append(nn.BatchNorm1d(num_filter)),
            self.layers.append(nn.ReLU())
            layer -= 1

        self.Linear = nn.Linear(num_filter, label_dim, bias=True)
        self.dropout = dropout
        self.ns = nn.parameter.Parameter(torch.from_numpy(nullspace).T, requires_grad=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if self.dropout > 0:
                x = F.dropout(x, self.dropout, training=self.training)
        x = self.Linear(x)
        # print(type(x[0][0]))
        # print(type(self.ns))
        # y = torch.matmul(x, self.ns.float())
        return x

# select cuda device if available
cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")
training_losses = []


def loss(preds, y):
    return mse_loss(preds, y)



def train(model,train_data, val_data, optimizer, scheduler, epoch, epochs, log_str_full = ''):
    loss_val, loss_train = 0.0, 0.0
    start_time = time.time()
    ####################
    # Train
    ####################
    count = 0
    model.train()
    for x, y in train_data:
        x, y = x.to(device), y.to(device)
        batch_size = x.shape[0]
        count += batch_size
        preds = model(x)
        # unsupervised loss
        # train_loss = -torch.mean(preds*(x[:,-1]==-1)) + w*torch.mean(F.relu(x[:,0]-preds) + F.relu(preds-x[:,1])) #+ criterion(preds, y)
        # supervised loss + unsupervised loss
        #train_loss = -torch.mean(preds[:,target]) + 10.0*torch.mean( F.relu(x[:,0]-preds) + F.relu(preds-x[:,1]) ) + criterion(preds, y)
        #supervised loss
        preds = preds.unsqueeze(2)
        # print(preds)
        # print(y.shape)
        train_loss = loss(preds, y)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        loss_train += train_loss.item() * batch_size

    loss_train /= count
    outstrtrain = 'Train loss:, %.6f, ' % (loss_train)
    ####################
    # Val
    ####################
    count = 0
    model.eval()
    with torch.no_grad():
        for x, y in val_data:
            x, y = x.to(device), y.to(device)
            batch_size = x.shape[0]
            count += batch_size
            preds = model(x)
            preds = preds.unsqueeze(2)
            val_loss = loss(preds, y)
            loss_val += val_loss.item() * batch_size

        loss_val /= count
    outstrval = 'Val loss:, %.6f, ' % (loss_val)
    scheduler.step(loss_val)

    ## save losses
    training_losses.append(loss_val)

    duration = "--- %.4f seconds ---" % (time.time() - start_time)
    log_str = ("%d, / ,%d, epoch," % (epoch, epochs))+outstrtrain+outstrval+duration+' lr: '+str(optimizer.param_groups[0]['lr'])
    log_str_full += log_str + '\n'
    return log_str
    # print(log_str)

    ####################
    # Save weights
    ####################
    # save_perform = loss_val
    # if save_perform <= best_test_mse:
    #     early_stopping = 0
    #     best_test_mse = save_perform
    #     # torch.save(model.state_dict(), log_path + '/model.t7')
    # else:
    #     early_stopping += 1
    # if early_stopping > 1000 or epoch == (epochs-1):
    #     # torch.save(model.state_dict(), log_path + '/model_latest.t7')
    #     break



def test(model, test_data, log_str_full = ''):
    model.eval()
    results = {}
    results['pred']=[]
    results['label']=[]
    results['bound']=[]
    results['M_label'] = []
    results['M_pred'] = []
    loss_test = 0.0
    count = 0
    sum_y = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_data):
            x, y = x.to(device), y.to(device)
            batch_size = x.shape[0]
            count += batch_size
            preds = model(x)
            preds = preds.unsqueeze(2)
            val_loss = loss(preds, y)
            # print(i)
            # print(torch.sum(y))
            sum_y += torch.sum(y)
            loss_test += val_loss.item() * batch_size
            results['pred'].append(preds.detach().cpu().numpy())
            results['label'].append(y.detach().cpu().numpy())
            results['bound'].append(x.detach().cpu().numpy())

            #M_label = MFG(y.detach().cpu().numpy(), S2m_neg, S2m_pos)
            #M_pred  = MFG(preds.detach().cpu().numpy(), S2m_neg, S2m_pos)

            #pk.dump({'M_label': M_label, 'M_pred': M_pred},
            #            open(method_name+str(i)+'.pk', 'wb'))

    results['pred'] = np.concatenate(results['pred'])
    results['label'] = np.concatenate(results['label'])
    #results['M_label'] = np.concatenate(results['M_label'])
    #results['M_pred'] = np.concatenate(results['M_pred'])
    results['bound'] = np.concatenate(results['bound'])
    # print(loss_test)
    # print(sum_y)
    loss_test /= sum_y
    outstrtest = 'Test loss:, %.6f, ' % (loss_test)
    log_str_full += outstrtest
    # print(outstrtest)
    return outstrtest, results['pred'].flatten(), results['label'].flatten()
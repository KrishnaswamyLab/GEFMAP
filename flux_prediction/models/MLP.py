import torch
import torch.nn as nn
from torch.nn import ReLU
from torch.nn.functional import mse_loss
import numpy as np
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 16):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.act_fn = ReLU()
    
    def forward(self,x):
        x = self.input_layer(x)
        x = self.act_fn(x)
        x = self.hidden_layer(x)
        x = self.act_fn(x)
        x = self.output_layer(x)
        return x


# select cuda device if available
cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")
training_losses = []

def loss(preds, y):
    return mse_loss(preds, y)


def train_MLP(model, train_data, val_data, optimizer, epoch, epochs, log_str_full = ''):
    loss_val, loss_train = 0.0, 0.0
    count = 0
    model.train()
    for x,y in train_data:
        # x, y = x.to(device), y.to(device)
        batch_size = x.shape[0]
        count += batch_size
        preds = model(x)
        train_loss = loss(preds, y)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        loss_train += train_loss.item() * batch_size
    
    loss_train /= count
    outstrtrain = 'Train loss:, %.6f, ' % (loss_train)

    count = 0
    model.eval()
    with torch.no_grad():
        for x, y in val_data:
            # x, y = x.to(device), y.to(device)
            batch_size = x.shape[0]
            count += batch_size
            preds = model(x)
            # preds = preds.unsqueeze(2)
            val_loss = loss(preds, y)
            loss_val += val_loss.item() * batch_size
    
        loss_val /= count
    outstrval = 'Val loss:, %.6f, ' % (loss_val)
    ## save losses
    training_losses.append(loss_val)

    log_str = ("%d, / ,%d, epoch," % (epoch, epochs))+outstrtrain+outstrval+' lr: '+str(optimizer.param_groups[0]['lr'])
    log_str_full += log_str + '\n'
    return log_str

def test_MLP(model, test_data, log_str_full = ''):
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
            # x, y = x.to(device), y.to(device)
            batch_size = x.shape[0]
            count += batch_size
            preds = model(x)
            # preds = preds.unsqueeze(2)
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
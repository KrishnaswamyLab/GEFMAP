import torch, math
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, in_c, nullspace, num_filter=2, label_dim=1, layer=2, dropout=False):
        super(FCN, self).__init__()

        self.layers = nn.ModuleList([nn.Conv1d(in_c, 1, 1), nn.ReLU(), nn.Flatten(),
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
        y = torch.matmul(x, self.ns)
        return y

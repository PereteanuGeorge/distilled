import torch
import torch.nn as nn
import torch.nn.functional as F


class Attack(nn.Module):
    def __init__(self, hidden_size):
        super(Attack, self).__init__()
        #self.linear1 = nn.Linear(hidden_size, 512)
        self.linear2 = nn.Linear(hidden_size, 256)
        self.linear3 = nn.Linear(256, 64)
        self.linear4 = nn.Linear(64, 10)


    def forward(self, x):
        #out = self.linear1(x)
        #out = F.relu(out)
        out = self.linear2(x)
        #out = F.dropout(out, 0.1)
        out = self.linear3(out)
        #out = F.relu(out)
        out = self.linear4(out)
        #out = F.relu(out)
        #out = self.linear5(out)
        #out = self.linear6(out)
        #out = self.linear7(out)
        #out = F.relu(out)
        #out = self.linear8(out)
        return out

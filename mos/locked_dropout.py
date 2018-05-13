import torch
import torch.nn as nn
# from torch.autograd import Variable

class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = torch.tensor(x.detach(), requires_grad=False).resize_(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        # mask = Variable(m, requires_grad=False) / (1 - dropout)
        # mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = m.div_(1 - dropout)
        mask = mask.expand_as(x)
        return mask * x

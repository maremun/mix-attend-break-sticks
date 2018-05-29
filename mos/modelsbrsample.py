import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MoShead(nn.Module):
    """MoShead functionality as a standalone module."""
    def __init__(self, ntoken, ninp, nhid, nhidlast, encoder, lockdrop, tie_weights=False, n_experts=10):
        super(MoShead, self).__init__()
        self.ntoken = ntoken
        self.ninp = ninp
        self.nhidlast = nhidlast
        self.n_experts = n_experts

        self.lockdrop = lockdrop
        #self.prior = nn.Linear(nhidlast, n_experts, bias=False)
        
        self.reduce = nn.Sequential(nn.Linear(nhidlast, n_experts), nn.Softplus())
        self.sigmoid = nn.Sigmoid()
        
        self.latent = nn.Sequential(nn.Linear(nhidlast, n_experts*ninp), nn.Tanh())
        self.decoder = nn.Linear(ninp, ntoken)
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = encoder.weight


    def forward(self, output, dropoutl):
        # output: [seq_len x batch_size x nhidlast]
        latent = self.latent(output)  # h
        latent = self.lockdrop(latent, dropoutl)  # h after variational dropout [seq_len x batch_size x n_experts * ninp]
        logit = self.decoder(latent.view(-1, self.ninp))  # HW [seq_len * batch_size * n_experts x voc_size]

        a = self.reduce(output.view(-1, self.nhidlast)) + 1e-8 # [seq_len * batch_size x n_experts]
        b = torch.sum(a, 1)
        c = torch.cumsum(a, 1)
        d = torch.abs(b[:, None] - c)
        a = a.to('cpu')
        d = d.to('cpu')
        beta = torch.distributions.Beta(a, d)
        sample = beta.rsample()  # [seq_len * batch_size x n_experts]
        sample = sample.to('cuda')
        rem = 1 - sample
        D = torch.diag(torch.ones(self.n_experts - 1, device=device), 1)
        rem = rem @ D
        rem[:, 0] = 1
        remprod = torch.cumprod(rem, 1)
        pis = remprod * sample
        prob = nn.functional.softmax(logit.view(-1, self.ntoken), dim=1).view(-1, self.n_experts, self.ntoken)  # exp(hw) / sum(exp(hw))
        prob = (prob * pis.unsqueeze(2).expand_as(prob)).sum(1)  # weighted sum
        # TODO maybe we can do this with logsoftmax
        return prob


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nhidlast, nlayers,
                 dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0,
                 tie_weights=False, ldropout=0.5, n_experts=10):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.encoder = nn.Embedding(ntoken, ninp)

        self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else nhidlast, 1, dropout=0) for l in range(nlayers)]
        if wdrop:
            self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)

        self.head = MoShead(ntoken, ninp, nhid, nhidlast, self.encoder, self.lockdrop, tie_weights, n_experts)

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nhidlast = nhidlast
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.ldropout = ldropout
        self.dropoutl = ldropout
        self.n_experts = n_experts
        self.ntoken = ntoken

        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('Param size: {}'.format(size))

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.head.decoder.bias.data.fill_(0)
        self.head.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False, return_prob=False):
        batch_size = input.size(1)
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        #emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        prob = self.head(output, self.dropoutl)

        if return_prob:
            model_output = prob
        else:
            log_prob = torch.log(prob.add_(1e-8))
            model_output = log_prob

        model_output = model_output.view(-1, batch_size, self.ntoken)

        if return_h:
            return model_output, hidden, raw_outputs, outputs
        return model_output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        # return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.nhidlast).zero_(),
        #          weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.nhidlast).zero_())
        #         for l in range(self.nlayers)]
        if torch.cuda.is_available():
            return [(torch.zeros((1, bsz, self.nhid if l != self.nlayers - 1 else self.nhidlast), requires_grad=True).cuda(),
                 torch.zeros((1, bsz, self.nhid if l != self.nlayers - 1 else self.nhidlast), requires_grad=True).cuda())
                for l in range(self.nlayers)]
        else:
            return [(torch.zeros((1, bsz, self.nhid if l != self.nlayers - 1 else self.nhidlast), requires_grad=True),
                 torch.zeros((1, bsz, self.nhid if l != self.nlayers - 1 else self.nhidlast), requires_grad=True))
                for l in range(self.nlayers)]

if __name__ == '__main__':
    model = RNNModel('LSTM', 10, 12, 12, 12, 2)
    # input = torch.LongTensor(13, 9, requires_grad=True).random_(0, 10)
    input = torch.zeros((5, 9), dtype=torch.long).random_(0, 10)
    input.requires_grad_()
    hidden = model.init_hidden(9)
    print(model(input, hidden)[0])

    # input = Variable(torch.LongTensor(13, 9).random_(0, 10))
    # hidden = model.init_hidden(9)
    # print(model.sample(input, hidden, 5, 6, 1, 2, sample_latent=True).size())
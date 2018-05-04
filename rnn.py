import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as pad

import const


class rnn_model(nn.Module):
    def __init__(self, rnn_type, voc_size, emb_dim, hid_dim, n_layers, dropout=0.3, tie_weights=False):
        super(rnn_model, self).__init__()
        
        self.rnn_type = rnn_type
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        
        self.dropout = nn.Dropout(dropout)
        self.emb = nn.Embedding(voc_size, emb_dim, padding_idx=const.PAD)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(emb_dim, hid_dim, n_layers, batch_first=True, dropout=dropout)
        self.output = nn.Linear(hid_dim, voc_size)

        if tie_weights:
            if emb_dim != hid_dim:
                raise ValueError('When using the tied flag, \
                                 hidden dim must be equal \
                                 to embedding size')
            self.emb.weight = self.output.weight

        self.init_weights()
        
    def init_weights(self):
        i = 0.1
        self.emb.weight.data.uniform_(-i, i)
        self.output.bias.data.zero_()
        self.output.weight.data.uniform_(-i, i)
        
    def forward(self, inputs, lengths):
        lengths = lengths.contiguous().data.view(-1).tolist()
        embs = self.dropout(self.emb(inputs))
        packed_embs = pack(embs, lengths, batch_first=True)
        output, _ = self.rnn(packed_embs)
        output = pad(output, batch_first=True)[0]
        output = self.dropout(output)
        output_flat = self.output(output.contiguous().view(
                        output.size(0) * output.size(1), 
                        output.size(2)))
        
        return output_flat 
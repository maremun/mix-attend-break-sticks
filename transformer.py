import torch
import torch.nn as nn

from modules import Linear, PosEncoding
from layers import DecoderLayer

import const

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def proj_prob_simplex(inputs):
    # project updated weights onto a probability simplex
    # see https://arxiv.org/pdf/1101.6081.pdf
    sorted_inputs, sorted_idx = torch.sort(inputs.view(-1), descending=True)
    dim = len(sorted_inputs)
    for i in reversed(range(dim)):
        t = (sorted_inputs[:i+1].sum() - 1) / (i+1)
        if sorted_inputs[i] > t:
            break
    return torch.clamp(inputs-t, min=0.0)


def get_attn_pad_mask(seq_q, seq_k):
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    b_size, len_q = seq_q.size()
    b_size, len_k = seq_k.size()
    #pad_attn_mask = seq_k.eq(const.PAD).unsqueeze(1)  # b_size x 1 x len_k
    pad_attn_mask = seq_k.eq(-1).unsqueeze(1)  # b_size x 1 x len_k
    pad_attn_mask = pad_attn_mask.expand(b_size, len_q, len_k) # b_size x len_q x len_k
    return pad_attn_mask


def get_attn_subsequent_mask(seq):
    assert seq.dim() == 2
    attn_shape = [seq.size(1), seq.size(1)]
    subsequent_mask = torch.triu(torch.ones(attn_shape, device=device), diagonal=1).byte()
    return subsequent_mask


class Decoder(nn.Module):
    def __init__(self, n_layers, d_k, d_v, d_model, d_ff, n_heads,
                 max_seq_len, tgt_vocab_size, dropout=0.1, weighted=False):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model) #, padding_idx=const.PAD)
        self.pos_emb = PosEncoding(max_seq_len * 10, d_model) # TODO: *10 fix
        self.dropout_emb = nn.Dropout(dropout)
        self.layer_type = DecoderLayer if not weighted else WeightedDecoderLayer
        self.layers = nn.ModuleList(
            [self.layer_type(d_k, d_v, d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])

        
    def forward(self, dec_inputs, dec_inputs_len, enc_inputs, enc_outputs, return_attn=False):
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs += self.pos_emb(dec_inputs_len)
        dec_outputs = self.dropout_emb(dec_outputs)

        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)

        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        if enc_inputs is not None:
            dec_enc_attn_pad_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
        else:
            dec_enc_attn_pad_mask = None

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs,
                                                             self_attn_mask=dec_self_attn_mask,
                                                             enc_attn_mask=dec_enc_attn_pad_mask)
            if return_attn:
                dec_self_attns.append(dec_self_attn)
                dec_enc_attns.append(dec_enc_attn)

        return dec_outputs, dec_self_attns, dec_enc_attns


class LMTransformer(nn.Module):
    def __init__(self, n_layers, d_k, d_v, d_model, d_ff, n_heads,
                 max_tgt_seq_len, tgt_vocab_size, dropout,
                 weighted_model, share_proj_weight):
        super(LMTransformer, self).__init__()
        self.decoder = Decoder(n_layers, d_k, d_v, d_model, d_ff, n_heads,
                               max_tgt_seq_len, tgt_vocab_size, dropout, weighted_model)
        self.tgt_proj = Linear(d_model, tgt_vocab_size, bias=False)
        self.weighted_model = weighted_model

        if share_proj_weight:
            print('Sharing target embedding and projection..')
            self.tgt_proj.weight = self.decoder.tgt_emb.weight


    def trainable_params(self):
        # Avoid updating the position encoding
        params = filter(lambda p: p[1].requires_grad, self.named_parameters())
        # Add a separate parameter group for the weighted_model
        param_groups = []
        base_params = {'params': [], 'type': 'base'}
        weighted_params = {'params': [], 'type': 'weighted'}
        for name, param in params:
            if 'w_kp' in name or 'w_a' in name:
                weighted_params['params'].append(param)
            else:
                base_params['params'].append(param)
        param_groups.append(base_params)
        param_groups.append(weighted_params)
        return param_groups

    
    def decode(self, dec_inputs, dec_inputs_len, enc_inputs, enc_outputs, return_attn=False):
        return self.decoder(dec_inputs, dec_inputs_len, enc_inputs, enc_outputs, return_attn)

    
    def forward(self, dec_inputs, dec_inputs_len, return_attn=False):
        dec_outputs, dec_self_attns, _ = \
            self.decoder(dec_inputs, dec_inputs_len, None, None, return_attn)
        dec_logits = self.tgt_proj(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), dec_self_attns

    
    def proj_grad(self):
        if self.weighted_model:
            for name, param in self.named_parameters():
                if 'w_kp' in name or 'w_a' in name:
                    param.data = proj_prob_simplex(param.data)
        else:
            pass

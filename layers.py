import torch
import torch.nn as nn

from sublayers import MultiHeadAttention, \
                      MultiBranchAttention, \
                      PoswiseFeedForwardNet


class DecoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.dec_enc_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)

    def forward(self, dec_inputs, enc_outputs, self_attn_mask, enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs,
                                                        dec_inputs, attn_mask=self_attn_mask)
        if enc_outputs is not None:  # We don't need enc_outputs in LM
            dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs,
                                                          enc_outputs, attn_mask=enc_attn_mask)
        else:
            dec_enc_attn = None
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn
    
    
class WeightedDecoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_branches, dropout=0.1):
        super(WeightedDecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_k, d_v, d_model, n_branches, dropout)
        self.dec_enc_attn = MultiBranchAttention(d_k, d_v, d_model, d_ff, n_branches, dropout)

    def forward(self, dec_inputs, enc_outputs, self_attn_mask, enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs,
                                                        dec_inputs, attn_mask=self_attn_mask)
        if enc_outputs is not None:  # We don't need enc_outputs in LM
            dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs,
                                                          enc_outputs, attn_mask=enc_attn_mask)
        else:
            dec_enc_attn = None
        return dec_outputs, dec_self_attn, dec_enc_attn
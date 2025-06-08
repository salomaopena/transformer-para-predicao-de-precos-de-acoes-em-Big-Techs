import torch.nn as nn

from app.src.transformer.multi_head_attention import MultiHeadAttention
from app.src.transformer.position_wise_feed_forward import PositionWiseFeedForward

class EncoderBlock(nn.Module):
    def __init__(self, n_dim: int, dropout: float, n_heads: int):
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(hidden_dim=n_dim, num_heads=n_heads)
        self.norm1 = nn.LayerNorm(n_dim)
        self.ff = PositionWiseFeedForward(n_dim, n_dim)
        self.norm2 = nn.LayerNorm(n_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, src_padding_mask=None):
        assert x.ndim==3, "Expected input to be 3-dim, got {}".format(x.ndim)
        att_output = self.mha(x, x, x, key_padding_mask=src_padding_mask)
        x = x + self.dropout(self.norm1(att_output))
        
        ff_output = self.ff(x)
        output = x + self.norm2(ff_output)
       
        return output
import torch.nn as nn

from app.src.transformer.multi_head_attention import MultiHeadAttention
from app.src.transformer.position_wise_feed_forward import PositionWiseFeedForward

class DecoderBlock(nn.Module):
    def __init__(self, n_dim: int, dropout: float, n_heads: int):
        super(DecoderBlock, self).__init__()
        
        # The first Multi-Head Attention has a mask to avoid looking at the future
        self.self_attention = MultiHeadAttention(hidden_dim=n_dim, num_heads=n_heads)
        self.norm1 = nn.LayerNorm(n_dim)
        
        # The second Multi-Head Attention will take inputs from the encoder as key/value inputs
        self.cross_attention = MultiHeadAttention(hidden_dim=n_dim, num_heads=n_heads)
        self.norm2 = nn.LayerNorm(n_dim)
        
        self.ff = PositionWiseFeedForward(n_dim, n_dim)
        self.norm3 = nn.LayerNorm(n_dim)
        # self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, tgt, memory, tgt_mask=None, tgt_padding_mask=None, memory_padding_mask=None):
        
        masked_att_output = self.self_attention(
            q=tgt, k=tgt, v=tgt, attention_mask=tgt_mask, key_padding_mask=tgt_padding_mask)
        x1 = tgt + self.norm1(masked_att_output)
        
        cross_att_output = self.cross_attention(
            q=x1, k=memory, v=memory, attention_mask=None, key_padding_mask=memory_padding_mask)
        x2 = x1 + self.norm2(cross_att_output)
        
        ff_output = self.ff(x2)
        output = x2 + self.norm3(ff_output)

        
        return output
import torch.nn as nn

from transformer_processor.multi_head_attention import MultiHeadAttention
from transformer_processor.position_wise_feed_forward import PositionWiseFeedForward

class EncoderBlock(nn.Module):
    def __init__(self, headDimension: int, numberHeads: int, dropoutProbability: float):
        super(EncoderBlock, self).__init__()

        self.multiHeadAttention = MultiHeadAttention(headDimension=headDimension, numberHeads=numberHeads)

        self.normalization1 = nn.LayerNorm(headDimension)
        self.positionWiseFeedForward = PositionWiseFeedForward(headDimension, headDimension)
        self.normalization2 = nn.LayerNorm(headDimension)
        self.dropout = nn.Dropout(dropoutProbability)
        
    def forward(self, x, paddingMask=None):
        print(f"Input shape: {x.shape}")
        assert x.ndim == 3, f"Expected input to be 3-dim [batch, seq_len, dim], got {x.ndim}-dim"

        attentionOutput = self.multiHeadAttention(x, x, x, keyPaddingMask=paddingMask)
        x = x + self.dropout(self.normalization1(attentionOutput))
        
        feedForwardOutput = self.positionWiseFeedForward(x)
        output = x + self.dropout(self.normalization2(feedForwardOutput))
       
        return output

import torch.nn as nn

from transformer.multi_head_attention import MultiHeadAttention
from transformer.position_wise_feed_forward import PositionWiseFeedForward

class EncoderBlock(nn.Module):
    def __init__(self, headDimension: int, numberHeads: int, dropout: float):
        super(EncoderBlock, self).__init__()

        self.multiHeadAttention = MultiHeadAttention(headDimension=headDimension, numberHeads=numberHeads)
        print(f"MultiHeadAttention: {self.multiHeadAttention}")

        self.normalization1 = nn.LayerNorm(headDimension)
        print(f"Normalization1 Weight: {self.normalization1.weight}")
        print(f"Normalization1 Bias: {self.normalization1.bias}")
        print(f"Normalization1 Shape: {self.normalization1.weight.shape}")

        self.positionWiseFeedForward = PositionWiseFeedForward(headDimension, headDimension)
        print(f"PositionWiseFeedForward: {self.positionWiseFeedForward}")

        self.normalization2 = nn.LayerNorm(headDimension)
        print(f"Normalization2 Weight: {self.normalization2.weight}")
        print(f"Normalization2 Bias: {self.normalization2.bias}")
        print(f"Normalization2 Shape: {self.normalization2.weight.shape}")

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, paddingMask=None):
        assert x.headDimension==3, "Expected input to be 3-dim, got {}".format(x.headDimension)
        attetionOutput = self.multiHeadAttention(x, x, x, key_padding_mask=paddingMask)
        x = x + self.dropout(self.normalization1(attetionOutput))
        
        feedForwardOutput = self.positionWiseFeedForward(x)
        output = x + self.normalization2(feedForwardOutput)
       
        return output
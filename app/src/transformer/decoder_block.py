import torch.nn as nn

from app.src.transformer.multi_head_attention import MultiHeadAttention
from app.src.transformer.position_wise_feed_forward import PositionWiseFeedForward

class DecoderBlock(nn.Module):
    def __init__(self, headDimension: int, numberHeads: int, dropout: float):
        super(DecoderBlock, self).__init__()
        
        # The first Multi-Head Attention has a mask to avoid looking at the future
        self.selfAttention = MultiHeadAttention(headDimension=headDimension, numberHeads=numberHeads)
        self.normalization1 = nn.LayerNorm(headDimension)
        # print(f"Normalization1 Weight: {self.normalization1.weight}")
        # print(f"Normalization1 Bias: {self.normalization1.bias}")
        # print(f"Normalization1 Shape: {self.normalization1.weight.shape}")
        
        # The second Multi-Head Attention will take inputs from the encoder as key/value inputs
        self.crossAttention = MultiHeadAttention(headDimension=headDimension, nummberHeads=numberHeads)
        self.normalization2 = nn.LayerNorm(headDimension)
        # print(f"Normalization2 Weight: {self.normalization2.weight}")
        # print(f"Normalization2 Bias: {self.normalization2.bias}")
        # print(f"Normalization2 Shape: {self.normalization2.weight.shape}")
        
        self.positionWiseFeedForward = PositionWiseFeedForward(headDimension, headDimension)
        # print(f"PositionWiseFeedForward: {self.positionWiseFeedForward}")

        self.normalization3 = nn.LayerNorm(headDimension)
        # print(f"Normalization3 Weight: {self.normalization3.weight}")
        # print(f"Normalization3 Bias: {self.normalization3.bias}")
        # print(f"Normalization3 Shape: {self.normalization3.weight.shape}")
        
    def forward(self, target, memory, targetMask=None, targetPaddingMask=None, memoryPaddingMask=None):
        
        maskedAttetionOutput = self.selfAttention(
            q=target, k=target, v=target, attention_mask=targetMask, key_padding_mask=targetPaddingMask)
        x1 = target + self.normalization1(maskedAttetionOutput)
        
        crossAttetionOutput = self.crossAttention(
            q=x1, k=memory, v=memory, attention_mask=None, key_padding_mask=memoryPaddingMask)
        x2 = x1 + self.normalization2(crossAttetionOutput)
        
        feedForwardOutput = self.positionWiseFeedForward(x2)
        output = x2 + self.normalization3(feedForwardOutput)

        return output
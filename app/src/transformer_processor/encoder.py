import torch.nn as nn
import math

from .positional_enconding import PositionalEncoding
from .encoder_block import EncoderBlock

class Encoder(nn.Module):
    def __init__(
        self, 
        headDimension: int, 
        numberHeads: int,
        dropout: float, 
        numberEncoderBlocks: int
    ):
        super(Encoder, self).__init__()

        self.headDimension = headDimension

        self.linear = nn.Linear(
            in_features=1, 
            out_features=headDimension
        )

        self.positionalEncoding = PositionalEncoding(
            model=headDimension, 
            dropoutProbability=dropout
        )

        self.encoderBlocks = nn.ModuleList([
            EncoderBlock(headDimension, dropout, numberHeads) for _ in range(numberEncoderBlocks)
        ])
        
    def forward(self, x, paddingMask=None):
        """
        x: Tensor of shape [batch_size, seq_len] with float values (ex: stock prices)
        paddingMask: Optional mask tensor of shape [batch_size, seq_len]
        """
        x = x.unsqueeze(-1)  # [B, S] -> [B, S, 1]
        x = self.linear(x) * math.sqrt(self.headDimension)  # [B, S, E]
        x = self.positionalEncoding(x)

        for block in self.encoderBlocks:
            x = block(x=x, paddingMask=paddingMask)

        return x  # shape: [B, S, E]
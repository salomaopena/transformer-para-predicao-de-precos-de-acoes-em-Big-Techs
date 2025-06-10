import torch.nn as nn
import math

from .positional_enconding import PositionalEncoding
from .decoder_block import DecoderBlock

class Decoder(nn.Module):
    def __init__(
        self, 
        headDimension: int, 
        numberHeads: int,
        dropout: float, 
        numberDecoderBlocks: int
    ):
        super(Decoder, self).__init__()

        self.headDimension = headDimension

        self.linear = nn.Linear(
            in_features=1, 
            out_features=headDimension
        )

        self.positionalEncoding = PositionalEncoding(
            model=headDimension, 
            dropoutProbability=dropout
        )
          
        self.decoderBlocks = nn.ModuleList([
            DecoderBlock(headDimension, dropout, numberHeads) for _ in range(numberDecoderBlocks)
        ])
        
    def forward(self, target, memory, targetMask=None, targetPaddingMask=None, memoryPaddingMask=None):
        # Assume que target é (batch_size, seq_len) com floats → precisa virar (batch_size, seq_len, 1)
        x = target.unsqueeze(-1)
        x = self.linear(x) * math.sqrt(self.headDimension)
        x = self.positionalEncoding(x)

        for block in self.decoderBlocks:
            x = block(
                x, 
                memory, 
                targetMask=targetMask, 
                targetPaddingMask=targetPaddingMask, 
                memoryPaddingMask=memoryPaddingMask
            )
        return x

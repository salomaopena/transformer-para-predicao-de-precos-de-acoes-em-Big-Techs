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
        print(f"Input shape before linear: {x.shape}")
        
        # # Verifica se está faltando a última dimensão (precisa ser [S, B, 1])
        # if x.ndim == 2:
        #     x = x.unsqueeze(-1)

        # x = self.linear(x) * math.sqrt(self.headDimension)  # Espera (S, B, 1) → (S, B, headDimension)
        print(f"Self input shape: {x.shape}")
        
        x = self.positionalEncoding(x)
        
        for block in self.encoderBlocks:
            x = block(x=x, paddingMask=paddingMask)

        return x
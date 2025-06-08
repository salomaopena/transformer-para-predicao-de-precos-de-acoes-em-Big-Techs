import torch.nn as nn
import math

from positional_enconding import PositionalEncoding
from encoder_block import EncoderBlock


class Encoder(nn.Module):
    def __init__(
            self, 
            vocabularySize: int, 
            headDimension: int, 
            numberHeads: int,
            dropout: float, 
            numberEncoderBlocks: int):
        
        super(Encoder, self).__init__()

        self.headDimension = headDimension

        self.embedding = nn.Embedding(
            num_embeddings=vocabularySize, 
            embedding_dim=headDimension
        )
        print(f"Embedding Weight: {self.embedding.weight}")

        self.positionalEncoding = PositionalEncoding(
            model=headDimension, 
            dropout=dropout
        )    
        
        self.encoderBlocks = nn.ModuleList([
            EncoderBlock(headDimension, dropout, numberHeads) for _ in range(numberEncoderBlocks)
        ])
        
    def forward(self, x, paddingMask=None):
        x = self.embedding(x) * math.sqrt(self.headDimension)
        x = self.positionalEncoding(x)
        for enconderBlock in self.encoderBlocks:
            x = enconderBlock(x=x, src_padding_mask=paddingMask)
        return x
import torch.nn as nn

from positional_enconding import PositionalEncoding
from decoder_block import DecoderBlock

class Decoder(nn.Module):
    def __init__(
        self, 
        vocabularySize: int, 
        headDimension: int, 
        numberHeads: int,
        dropout: float, 
        numberDecoderBlocks: int):
        
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=vocabularySize, 
            embedding_dim=headDimension,
            padding_idx=0
        )
        self.positionalEncoding = PositionalEncoding(
            model=headDimension, 
            dropout=dropout
        )
          
        self.decoderBlocks = nn.ModuleList([
            DecoderBlock(headDimension, dropout, numberHeads) for _ in range(numberDecoderBlocks)
        ])
        
    def forward(self, target, memory, targetMask=None, targetPaddingMask=None, memoryPaddingMask=None):
        x = self.embedding(target)
        x = self.positionalEncoding(x)

        for block in self.decoderBlocks:
            x = block(
                x, 
                memory, 
                targetMask=targetMask, 
                targetPaddingMask=targetPaddingMask, 
                memoryPaddingMask=memoryPaddingMask)
        return x
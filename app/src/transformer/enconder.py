import torch.nn as nn
import math

from app.src.transformer.positional_enconding import PositionalEncoding
from app.src.transformer.encoderBlock import EncoderBlock


class Encoder(nn.Module):
    def __init__(
            self, 
            vocab_size: int, 
            n_dim: int, 
            dropout: float, 
            n_encoder_blocks: int,
            n_heads: int):
        
        super(Encoder, self).__init__()
        self.n_dim = n_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=n_dim
        )
        self.positional_encoding = PositionalEncoding(
            d_model=n_dim, 
            dropout=dropout
        )    
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(n_dim, dropout, n_heads) for _ in range(n_encoder_blocks)
        ])
        
        
    def forward(self, x, padding_mask=None):
        x = self.embedding(x) * math.sqrt(self.n_dim)
        x = self.positional_encoding(x)
        for block in self.encoder_blocks:
            x = block(x=x, src_padding_mask=padding_mask)
        return x
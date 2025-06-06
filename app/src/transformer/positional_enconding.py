import torch
import torch.nn as nn
import math


# Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model
class PositionalEncoding(nn.Module):

    def __init__(self, model, dropoutProbability=0.1, maxLength=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropoutProbability)
        
        pe = torch.zeros(maxLength, model)
        position = torch.arange(maxLength, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model, 2).float() * (-math.log(10000.0) / model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, tensor):
        """
        Arguments:
            tensor: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        tensor = tensor + self.pe[:, :tensor.size(1), :]
        return tensor
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt


# Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model
class PositionalEncoding(nn.Module):

    def __init__(self, model, dropoutProbability=0.1, maxLength=400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropoutProbability)
        
        positionalEnconding = torch.zeros(maxLength, model)
        position = torch.arange(maxLength, dtype=torch.float).unsqueeze(1)
        #print(f"Position: {position}")

        modelArange = torch.arange(0, model, 2, dtype=torch.float)
        #print(f"modelArange: {modelArange}")
        #print(f"modelArange Shape: {modelArange.shape}")

        naturalLogarithmByModel = (-math.log(10000.0) / model)
        #print(f"Natural Logarithm: {naturalLogarithmByModel}")

        divTerm = torch.exp(modelArange * naturalLogarithmByModel)
        #print(f"DivTerm: {divTerm}")
        #print(f"DivTerm Shape: {divTerm.shape}")

        positionalEnconding[:, 0::2] = torch.sin(position * divTerm)
        positionalEnconding[:, 1::2] = torch.cos(position * divTerm)
        positionalEnconding = positionalEnconding.unsqueeze(0)
        #print(f"Positional Encoding Shape: {positionalEnconding.shape}")
        

        # for i in range(0, maxLength):
        #     print(f"Positional Encoding at position {i}: {positionalEnconding[0, i, :]}")
        #     plt.figure(figsize=(400, 64))
        #     plt.plot(positionalEnconding[0, i, :].numpy())
        #     plt.title(f'Codificação Posicional - Posição {i}')
        #     plt.xlabel('Posição')
        #     plt.ylabel('Valor')
        #     plt.grid(True)
        #     plt.show()
        
        self.register_buffer('pe', positionalEnconding)

    

    def forward(self, tensor):
        """
        Arguments:
            tensor: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        tensor = tensor + self.pe[:, :tensor.size(1), :]
        return tensor
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

class PositionalEncoding(nn.Module):

    def __init__(self, model, dropoutProbability=0.1, maxLength=63):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropoutProbability)
        
        positionalEnconding = torch.zeros(maxLength, model)
        #print(f"Positional Encoding Shape: {positionalEnconding.shape}")
        position = torch.arange(maxLength, dtype=torch.float).unsqueeze(1)
        #print(f"Position: {position.shape}")

        modelArange = torch.arange(0, model, 2, dtype=torch.float)
        #print(f"modelArange: {modelArange}")
        #print(f"modelArange Shape: {modelArange.shape}")

        naturalLogarithmByModel = (-math.log(10000.0) / model)
        #print(f"Natural Logarithm: {naturalLogarithmByModel}")

        divTerm = torch.exp(modelArange * naturalLogarithmByModel)
        #print(f"DivTerm: {divTerm}")
        #print(f"DivTerm Shape: {divTerm.shape}")

        if(model % 2 == 0):
            positionalEnconding[:, 0::2] = torch.sin(position * divTerm)
            positionalEnconding[:, 1::2] = torch.cos(position * divTerm)
            positionalEnconding = positionalEnconding.unsqueeze(0)
            #print(f"Positional Encoding Shape: {positionalEnconding.shape}")
        else:
            raise ValueError("O modelo deve ter um número par de dimensões de embedding.")

        # for i in range(0, maxLength):
        # plt.figure(figsize=(63, 63))
        # plt.plot(positionalEnconding[0, 62, :].numpy())
        # plt.title(f'Codificação Posicional - Posição {62}')
        # plt.xlabel('Posição')
        # plt.ylabel('Valor')
        # plt.grid(True)
        # plt.show()
        
        self.register_buffer('positionalEnconding', positionalEnconding)

    def forward(self, tensor):
        """
        Arguments:
            tensor: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        tensor = tensor + self.positionalEnconding[:, :tensor.size(1), :]
        return tensor
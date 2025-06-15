import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, **kwargs):
        super(Transformer, self).__init__()

        # Hiperparâmetros
        self.model = kwargs.get('model')
        self.dropout = kwargs.get('dropout')
        self.numberEncoderLayers = kwargs.get('numberEncoderLayers')
        self.numberDecoderLayers = kwargs.get('numberDecoderLayers')
        self.numberHeads = kwargs.get('numberHeads')
        self.inputDimension = kwargs.get('inputDim')  # número de features de entrada
        self.outputDimension = kwargs.get('outputDim', 1)  # número de features de saída
        
        # Embeddings lineares para entrada contínua
        self.inputEmbedding = nn.Linear(self.inputDimension, self.model)
        self.outputEmbedding = nn.Linear(self.outputDimension, self.model)

        # Encoder e Decoder
        self.encoder = Encoder(self.model, self.dropout, self.numberEncoderLayers, self.numberHeads)
        self.decoder = Decoder(self.model, self.dropout, self.numberDecoderLayers, self.numberHeads)

        # Saída final
        self.fc = nn.Linear(self.model, self.outputDimension)

    @staticmethod
    def generate_square_subsequent_mask(size: int):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
        return mask
    
    def encode(self, x):
        # x: (S, B, input_dim)
        x = self.inputEmbedding(x)  # aplica linear sobre a última dimensão
        encoderOutput = self.encoder(x)  # espera (S, B, model_dim)
        return encoderOutput

    def decode(self, y, memory):
        # y: (T, B, output_dim)
        y = self.outputEmbedding(y)  # aplica linear sobre a última dimensão
        y = y.transpose(0, 1)    
        targetMask = self.generate_square_subsequent_mask(y.size(0)).to(y.device)
        decoderOutput = self.decoder(y, memory, targetMask=targetMask)  # espera (T, B, model_dim)
        print(f"y shape after decoder: {decoderOutput.shape}")
        return self.fc(decoderOutput)  # (T, B, output_dim)

    def forward(self, x, y):
        """
        x: (B, S, input_dim)
        y: (B, T, output_dim)
        """
        print(f"x shape at forward start: {x.shape}")  # (B, S, 1) esperado

        x = x.permute(1, 0, 2)  # (S, B, input_dim)
        y = y.permute(1, 0, 2)  # (T, B, output_dim)

        memory = self.encode(x)


        output = self.decode(y, memory)
        output = output.permute(1, 0, 2)  # volta para (B, T, output_dim)
        #print(f"Output shape: {output.shape}")
        return output
import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from tqdm import tqdm

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2

class Transformer(nn.Module):
    def __init__(self, **kwargs):
        super(Transformer, self).__init__()
        
        for key, value in kwargs.items():
            print(f" * {key}={value}")
        
        self.vocabularySize = kwargs.get('vocabularySize')
        self.model = kwargs.get('model')
        self.dropout = kwargs.get('dropout')
        self.numberEncoderLayers = kwargs.get('numberEncoderLayers')
        self.numberDecoderLayers = kwargs.get('numberDecoderLayers')
        self.numberHeads = kwargs.get('numberHeads')
        self.batchSize = kwargs.get('batchSize')
        self.padIdx = kwargs.get('padIdx', 0)

        self.encoder = Encoder(
            self.model, self.dropout, self.numberEncoderLayers, self.numberHeads)
        
        self.decoder = Decoder(
            self.model, self.dropout, self.numberDecoderLayers, self.numberHeads)
        
        self.fc = nn.Linear(self.model, self.vocabularySize)

    @staticmethod    
    def generate_square_subsequent_mask(size: int):
            """Generate a triangular (size, size) mask. From PyTorch docs."""
            mask = (1 - torch.triu(torch.ones(size, size), diagonal=1)).bool()
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            return mask

    def encode(
            self, 
            x: torch.Tensor, 
        ) -> torch.Tensor:
        """
        Input
            x: (B, S) with elements in (0, C) where C is num_classes
        Output
            (B, S, E) embedding
        """
        mask = (x == self.padIdx).float()
        encoderPaddingMask = mask.masked_fill(mask == 1, float('-inf'))
        
        # (B, S, E)
        encoderOutput = self.encoder(
            x, 
            paddingMask=encoderPaddingMask
        )  
        
        return encoderOutput, encoderPaddingMask
    
    def decode(
            self, 
            target: torch.Tensor, 
            memory: torch.Tensor, 
            memoryPaddingMask=None
        ) -> torch.Tensor:
        """
        B = Batch size
        S = Source sequence length
        L = Target sequence length
        E = Model dimension
        
        Input
            encoded_x: (B, S, E)
            y: (B, L) with elements in (0, C) where C is num_classes
        Output
            (B, L, C) logits
        """
        mask = (target == self.padIdx).float()
        targetPaddingMask = mask.masked_fill(mask == 1, float('-inf'))

        decoderOutput = self.decoder(
            target=target, 
            memory=memory, 
            targetMask=self.generate_square_subsequent_mask(target.size(1)), 
            targetPaddingMask=targetPaddingMask, 
            memoryPaddingMask=memoryPaddingMask,
        )  
        output = self.fc(decoderOutput)  # shape (B, L, C)
        return output

    def forward(
            self, 
            x: torch.Tensor, 
            y: torch.Tensor, 
        ) -> torch.Tensor:
        """
        Input
            x: (B, Sx) with elements in (0, C) where C is num_classes
            y: (B, Sy) with elements in (0, C) where C is num_classes
        Output
            (B, L, C) logits
        """
        
        # Encoder output shape (B, S, E)
        encoderOutput, encoderPaddingMask = self.encode(x)  

        # Decoder output shape (B, L, C)
        decoderOutput = self.decode(
            target=y, 
            memory=encoderOutput, 
            memoryPaddingMask=encoderPaddingMask
        )  
        
        return decoderOutput
    
    # Define a prediction method for the Transformer model
    def predict(
        self,
        x: torch.Tensor,
        startOfSentenceIdx: int=1,
        endOfSentenceIdx: int=2,
        maxLength: int=None
    ) -> torch.Tensor:
        """
        Method to use at inference time. Predict y from x one token at a time. This method is greedy
        decoding.
        Input
            x: str
        Output
            (B, L, C) logits
        """

        # Pad the tokens with beginning and end of sentence tokens
        x = torch.cat([
            torch.tensor([startOfSentenceIdx]), 
            x, 
            torch.tensor([endOfSentenceIdx])]
        ).unsqueeze(0)

        encoder_output, mask = self.transformer.encode(x) # (B, S, E)
        
        if not maxLength:
            maxLength = x.size(1)

        outputs = torch.ones((x.size()[0], maxLength)).type_as(x).long() * startOfSentenceIdx
        for step in range(1, maxLength):
            y = outputs[:, :step]
            probs = self.transformer.decode(y, encoder_output)
            output = torch.argmax(probs, dim=-1)
            
            # Prediction step by step
            # print(f"Knowing {y} we output {output[:, -1]}")

            if output[:, -1].detach().numpy() in (endOfSentenceIdx, startOfSentenceIdx):
                break
            outputs[:, step] = output[:, -1]
            
        return outputs
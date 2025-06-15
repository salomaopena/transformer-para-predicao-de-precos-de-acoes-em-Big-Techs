import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, headDimension=256, numberHeads=4):
        """
        headDimension: Dimensionalidade total do modelo (ex: 256).
        numberHeads: Número de cabeças de atenção (ex: 4).
        """
        super(MultiHeadAttention, self).__init__()
        self.headDimension = headDimension
        self.numberHeads = numberHeads

        assert headDimension % numberHeads == 0, "headDimension deve ser divisível por numberHeads"
        self.dimensionPerHead = headDimension // numberHeads

        # Camadas lineares para Q, K e V, todas mantendo a dimensão total
        self.queryLinear = nn.Linear(headDimension, headDimension, bias=False)
        self.keyLinear = nn.Linear(headDimension, headDimension, bias=False)
        self.valueLinear = nn.Linear(headDimension, headDimension, bias=False)

        # Camada final para recombinar as cabeças
        self.outputLinear = nn.Linear(headDimension, headDimension, bias=False)

    def split_into_heads(self, x):
        batchSize, seqLen, hiddenDim = x.size()
        x = x.view(batchSize, seqLen, self.numberHeads, self.dimensionPerHead)
        return x.transpose(1, 2)  # (batch, heads, seqLen, dimPerHead)

    def combine_heads(self, x):
        batchSize, numHeads, seqLen, dimensionPerHead = x.size()
        return x.transpose(1, 2).contiguous().view(batchSize, seqLen, numHeads * dimensionPerHead)

    def scaled_dot_product_attention(self, query, key, value, attentionMask=None, keyPaddingMask=None):
        dk = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dk)  # (B, H, tgt_len, src_len)

        if attentionMask is not None:
            scores += attentionMask.unsqueeze(0)  # broadcast se for (tgt_len, src_len)

        if keyPaddingMask is not None:
        # Esperado (B, src_len), transforma para (B, 1, 1, src_len)
            scores = scores.masked_fill(keyPaddingMask.unsqueeze(1).unsqueeze(2).bool(), float('-inf'))

        attentionWeights = torch.softmax(scores, dim=-1)
        attentionOutput = torch.matmul(attentionWeights, value)  # (B, H, tgt_len, dimPerHead)

        return attentionOutput, attentionWeights

    def forward(self, query, key, value, attentionMask=None, keyPaddingMask=None):
        # Passa pelas projeções lineares
        query = self.queryLinear(query)
        key = self.keyLinear(key)
        value = self.valueLinear(value)

        # Divide em cabeças
        query = self.split_into_heads(query)
        key = self.split_into_heads(key)
        value = self.split_into_heads(value)

        # Atenção escalada
        attnOutput, attnWeights = self.scaled_dot_product_attention(
            query, key, value, attentionMask, keyPaddingMask
        )

        # Junta cabeças
        combined = self.combine_heads(attnOutput)
        output = self.outputLinear(combined)

        # Salva pesos da atenção (opcional)
        self.attentionWeights = attnWeights

        return output

import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, headDimension=256, numberHeads=4):
        """
        headDimension: Dimensionality of the input, by default 256.
        numberHeads: The number of attention heads to split the input into, by default 4.
        """
        super(MultiHeadAttention, self).__init__()
        self.headDimension = headDimension
        self.numberHeads = numberHeads

        assert headDimension % numberHeads == 0, "Head Dimension must be divisible by the number of heads."
        
        self.valueLinearTransformation = nn.Linear(headDimension, numberHeads, bias=False) # the Value part
        self.keyLinearTransformation = nn.Linear(headDimension, numberHeads, bias=False) # the Key part
        self.queryLinearTransformation = nn.Linear(headDimension, headDimension, bias=False) # the Query part
        self.outputLinearTransformation = nn.Linear(headDimension, headDimension, bias=False) # the output layer
        
    def check_scaled_dot_product_attention_inputs(self, x):
        assert x.size(1) == self.numberHeads, f"Expected size of x to be ({-1, self.numberHeads, -1, self.headDimension // self.numberHeads}), got {x.size()}"
        assert x.size(3) == self.headDimension // self.numberHeads
           
    def scaled_dot_product_attention(
            self, 
            query, 
            key, 
            value, 
            attentionMask=None, 
            keyPaddingMask=None):
        """
        query : tensor of shape (batch_size, num_heads, query_sequence_length, hidden_dim//num_heads)
        key : tensor of shape (batch_size, num_heads, key_sequence_length, hidden_dim//num_heads)
        value : tensor of shape (batch_size, num_heads, key_sequence_length, hidden_dim//num_heads)
        attention_mask : tensor of shape (query_sequence_length, key_sequence_length)
        key_padding_mask : tensor of shape (sequence_length, key_sequence_length)
        
        """
        self.check_scaled_dot_product_attention_inputs(query)
        self.check_scaled_dot_product_attention_inputs(key)
        self.check_scaled_dot_product_attention_inputs(value)
        
        dk = query.size(-1)
        print(f"query: {query.size()}, dk: {dk}")
        targetLength, sourceLength = query.size(-2), key.size(-2)

        # logits = (B, H, tgt_len, E) * (B, H, E, src_len) = (B, H, tgt_len, src_len)
        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dk) 
        
        # Attention mask here
        if attentionMask is not None:
            if attentionMask.dim() == 2:
                assert attentionMask.size() == (targetLength, sourceLength)
                attentionMask = attentionMask.unsqueeze(0)
                logits = logits + attentionMask
            else:
                raise ValueError(f"Attention mask size {attentionMask.size()}")
                
        # Key mask here
        if keyPaddingMask is not None:
            keyPaddingMask = keyPaddingMask.unsqueeze(1).unsqueeze(2) # Broadcast over batch size, num heads
            logits = logits + keyPaddingMask
        
        attention = torch.softmax(logits, dim=-1)
        output = torch.matmul(attention, value) # (batch_size, num_heads, sequence_length, hidden_dim)
        
        return output, attention

    def split_into_heads(self, x, numberHeads):
        batchSize, sequenceLength, hiddenDimension = x.size()
        x = x.view(batchSize, sequenceLength, numberHeads, hiddenDimension // numberHeads)
        
        return x.transpose(1, 2) # Final dim will be (batch_size, num_heads, seq_length, , hidden_dim // num_heads)

    def combine_heads(self, x):
        batchSize, numHeads, sequenceLength, headHiddenDimension = x.size()
        return x.transpose(1, 2).contiguous().view(batchSize, sequenceLength, numHeads * headHiddenDimension)
        
    def forward(
            self, 
            query, 
            key, 
            value, 
            attentionMask=None, 
            keyPaddingMask=None):
        """
        q : tensor of shape (batch_size, query_sequence_length, hidden_dim)
        k : tensor of shape (batch_size, key_sequence_length, hidden_dim)
        v : tensor of shape (batch_size, key_sequence_length, hidden_dim)
        attention_mask : tensor of shape (query_sequence_length, key_sequence_length)
        key_padding_mask : tensor of shape (sequence_length, key_sequence_length)
       
        """
        query = self.queryLinearTransformation(query)
        key = self.keyLinearTransformation(key)
        value = self.valueLinearTransformation(value)

        query = self.split_into_heads(query, self.numberHeads)
        key = self.split_into_heads(key, self.numberHeads)
        value = self.split_into_heads(value, self.numberHeads)
        
        # attn_values, attn_weights = self.multihead_attn(q, k, v, attn_mask=attention_mask)
        attentionValues, attentionWeights  = self.scaled_dot_product_attention(
            query=query, 
            key=key, 
            value=value, 
            attentionMask=attentionMask,
            keyPaddingMask=keyPaddingMask,
        )
        grouped = self.combine_heads(attentionValues)
        output = self.outputLinearTransformation(grouped)
        
        self.attentionWeights = attentionWeights
        
        return output
import torch
import torch.nn as nn

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2

class StockPredictor(nn.Module):
    def __init__(self, transformer):
        super(StockPredictor, self).__init__()
        self.transformer = transformer
    
    @staticmethod
    def str_to_tokens(s):
        return [ord(z)-97+3 for z in s]
    
    @staticmethod
    def tokens_to_str(tokens):
        return "".join([chr(x+94) for x in tokens])
    
    def __call__(self, sentence, maxLength=None, padding=False):
        
        x = torch.tensor(self.str_to_tokens(sentence))
        x = torch.cat([torch.tensor([SOS_IDX]), x, torch.tensor([EOS_IDX])]).unsqueeze(0)
        
        encoderOutput, mask = self.transformer.encode(x) # (B, S, E)
        
        if not maxLength:
            maxLength = x.size(1)
            
        outputs = torch.ones((x.size()[0], maxLength)).type_as(x).long() * SOS_IDX
        
        for step in range(1, maxLength):
            y = outputs[:, :step]
            probs = self.transformer.decode(y, encoderOutput)
            output = torch.argmax(probs, dim=-1)
            print(f"Knowing {y} we output {output[:, -1]}")
            if output[:, -1].detach().numpy() in (EOS_IDX, SOS_IDX):
                break
            outputs[:, step] = output[:, -1]
            
        return self.tokens_to_str(outputs[0])
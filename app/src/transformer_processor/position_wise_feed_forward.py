import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    def __init__(self, model: int, feedForward: int):
        super(PositionWiseFeedForward, self).__init__()

        self.feedForwardLinearTransformation1 = nn.Linear(model, feedForward)
        # print(f"FeedForward1 Weight: {self.feedForwardLinearTransformation1.weight[0, :]}")
        # print(f"FeedForward1 Weight Shape: {self.feedForwardLinearTransformation1.weight.shape}")
        # print(f"FeedForward1 Bias: {self.feedForwardLinearTransformation1.bias}")
        # print(f"FeedForward1 Bias Shape: {self.feedForwardLinearTransformation1.bias.shape}")
        
        self.feedForwardLinearTransformation2 = nn.Linear(feedForward, model)
        # print(f"FeedForward2 Weight: {self.feedForwardLinearTransformation2.weight[0, :]}")
        # print(f"FeedForward2 Weight Shape: {self.feedForwardLinearTransformation2.weight.shape}")
        # print(f"FeedForward2 Bias: {self.feedForwardLinearTransformation2.bias}")
        # print(f"FeedForward2 Bias Shape: {self.feedForwardLinearTransformation2.bias.shape}")

        self.linearUnitFuction = nn.ReLU()

    def forward(self, x):
        return self.feedForwardLinearTransformation2(self.linearUnitFuction(self.feedForwardLinearTransformation1(x)))
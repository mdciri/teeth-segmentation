import torch
import torch.nn as nn

class MeanDiceScore(nn.Module):
    """ calculates the mean dice score
    """

    def __init__(self, softmax=True, weights=None, epsilon=1.e-5):
        super().__init__()

        self.softmax = softmax
        self.weights = weights
        self.eps = epsilon

    def forward(self, inputs, targets):

        if self.softmax:
            inputs = nn.Softmax(dim=1)(inputs)

        if self.weights == None:
            self.weights = torch.ones(inputs.shape[1])
        w = self.weights[None, :, None, None]
        w = w.to(inputs.device)

        num = 2 * torch.sum(inputs * targets * w, dim=(1, 2, 3))
        den = torch.sum((inputs + targets) * w, dim=(1, 2, 3)) + self.eps

        return torch.mean(num/den)
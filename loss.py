import torch
import torch.nn as nn
from metric import MeanDiceScore

class MeanDiceLoss(nn.Module):
    """ calculates the mean dice loss
    """

    def __init__(self, softmax=True, weights=None, epsilon=1.e-5):
        super().__init__()

        self.dice = MeanDiceScore(softmax, weights, epsilon)

    def forward(self, inputs, targets):

        dice_score = self.dice(inputs, targets)

        return 1 - dice_score
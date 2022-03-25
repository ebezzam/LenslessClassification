from torch import nn
import numpy as np


class MultiClassLogistic(nn.Module):
    """
    Example: https://gist.github.com/xmfbit/b27cdbff68870418bdb8cefa86a2d558
    """

    def __init__(self, input_shape):
        super(MultiClassLogistic, self).__init__()
        self.flatten = nn.Flatten()
        self.multiclass_logistic_reg = nn.Sequential(
            nn.Linear(int(np.prod(input_shape)), 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.multiclass_logistic_reg(x)
        return logits

    def name(self):
        return "MultiClassLogistic"

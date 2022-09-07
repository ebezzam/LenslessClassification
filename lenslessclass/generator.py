from torch import nn
import numpy as np


class SingleHidden(nn.Module):
    """
    Example: https://blog.paperspace.com/autoencoder-image-compression-keras/

    Default to MNIST output dimension.
    """

    def __init__(self, input_shape, hidden_dim=300, n_output=784):
        assert isinstance(hidden_dim, int)
        self.hidden_dim = hidden_dim
        super(SingleHidden, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(int(np.prod(input_shape)), hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, n_output)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        return x

    def name(self):
        return f"SingleHidden{self.hidden_dim}"

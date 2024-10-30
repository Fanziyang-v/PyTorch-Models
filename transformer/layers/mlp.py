"""
Pointwise fully connected network. i.e. MultiLayer Perceptron(MLP).
"""

from torch import nn, Tensor


class MLP(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, drop_prob: float) -> None:
        """Initialize a MLP.

        Args:
            d_model (int): dimensionality of the model.
            d_hidden (int): dimensionality of the hidden layer.
            drop_prob (float): dropout ratio in dropout layer.
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x: Tensor):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

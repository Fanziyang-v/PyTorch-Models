import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F


class Inception(nn.Module):
    """Inception Module in GoogLeNet."""

    def __init__(
        self,
        in_channels: int,
        c1: int,
        c2: tuple[int, int],
        c3: tuple[int, int],
        c4: int,
    ) -> None:
        """Initialize an Inception Module."""
        super(Inception, self).__init__()
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1, stride=1)

        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1, stride=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, stride=1, padding=1)

        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1, stride=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, stride=1, padding=2)

        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1, stride=1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass in Inception Module."""
        out1 = F.relu(self.p1_1(x))
        out2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        out3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        out4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat([out4, out3, out2, out1], dim=1)

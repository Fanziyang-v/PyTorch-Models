from torch import nn
from torch import Tensor


class VGGBlock(nn.Module):
    """VGG Block."""
    def __init__(self, in_channels: int, out_channels: int, num_convs: int) -> None:
        """Initialize a VGG Block.

        Args:
            in_channels (int): number of channels of input feature map.
            out_channels (int): number of channels of output feature map.
            num_convs (int): number of convolution layers.
        """
        super(VGGBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)]
        for _ in range(num_convs - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        return self.block(x)

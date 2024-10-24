from torch import nn
from torch import Tensor


class BasicBlock(nn.Module):
    """Basic Residual Block in ResNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        """Initialize a basic residual block.

        Args:
            in_channels (int): number of channels of input feature map.
            out_channels (int): number of channels of output feature map.
            stride(int): stride of the first convolution. Defaults to 1.
            downsample (nn.Module | None, optional): downsampling module. Defaults to None.
        """
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: Tensor):
        shortcut = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            shortcut = self.downsample(x)
        out += shortcut
        out = self.relu2(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck Block in ResNet."""

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        """Initialize a bottleneck block in ResNet.

        Args:
            in_channels (int): number of channels of input feature map.
            mid_channels (int): number of channels of middle feature map.
            out_channels (int): number of channels of output feature map.
            stride (int, optional): stride of convolution. Defaults to 1.
            downsample (nn.Module | None, optional): downsampling module. Defaults to None.
        """
        super(Bottleneck, self).__init__()
        mid_channels = out_channels // self.expansion
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(mid_channels, mid_channels, stride=stride)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x: Tensor):
        shortcut = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            shortcut = self.downsample(x)
        out += shortcut
        out = self.relu3(out)
        return out


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """Construct a convolutional layer with kernel size of 3 and padding of 1.

    Args:
        in_channels (int): number of channels of input feature map.
        out_channels (int): number of channels of output feature map.
        stride (int): stride of convolution. Defaults to 1.

    Returns:
        nn.Conv2d: 3x3 convolution layer with specific stride.
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)

"""PyTorch implementation for Deep Residual Learning for Image Recognition.

For more details, see: http://arxiv.org/abs/1512.03385
"""

from torch import nn
from torch import Tensor
from residual_blocks import BasicBlock, Bottleneck


class ResNet(nn.Module):
    """ResNet."""

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        layers: list[int],
        channels: list[int],
        block: type[BasicBlock | Bottleneck],
    ) -> None:
        """Initialize ResNet.

        Args:
            num_channels (int): number of channels of input images.
            num_classes (int): number of classes of images.
            layers (list[int]): number of residual blocks each residual block.
            channels (list[int]): number of output channels each residual block.
            block (type[BasicBlock  |  Bottleneck]): residual block type.

        Examples:
        >>> model = ResNet(3, 1000, [2, 2, 2, 2], [64, 128, 256, 512], BasicBlock) # ResNet-18
        >>> model = ResNet(3, 1000, [3, 4, 6, 3], [64, 128, 256, 512], BasicBlock) # ResNet-34
        >>> model = ResNet(3, 1000, [3, 4, 6, 3], [256, 512, 1024, 2048], Bottleneck) # ResNet-50
        >>> model = ResNet(3, 1000, [3, 4, 23, 3], [256, 512, 1024, 2048], Bottleneck) # ResNet-101
        >>> model = ResNet(3, 1000, [3, 8, 36, 3], [256, 512, 1024, 2048], Bottleneck) # ResNet-152
        """
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(channels[0], layers[0], block)
        self.layer2 = self.make_layer(channels[1], layers[1], block, stride=2)
        self.layer3 = self.make_layer(channels[2], layers[2], block, stride=2)
        self.layer4 = self.make_layer(channels[3], layers[3], block, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, num_classes)

    def forward(self, images: Tensor):
        out = self.conv(images)
        out = self.bn(out)
        out = self.pool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(images.size(0), -1)
        out = self.fc(out)
        return out

    def make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        block: BasicBlock | Bottleneck,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                ),
                nn.BatchNorm2d(out_channels),
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(num_blocks - 1):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

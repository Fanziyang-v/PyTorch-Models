from torch import nn
from torch import Tensor
from xception_modules import XceptionModule1, XceptionModule2, DepthwiseSeparableConv2d


class Xception(nn.Module):
    """Xception."""

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """Initialize Xception.

        Args:
            num_channels (int): number of channels of input feature map.
            num_classes (int): number of image classes.
        """
        super(Xception, self).__init__()
        self.model = nn.Sequential(
            # Entry flow.
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            XceptionModule1(64, 128, 128, relu=False),
            XceptionModule1(128, 256, 256),
            XceptionModule1(256, 728, 728),
            # Middle flow.
            XceptionModule2(728),
            XceptionModule2(728),
            XceptionModule2(728),
            XceptionModule2(728),
            XceptionModule2(728),
            XceptionModule2(728),
            XceptionModule2(728),
            XceptionModule2(728),
            # Exit flow.
            XceptionModule1(728, 728, 1024),
            DepthwiseSeparableConv2d(1024, 1536, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv2d(1536, 2048, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, num_classes),
        )

    def forward(self, images: Tensor):
        return self.model(images)

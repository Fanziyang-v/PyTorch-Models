"""
PyTorch implementation for Going Deeper with Convolutions.

For more details: see: http://arxiv.org/abs/1409.4842
"""

from torch import nn
from torch import Tensor
from inception import Inception


class GoogLeNet(nn.Module):
    """GoogLeNet."""

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """Initialize GoogLeNet.

        Args:
            num_channels(int): number of channels of input images.
            num_classes(int): number of classes.
        """
        super(GoogLeNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3_1 = Inception(192, 64, (96, 128), (16, 32), 32)
        self.inception3_2 = Inception(256, 128, (128, 192), (32, 96), 64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4_1 = Inception(480, 192, (96, 208), (16, 48), 64)
        self.inception4_2 = Inception(512, 160, (112, 224), (24, 64), 64)
        self.inception4_3 = Inception(512, 128, (128, 256), (24, 64), 64)
        self.inception4_4 = Inception(512, 112, (144, 288), (32, 64), 64)
        self.inception4_5 = Inception(528, 256, (160, 320), (32, 128), 128)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5_1 = Inception(832, 256, (160, 320), (32, 128), 128)
        self.inception5_2 = Inception(832, 384, (192, 384), (48, 128), 128)
        self.pool5 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(1024, num_classes)

    def forward(self, images: Tensor):
        """Forward pass in GoogLeNet.

        Args:
            images(Tensor): input images of shape (N, C, H, W)
        """
        out = self.conv1(images)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.inception3_1(out)
        out = self.inception3_2(out)
        out = self.pool3(out)

        out = self.inception4_1(out)
        out = self.inception4_2(out)
        out = self.inception4_3(out)
        out = self.inception4_4(out)
        out = self.inception4_5(out)
        out = self.pool4(out)

        out = self.inception5_1(out)
        out = self.inception5_2(out)
        out = self.pool5(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

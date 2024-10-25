from torch import nn
from torch import Tensor


class AlexNet(nn.Module):
    """AlexNet.

    Architecture: [conv - relu - max pool] x 2 - [conv - relu] x 3 - max pool - [affine - dropout] x 2 - affine - softmax
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """Initialize AlexNet.

        Args:
            num_channels(int): number of channels of input images.
            num_classes(int): number of classes.
        """
        super(AlexNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(num_channels, 96, kernel_size=11, stride=4, padding=2),  # 55x55
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 27x27
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),  # 27x27
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 13x13
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),  # 13x13
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),  # 13x13
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),  # 13x13
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 6x6
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, images: Tensor) -> Tensor:
        """Forward pass in AlexNet.

        Args:
            images(Tensor): input images of shape(N, num_channels, 224, 224)

        Returns:
            Tensor: scores matrix of shape (N, num_classes)
        """
        return self.model(images)

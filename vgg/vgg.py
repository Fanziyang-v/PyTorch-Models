from torch import nn
from torch import Tensor
from vgg_block import VGGBlock


class VGG(nn.Module):
    def __init__(self, num_channels: int, num_classes: int, layers: list[int]) -> None:
        super(VGG, self).__init__()
        self.model = nn.Sequential(
            VGGBlock(num_channels, 64, layers[0]),
            VGGBlock(64, 128, layers[1]),
            VGGBlock(128, 256, layers[2]),
            VGGBlock(256, 512, layers[3]),
            VGGBlock(512, 512, layers[4]),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, images: Tensor):
        return self.model(images)

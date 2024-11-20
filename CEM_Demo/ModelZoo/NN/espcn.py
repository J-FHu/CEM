import math
from torch import nn


class ESPCN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, width=64, depth=16):
        super(ESPCN, self).__init__()
        layers = [nn.Conv2d(num_channels, width, kernel_size=5, padding=5//2), nn.Tanh()]
        for i in range(depth):
            layers.append(nn.Conv2d(width, width, kernel_size=5, padding=5//2))
            layers.append(nn.Tanh())
        self.first_part = nn.Sequential(*layers)
        self.last_part = nn.Sequential(
            nn.Conv2d(32, num_channels * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
            nn.PixelShuffle(scale_factor)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels == 32:
                    nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                    nn.init.zeros_(m.bias.data)
                else:
                    nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.last_part(x)
        return x
import torch
import torch.nn as nn

class Upscaler(nn.Module):
    def __init__(self, low_res=32, kernel_size=3):
        super(Upscaler, self).__init__()

        self.low_res = low_res

        self.seq = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 3, kernel_size=kernel_size, padding='same'),
            nn.Tanh()
        )

    def forward(self, inputs) -> torch.Tensor:
        img, mask = inputs

        x = self.seq(img)
        x = x * (1 - mask.unsqueeze(1))
        x = x + img

        return x

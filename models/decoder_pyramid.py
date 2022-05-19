import torch
import torch.nn as nn
from .basemodule import BaseModule, ResnetBlock


class DecoderPyramid(BaseModule):
    def __init__(self, dim=512, init_weight=True):
        super(DecoderPyramid, self).__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=dim, out_channels=dim // 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(dim // 2, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=dim // 2, out_channels=dim // 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(dim // 4, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=dim // 4, out_channels=dim // 8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(dim // 8, track_running_stats=False),
            nn.ReLU(True),

            # nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=dim // 8, out_channels=3, kernel_size=1, padding=0)
        )

        if init_weight:
            self.init_weights()

    def forward(self, x, src, attn_map):
        x = self.decoder(x)
        x = (torch.tanh(x) + 1) / 2
        return x


def build_decoder_pyramid(config):
    dim = config.dim_model
    model = DecoderPyramid(dim)
    return model

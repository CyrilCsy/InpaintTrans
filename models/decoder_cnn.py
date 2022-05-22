import torch
import torch.nn as nn
from .basemodule import BaseModule, ResnetBlock
from util.misc import NestedTensor
from typing import Optional, List


class DecoderCNN(nn.Module):
    def __init__(self, dim=512, skip_connect=True):
        super(DecoderCNN, self).__init__()
        cat = 2 if skip_connect else 1
        self.skip_con = skip_connect
        self.dec1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=dim * cat, out_channels=dim // 2, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(dim // 2, track_running_stats=False),
                nn.ReLU(True))  # 512 -> 256
        self.dec2 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=dim // 2 * cat, out_channels=dim // 4, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(dim // 4, track_running_stats=False),
                nn.ReLU(True))  # 256 -> 128
        self.dec3 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=dim // 4 * cat, out_channels=dim // 8, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(dim // 8, track_running_stats=False),
                nn.ReLU(True))  # 128 -> 64
        self.dec4 = nn.Sequential(
                # nn.ReflectionPad2d(3),
                nn.Conv2d(in_channels=dim // 8 * cat, out_channels=3, kernel_size=1, stride=1, padding=0))   # 64 -> 3
        # self.decs = []
        # self.decs.extend(
        #     [nn.Sequential(
        #         nn.ConvTranspose2d(in_channels=dim * cat, out_channels=dim // 2, kernel_size=4, stride=2, padding=1),
        #         nn.InstanceNorm2d(dim // 2, track_running_stats=False),
        #         nn.ReLU(True)
        #     ),  # 512 -> 256
        #
        #     nn.Sequential(
        #         nn.ConvTranspose2d(in_channels=dim // 2 * cat, out_channels=dim // 4, kernel_size=4, stride=2, padding=1),
        #         nn.InstanceNorm2d(dim // 4, track_running_stats=False),
        #         nn.ReLU(True)
        #     ),  # 256 -> 128
        #
        #     nn.Sequential(
        #         nn.ConvTranspose2d(in_channels=dim // 4 * cat, out_channels=dim // 8, kernel_size=4, stride=2, padding=1),
        #         nn.InstanceNorm2d(dim // 8, track_running_stats=False),
        #         nn.ReLU(True)
        #     ),  # 128 -> 64
        #
        #     nn.Sequential(
        #         # nn.ReflectionPad2d(3),
        #         nn.Conv2d(in_channels=dim // 8 * cat, out_channels=3, kernel_size=1, stride=1, padding=0)
        #     )   # 64 -> 3
        #     ])
        self._reset_parameters()

    def _reset_parameters(self):  # init weight with xaiver_uniform
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, a=0, mode='fan_in')

    def forward(self, x, nts=None, attn_map=None):
        x = x + self.attention_transfer(x, attn_map)
        if self.skip_con:
            x = torch.cat((x, nts[-1].tensors), dim=1)
        x = self.dec1(x)

        x = x + self.attention_transfer(x, attn_map)
        if self.skip_con:
            x = torch.cat((x, nts[-2].tensors), dim=1)
        x = self.dec2(x)

        x = x + self.attention_transfer(x, attn_map)
        if self.skip_con:
            x = torch.cat((x, nts[-3].tensors), dim=1)
        x = self.dec3(x)

        x = x + self.attention_transfer(x, attn_map)
        if self.skip_con:
            x = torch.cat((x, nts[-4].tensors), dim=1)
        x = self.dec4(x)

        x = (torch.tanh(x) + 1) / 2
        return x

    def attention_transfer(self, x, attn_map):
        b, c, h, w = x.size()
        feature = self.extract_image_patches(x, 32)   # (b,n*n,p*p*c)
        out = torch.bmm(attn_map, feature)
        out = torch.reshape(out, [b, 32, 32, h//32, w//32, c])
        out = out.permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)
        return out

    def extract_image_patches(self, img, num):
        b, c, h, w = img.size()
        img = torch.reshape(img, [b, c, num, h//num, num, w//num])
        img = img.permute([0, 2, 4, 3, 5, 1]).reshape(b, num * num, -1)
        return img


def build_decoder_cnn(config):
    dim = config.dim_model
    model = DecoderCNN(dim)
    return model

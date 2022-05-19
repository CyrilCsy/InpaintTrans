import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from models.basemodule import BaseModule
from models.conv2d import PartialConv2d
from models.attention_norm import AttentionNormalization
from util.misc import NestedTensor


class BackboneCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=512, dim=None, threshold=0.5):
        super(BackboneCNN, self).__init__()

        dim = out_channels if dim is None else dim

        self.pad = nn.ReflectionPad2d(3)

        self.conv1 = PartialConv2d(in_channels=in_channels, out_channels=out_channels // 8, kernel_size=7, stride=1,
                                   padding=0, return_mask=True, threshold=threshold)  # in -> 64
        self.conv2 = PartialConv2d(in_channels=out_channels // 8, out_channels=out_channels // 4, kernel_size=5,
                                   stride=2, padding=2, return_mask=True, threshold=threshold)  # 64 -> 128
        self.conv3 = PartialConv2d(in_channels=out_channels // 4, out_channels=out_channels // 2, kernel_size=3,
                                   stride=2, padding=1, return_mask=True, threshold=threshold)  # 128 -> 256
        self.conv4 = PartialConv2d(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=3,
                                   stride=2, padding=1, return_mask=True, threshold=threshold)  # 256 -> 512

        self.norm_act1 = nn.Sequential(nn.InstanceNorm2d(out_channels // 8, track_running_stats=False),
                                       nn.ReLU(True))
        self.norm_act2 = nn.Sequential(nn.InstanceNorm2d(out_channels // 4, track_running_stats=False),
                                       nn.ReLU(True))
        self.norm_act3 = nn.Sequential(nn.InstanceNorm2d(out_channels // 2, track_running_stats=False),
                                       nn.ReLU(True))
        self.norm_act4 = nn.Sequential(nn.InstanceNorm2d(out_channels, track_running_stats=False), nn.ReLU(True))

        num_heads = 6
        self.linear1 = nn.Linear(dim, out_channels//8)  # 64
        self.linear2 = nn.Linear(dim, out_channels//4)  # 128
        self.linear3 = nn.Linear(dim, out_channels//2)  # 256
        self.linear4 = nn.Linear(dim, out_channels)     # 512

        self._reset_parameters()

        self.in_channels = in_channels

    def _reset_parameters(self):  # init weight with xaiver_uniform
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, a=0, mode='fan_in')

    def forward(self, nt: NestedTensor, pos, memory, memory_pos) -> List[NestedTensor]:

        outs = []
        x = self.pad(nt.tensors)
        mask = self.pad(nt.mask)

        if self.in_channels == 4:
            x = torch.cat((x, mask), dim=1)
        x, mask = self.conv1(x, mask)
        x = self.norm_act1(x)
        outs.append(NestedTensor(x, mask))

        mem = self.linear1(memory)
        res = attention_for_hole(x, mem)
        x = x * mask + res * (1 - mask)
        x, mask = self.conv2(x, mask)
        x = self.norm_act2(x)
        outs.append(NestedTensor(x, mask))

        mem = self.linear2(memory)
        res = attention_for_hole(x, mem)
        x = x * mask + res * (1 - mask)
        x, mask = self.conv3(x, mask)
        x = self.norm_act3(x)
        outs.append(NestedTensor(x, mask))

        mem = self.linear3(memory)
        res = attention_for_hole(x, mem)
        x = x * mask + res * (1 - mask)
        x, mask = self.conv4(x, mask)
        x = self.norm_act4(x)
        outs.append(NestedTensor(x, mask))

        return outs


def attention_for_hole(features, keys, values=None):
    b, n, dim = keys.size()
    attn_score = []
    outs = []

    values = keys if values is None else values
    for i in range(b):
        feature = features[i:i+1]
        key = keys[i].view(n, dim, 1, 1)
        value = values[i].view(n, dim, 1, 1)

        norm_factor = torch.sum(key**2, [1, 2, 3], keepdim=True) ** 0.5
        key = key / norm_factor
        attn = F.conv2d(feature, key, stride=1, padding=0)
        attn = F.softmax(attn, dim=1)
        attn_score.append(attn)
        out = F.conv_transpose2d(attn, value, stride=1, padding=0)
        outs.append(out)

    out = torch.cat(outs, dim=0)
    return out


def build_backbone_cnn(config):
    dim = config.dim_model
    model = BackboneCNN(in_channels=4, out_channels=dim, threshold=0.5)
    return model
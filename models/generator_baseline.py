import torch
import torch.nn as nn
from models.basemodule import BaseModule, ResnetBlock
from util.misc import NestedTensor
from models.backbone_cnn import build_backbone_cnn
from models.transformer import build_transformer
from models.decoder_cnn import build_decoder_cnn
from models.position_encoding import build_position_encoding


class GeneratorBaseline(nn.Module):
    def __init__(self, backbone, bottleneck, decoder):
        super(GeneratorBaseline, self).__init__()

        self.backbone = backbone
        self.bottleneck = bottleneck
        self.decoder = decoder

    def forward(self, nt: NestedTensor, emb=None):
        nts = self.backbone(NestedTensor(torch.cat((nt.tensors, nt.mask), dim=1), nt.mask))
        x, mask = nts[-1].decompose()
        x = self.bottleneck(x)
        x = self.decoder(x, nts)

        return x


def build_generator_baseline(config):
    backbone = build_backbone_cnn(config)
    res_net = ResnetBlock(dim=512, n=8, dilation=1, use_spectral_norm=False)
    decoder = build_decoder_cnn(config)
    model = GeneratorBaseline(backbone, res_net, decoder)
    return model

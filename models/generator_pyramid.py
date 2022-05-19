import torch
import torch.nn as nn
from models.basemodule import BaseModule, ResnetBlock
from util.misc import NestedTensor
from models.backbone_cnn import build_backbone_cnn
from models.transformer import build_transformer
from models.decoder_pyramid import build_decoder_pyramid
from models.position_encoding import build_position_encoding



class GeneratorPyramid(nn.Module):
    def __init__(self, backbone, transformer, decoder, position_encoding, project=True):
        super(GeneratorPyramid, self).__init__()

        self.backbone = backbone
        self.transformer = transformer
        self.decoder = decoder
        self.position_encoding = position_encoding

        channel_backbone = backbone.num_channels
        dim_trans = transformer.dim_model
        assert project or channel_backbone == dim_trans, "need to project into same dim"
        if project:
            self.input_proj = nn.Conv2d(channel_backbone, dim_trans, kernel_size=1)

    def forward(self, nt: NestedTensor, emb=None):
        nt_list = self.backbone(nt)
        x, mask = nt_list[-1].decompose()

        pos = self.position_encoding(nt_list[-1])
        if self.input_proj is not None:
            x = self.input_proj(x)
        x, attn_map = self.transformer(x, mask, pos, emb)
        x = self.decoder(x, nt_list, attn_map)

        return x


def build_generator_baseline(config):
    backbone = build_backbone_cnn(config)
    transformer = build_transformer(config)
    decoder = build_decoder_pyramid(config)
    position_encoding = build_position_encoding(config)
    model = GeneratorPyramid(backbone, transformer, decoder, position_encoding)
    return model

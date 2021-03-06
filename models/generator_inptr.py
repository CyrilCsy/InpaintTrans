import torch
import torch.nn as nn
import torch.nn.functional as F
from models.basemodule import BaseModule, ResnetBlock
from util.misc import NestedTensor
from models.backbone_cnn import build_backbone_cnn
from models.sampler import build_sampler
from models.transformer import build_transformer, build_transformer_decoder
from models.decoder_cnn import build_decoder_cnn
from models.position_encoding import build_position_encoding


class GeneratorInpTrans(nn.Module):
    def __init__(self, sampler, backbone, transformer, decoder, position_encoding):
        super(GeneratorInpTrans, self).__init__()

        self.sampler = sampler
        dim = sampler.dim
        self.backbone = backbone
        self.transformer_decoder = transformer
        self.decoder = decoder
        self.position_encoding = position_encoding

    def forward(self, nt: NestedTensor):
        src, src_pos = self.sampler(nt)  # (b, n ,dim)

        x, mask = nt.decompose()
        # x = x * mask.float()
        pos = self.position_encoding(NestedTensor(nt.tensors, None))
        nts, memory = self.backbone(NestedTensor(x, mask), pos, src, src_pos)

        tgt, mask = nts[-1].decompose()
        b, c, h, w = tgt.size()
        tgt = tgt.flatten(2).permute(0, 2, 1)
        mask = mask.flatten(2).permute(0, 2, 1).squeeze(dim=-1)
        tgt_pos = F.interpolate(pos, scale_factor=0.125, mode='bilinear')
        tgt_pos = tgt_pos.flatten(2).permute(0, 2, 1)
        x, attn_weights = self.transformer_decoder(x=tgt,
                                                   memory=memory,
                                                   mask=mask,
                                                   pos=tgt_pos,
                                                   memory_pos=src_pos)
        # x = self.transformer(src=src,
        #                      tgt=tgt,
        #                      src_mask=None,
        #                      tgt_mask=mask,
        #                      src_pos_embed=src_pos,
        #                      tgt_pos_embed=tgt_pos)
        x = x.permute(0, 2, 1).reshape(b, c, h, w)
        attn_map = attn_weights
        x = self.decoder(x, nts, attn_map)

        return x


def build_generator_inptr(config):
    sampler = build_sampler(config)
    dim = sampler.sample_dim
    backbone_cnn = build_backbone_cnn(config, dim)
    transformer_decoder = build_transformer_decoder(config)
    decoder = build_decoder_cnn(config)
    position_encoding = build_position_encoding(config)
    model = GeneratorInpTrans(sampler, backbone_cnn, transformer_decoder, decoder, position_encoding)
    sum_ = 0
    for name, param in model.named_parameters():
        mul = 1
        for size_ in param.shape:
            mul *= size_  # ????????????????????????
        sum_ += mul  # ????????????????????????
        print('%14s : %s' % (name, param.shape))  # ??????????????????????????????
        # print('%s' % param)						# ???????????????????????????????????????????????????????????????
    print('???????????????', sum_)
    return model

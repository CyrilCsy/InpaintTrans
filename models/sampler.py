import torch
import torch.nn as nn
import torch.nn.functional as F
from models.basemodule import MLP
from models.position_encoding import build_position_encoding
from util.misc import NestedTensor


class Sampler(nn.Module):
    def __init__(self, input_size, patch_size,
                 stride=None,
                 num_patch=None,
                 threshold=1.0,
                 in_channel=3,
                 out_dim=512,
                 position_encoding=None):
        super(Sampler, self).__init__()

        self.in_size = input_size
        self.p_size = patch_size
        self.p_num = num_patch if num_patch is not None else ((input_size - patch_size) // stride + 1) ** 2
        self.stride = stride if stride is not None else patch_size

        self.th = threshold
        self.sample_dim = patch_size ** 2 * in_channel
        self.dim = out_dim
        self.pos_enc = position_encoding

        # self.to_patch_embedding = nn.Linear(self.sample_dim, out_dim)
        # self.norm = nn.LayerNorm(out_dim)
        # self._reset_parameters()

    def forward(self, nt: NestedTensor):
        x, mask = nt.decompose()
        pos = None if self.pos_enc is None else self.pos_enc(NestedTensor(x, None))

        b, c, h, w = x.size()
        p = self.p_size
        n = self.p_num
        d = self.dim

        # stride = [self.stride] if not isinstance(self.stride, (list, tuple)) else self.stride
        # x_unfold = []
        # mask_unfold = []
        # pos_unfold = []

        # x, mask: (b,c,h,w) -> (b,c*p*p,((h-p)/s+1)**2)
        # for s in stride:
        #     x_unfold.append(F.unfold(x, kernel_size=self.p_size, dilation=1, stride=s, padding=0))
        #     mask_unfold.append(F.unfold(mask, kernel_size=self.p_size, dilation=1, stride=s, padding=0))
        #     pos_unfold.append(F.unfold(pos, kernel_size=self.p_size, dilation=1, stride=s, padding=0))
        # x_unfold = torch.cat(x_unfold, dim=-1)
        # mask_unfold = torch.cat(mask_unfold, dim=-1)
        # pos_unfold = torch.cat(pos_unfold, dim=-1)

        s = self.stride
        x_unfold = F.unfold(x, kernel_size=self.p_size, dilation=1, stride=s, padding=0)
        mask_unfold = F.unfold(mask, kernel_size=self.p_size, dilation=1, stride=s, padding=0)
        pos_unfold = F.unfold(pos, kernel_size=self.p_size, dilation=1, stride=s, padding=0)
        mask_idx = torch.mean(mask_unfold, dim=1, keepdim=False).squeeze(dim=1)
        mask_idx = (mask_idx >= self.th).to(dtype=torch.float)
        idx = torch.sort(torch.multinomial(mask_idx, n)).values
        idx_x = torch.unsqueeze(idx, dim=1).expand(b, c * p * p, n)
        idx_pos = torch.unsqueeze(idx, dim=1).expand(b, d, n)
        x_sample = torch.gather(x_unfold, dim=-1, index=idx_x)
        pos_sample = torch.gather(pos_unfold, dim=-1, index=idx_pos)
        x_sample = x_sample.permute(0, 2, 1)
        pos_sample = pos_sample.permute(0, 2, 1)

        # x_sample = self.to_patch_embedding(x_sample)
        # x_sample = self.norm(x_sample)
        # x, pos: (b, n, dim)
        # assert x_sample.size() == pos_sample.size()
        return x_sample, pos_sample

    def _reset_parameters(self):  # init weight with xaiver_uniform
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


def build_sampler(config):
    pos_enc = build_position_encoding(config)
    model = Sampler(input_size=256,
                    patch_size=8,
                    stride=8,
                    num_patch=160,
                    in_channel=3,
                    out_dim=512,
                    threshold=1.0,
                    position_encoding=pos_enc)
    return model

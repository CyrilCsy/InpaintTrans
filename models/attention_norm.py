import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionNormalization(nn.Module):
    def __init__(self, dim=512, nClass=16, orth_lambda=1e-3):
        super(AttentionNormalization, self).__init__()
        self.entity_filters = nn.Parameter(torch.randn(nClass, dim, 1, 1), requires_grad=True)
        # self.alpha = nn.Parameter(torch.ones(size=(1, nClass, 1, 1)) * 0.1, requires_grad=True)
        self.sigma = nn.Parameter(torch.ones([1]))

        self.nClass = nClass
        self.orth_lambda = orth_lambda

    def forward(self, x, sem_embed=None):
        b, c, h, w = x.size()
        n = self.nClass
        entity_mask = F.conv2d(sem_embed, self.entity_filters, stride=1)    # (b,n,h,w)
        # 原版不仅使用entity计算softmax
        entity_mask = torch.softmax(entity_mask, dim=1).view(b, 1, n, h, w)
        x_expand = x.view(b, c, 1, h, w).repeat(1, 1, n, 1, 1)
        hot_area = x_expand * entity_mask
        cnt = torch.sum(entity_mask, [-1, -2], keepdim=True) + 1e-7
        mean = torch.mean(hot_area, [-1, -2], keepdim=True) / cnt
        std = torch.sqrt(torch.sum((hot_area - mean) ** 2, [-1, -2], keepdim=True) / cnt)
        xn = torch.sum((x_expand - mean) / (std + self.eps) * entity_mask, dim=2)
        x = x + self.sigma * xn

        # orth_loss
        f_w = torch.reshape(self.entity_filters.permute(2, 3, 0, 1), [1, n, c])
        f_w_s = torch.bmm(f_w, f_w.permute(0, 2, 1))
        orth_loss = f_w_s - cuda(torch.eye(n))  # (b,n,n)
        orth_loss = self.orth_lambda * torch.sum(torch.mean(orth_loss, dim=0), dim=(0, 1))

        return x, orth_loss


def cuda(x, use_gpu=True):
    if use_gpu:
        return x.cuda()
    else:
        return x
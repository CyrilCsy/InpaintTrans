import torch.nn as nn
import torch.nn.functional as F


class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    def init_weights(self, init_type='kaiming', gain=0.02, scale=1):
        '''
        initialize module's weights
        init_type: normal | xavier | kaiming | orthogonal
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                # m.weight.data *= scale  # for residual block  #from RN

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):  # hidden_dim, hidden_dim, 4, 3
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            layer = layer.cuda()
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ResnetBlock(BaseModule):
    def __init__(self, dim, n=8, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.res_blocks = []
        for _ in range(n):
            res_block = nn.Sequential(
                nn.ReflectionPad2d(dilation),
                spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation,
                                        bias=not use_spectral_norm), use_spectral_norm),
                nn.InstanceNorm2d(dim, track_running_stats=False),
                nn.ReLU(True),

                nn.ReflectionPad2d(1),
                spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1,
                                        bias=not use_spectral_norm), use_spectral_norm),
                nn.InstanceNorm2d(dim, track_running_stats=False),
            )
            self.res_blocks.append(res_block)
        self.init_weights()

    def forward(self, x):
        for res in self.res_blocks:
            out = x + res(x)
            x = out
        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

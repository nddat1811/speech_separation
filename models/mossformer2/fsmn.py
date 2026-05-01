import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.nn.parameter import Parameter
import numpy as np
import os

class UniDeepFsmn(nn.Module):

    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None):
        super(UniDeepFsmn, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        if lorder is None:
            return
        self.lorder = lorder
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_dim, hidden_size)
        self.project = nn.Linear(hidden_size, output_dim, bias=False)
        self.conv1 = nn.Conv2d(output_dim, output_dim, [lorder+lorder-1, 1], [1, 1], groups=output_dim, bias=False)

    def forward(self, input):
        f1 = F.relu(self.linear(input))
        p1 = self.project(f1)
        x = th.unsqueeze(p1, 1)
        x_per = x.permute(0, 3, 2, 1)
        y = F.pad(x_per, [0, 0, self.lorder - 1, self.lorder - 1])
        out = x_per + self.conv1(y)
        out1 = out.permute(0, 3, 2, 1)
        return input + out1.squeeze()

class UniDeepFsmn_dual(nn.Module):

    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None):
        super(UniDeepFsmn_dual, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        if lorder is None:
            return
        self.lorder = lorder
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_dim, hidden_size)
        self.project = nn.Linear(hidden_size, output_dim, bias=False)
        self.conv1 = nn.Conv2d(output_dim, output_dim, [lorder+lorder-1, 1], [1, 1], groups=output_dim, bias=False)
        self.conv2 = nn.Conv2d(output_dim, output_dim, [lorder+lorder-1, 1], [1, 1], groups=output_dim//4, bias=False)

    def forward(self, input):

        f1 = F.relu(self.linear(input))
        p1 = self.project(f1)
        x = th.unsqueeze(p1, 1)
        x_per = x.permute(0, 3, 2, 1)
        y = F.pad(x_per, [0, 0, self.lorder - 1, self.lorder - 1])
        conv1_out = x_per + self.conv1(y)
        z = F.pad(conv1_out, [0, 0, self.lorder - 1, self.lorder - 1])
        out = conv1_out + self.conv2(z)
        out1 = out.permute(0, 3, 2, 1)
        return input + out1.squeeze()

class DilatedDenseNet(nn.Module):
    def __init__(self, depth=4, lorder=20, in_channels=64):
        super(DilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = lorder*2-1
        self.kernel_size = (self.twidth, 1)
        for i in range(self.depth):
            dil = 2 ** i
            pad_length = lorder + (dil - 1) * (lorder - 1) - 1
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((0, 0, pad_length, pad_length), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    nn.Conv2d(self.in_channels*(i+1), self.in_channels, kernel_size=self.kernel_size,
                              dilation=(dil, 1), groups=self.in_channels, bias=False))
            setattr(self, 'norm{}'.format(i + 1), nn.InstanceNorm2d(in_channels, affine=True))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)            
            skip = th.cat([out, skip], dim=1)
        return out

class UniDeepFsmn_dilated(nn.Module):

    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None, depth=2):
        super(UniDeepFsmn_dilated, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth=depth
        if lorder is None:
            return
        self.lorder = lorder
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_dim, hidden_size)
        self.project = nn.Linear(hidden_size, output_dim, bias=False)
        self.conv = DilatedDenseNet(depth=self.depth, lorder=lorder, in_channels=output_dim)

    def forward(self, input):
        f1 = F.relu(self.linear(input))
        p1 = self.project(f1)
        x = th.unsqueeze(p1, 1)
        x_per = x.permute(0, 3, 2, 1)
        out = self.conv(x_per)
        out1 = out.permute(0, 3, 2, 1)

        return input + out1.squeeze()

class LightDilatedNet1(nn.Module):
    """Thay DilatedDenseNet - bỏ dense concat, dùng Conv1d, residual thay concat"""
    def __init__(self, depth=4, lorder=20, in_channels=64):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.acts = nn.ModuleList()
        
        for i in range(depth):
            dil = 2 ** i
            pad = (lorder * 2 - 1 - 1) * dil // 2  # symmetric, Conv1d
            self.layers.append(
                nn.Conv1d(in_channels, in_channels,
                          kernel_size=lorder * 2 - 1,
                          dilation=dil,
                          padding=pad,
                          groups=in_channels,  # depthwise
                          bias=False)
            )
            self.norms.append(nn.GroupNorm(1, in_channels))
            self.acts.append(nn.PReLU(in_channels))

    def forward(self, x):
        # x: (B, C, T) - Conv1d trực tiếp, không cần permute
        out = x
        for i in range(self.depth):
            residual = out
            out = self.layers[i](out)
            out = self.norms[i](out)
            out = self.acts[i](out)
            out = out + residual  # residual thay vì dense concat
        return out

class LightDilatedNet(nn.Module):
    def __init__(self, depth=4, lorder=20, in_channels=64):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.acts = nn.ModuleList()

        for i in range(depth):
            dil = 2 ** i
            pad = ((lorder * 2 - 1) - 1) * dil // 2
            self.layers.append(
                nn.Conv1d(
                    in_channels * (i + 1),  # ← giữ đúng như gốc: C, 2C, 3C, 4C
                    in_channels,             # ← output luôn C
                    kernel_size=lorder * 2 - 1,
                    dilation=dil,
                    padding=pad,
                    groups=in_channels,      # depthwise trên C channels
                    bias=False
                )
            )
            self.norms.append(nn.GroupNorm(1, in_channels))
            self.acts.append(nn.PReLU(in_channels))

    def forward(self, x):
        # x: (B, C, T)
        skip = x
        for i in range(self.depth):
            out = self.layers[i](skip)       # (B, C, T)
            out = self.norms[i](out)
            out = self.acts[i](out)
            skip = th.cat([out, skip], dim=1)  # ← dense concat đúng như gốc
        return out  # trả về output layer cuối, shape (B, C, T)

class UniDeepFsmn_dilated_light(nn.Module):
    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None, depth=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        if lorder is None:
            return
        self.lorder = lorder
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_dim, hidden_size)
        self.project = nn.Linear(hidden_size, output_dim, bias=False)
        # Thay DilatedDenseNet → LightDilatedNet
        self.conv = LightDilatedNet(depth=depth, lorder=lorder, in_channels=output_dim)

    def forward(self, input):
        f1 = F.relu(self.linear(input))
        p1 = self.project(f1)
        # Bỏ hoàn toàn unsqueeze/permute vì dùng Conv1d
        out = self.conv(p1.transpose(1, 2))   # (B, C, T)
        out = out.transpose(1, 2)             # (B, T, C)
        return input + out


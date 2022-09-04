import torch.nn.functional as F
from torch.nn.modules.utils import _quadruple
import torch.nn as nn
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch import nn
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import time
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.functional import unfold, pad
import torch
from torch import nn
from torch.nn.parameter import Parameter
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as Func
from torch.nn.modules.utils import _quadruple

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 80,1,5,5
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 80,1,5,5
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 80,1,5,5
        return self.sigmoid(x)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=10, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )

        self.pool_types = pool_types
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # 50，640，1，1
                channel_att_raw = self.mlp(avg_pool)  # 50，640
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # 50，640，1，1
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3)  # 50，640，1，1
        return scale

# class SCR(nn.Module):
#     def __init__(self, planes=[640, 64, 64, 64, 640], stride=(1, 1, 1), ksize=3, do_padding=False, bias=False):
#         super(SCR, self).__init__()
#         self.ksize = _quadruple(ksize) if isinstance(ksize, int) else ksize
#         padding1 = (0, self.ksize[2] // 2, self.ksize[3] // 2) if do_padding else (0, 0, 0)
#
#         self.conv1x1_in = nn.Sequential(nn.Conv2d(planes[0], planes[1], kernel_size=1, bias=False, padding=0),
#                                         nn.BatchNorm2d(planes[1]),
#                                         nn.ReLU(inplace=True))
#         self.conv1 = nn.Sequential(nn.Conv3d(planes[1], planes[2], (1, self.ksize[2], self.ksize[3]),
#                                              stride=stride, bias=bias, padding=padding1),
#                                    nn.BatchNorm3d(planes[2]),
#                                    nn.ReLU(inplace=True))
#         self.conv2 = nn.Sequential(nn.Conv3d(planes[2], planes[3], (1, self.ksize[2], self.ksize[3]),
#                                              stride=stride, bias=bias, padding=padding1),
#                                    nn.BatchNorm3d(planes[3]),
#                                    nn.ReLU(inplace=True))
#         self.conv1x1_out = nn.Sequential(
#             nn.Conv2d(planes[3], planes[4], kernel_size=1, bias=False, padding=0),
#             nn.BatchNorm2d(planes[4]))
#
#         self.ChannelGate = ChannelGate(640)
#         self.ChannelAttention = ChannelAttention(640)
#     def forward(self, x):
#         b, c, h, w, u, v = x.shape
#         x = x.view(b, c, h * w, u * v)
#
#         x = self.conv1x1_in(x)   # [80, 640, hw, 25] -> [80, 64, HW, 25]
#
#         c = x.shape[1]
#         x = x.view(b, c, h * w, u, v)
#         x = self.conv1(x)  # [80, 64, hw, 5, 5] --> [80, 64, hw, 3, 3]
#         x = self.conv2(x)  # [80, 64, hw, 3, 3] --> [80, 64, hw, 1, 1]
#
#         c = x.shape[1]
#         x = x.view(b, c, h, w)
#         x = self.conv1x1_out(x)  # [80, 64, h, w] --> [80, 640, h, w]
#         return x
#
#
# class SelfCorrelationComputation(nn.Module):
#     def __init__(self, kernel_size=(5, 5), padding=2):
#         super(SelfCorrelationComputation, self).__init__()
#         self.kernel_size = kernel_size
#         self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding)
#         self.relu = nn.ReLU(inplace=True)
#         conv_nd = nn.Conv2d
#         bn = nn.BatchNorm2d
#         self.in_channels = 640
#         self.g = conv_nd(in_channels=640, out_channels=64,
#                          kernel_size=1, stride=1, padding=0)
#         self.theta = conv_nd(in_channels=640, out_channels=64,
#                              kernel_size=1, stride=1, padding=0)
#         self.phi = conv_nd(in_channels=640, out_channels=64,
#                            kernel_size=1, stride=1, padding=0)
#         self.Q = nn.Sequential(
#             conv_nd(in_channels=64, out_channels=640,
#                     kernel_size=1, stride=1, padding=0),
#             bn(self.in_channels)
#         )
#         self.ChannelGate = ChannelGate(640)
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#
#         x = self.relu(x)
#         x = F.normalize(x, dim=1, p=2)
#         identity = x
#
#         x1 = self.g(identity).view(b, 64, -1)
#         x1 = x1.permute(0, 2, 1)
#         theta_x = self.theta(x).view(b,64, -1)
#         theta_x = theta_x.permute(0, 2, 1)
#         phi_x = self.phi(x).view(b, 64, -1)
#         f = torch.matmul(theta_x, phi_x)
#         f_div_C = F.softmax(f, dim=-1)
#         y = torch.matmul(f_div_C, x1)
#         y = y.permute(0,2,1).contiguous()
#
#         y = y.view(b, 64, h, w)
#         y = self.Q(y)
#         identity = identity + y
#
#         x = self.unfold(x)  # 提取出滑动的局部区域块，这里滑动窗口大小为5*5，步长为1
#         # b, cuv, h, w  （80,640*5*5,5,5)
#         x = x.view(b, c, self.kernel_size[0], self.kernel_size[1], h, w)  # b, c, u, v, h, w
#         x = x * identity.unsqueeze(2).unsqueeze(2)  # 通过unsqueeze增维使identity和x变为同维度  公式（1）
#         # b, c, u, v, h, w * b, c, 1, 1, h, w
#         x = x.permute(0, 1, 4, 5, 2, 3).contiguous()  # b, c, h, w, u, v
#         # torch.contiguous()方法首先拷贝了一份张量在内存中的地址，然后将地址按照形状改变后的张量的语义进行排列
#         return x

class SCR(nn.Module):
    def __init__(self, planes=[640, 64, 64, 64, 640], stride=1, ksize=3, do_padding=False, bias=False):
        super(SCR, self).__init__()
        self.ksize = _quadruple(ksize) if isinstance(ksize,
                                                     int) else ksize  # 4倍 isinstance() 函数来判断一个对象是否是一个已知的类型（是否与int一个类型）
        padding1 = (0, self.ksize[2] // 2, self.ksize[3] // 2) if do_padding else (0, 0, 0)

        self.conv1x1_in = nn.Sequential(nn.Conv2d(planes[0], planes[1], kernel_size=1, bias=False, padding=0),
                                        nn.BatchNorm2d(planes[1]),
                                        nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv3d(planes[1], planes[2], (1, self.ksize[2], self.ksize[3]),
                                             stride=stride, bias=bias, padding=padding1),
                                   nn.BatchNorm3d(planes[2]),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv3d(planes[2], planes[3], (1, self.ksize[2], self.ksize[3]),
                                             stride=stride, bias=bias, padding=padding1),
                                   nn.BatchNorm3d(planes[3]),
                                   nn.ReLU(inplace=True))
        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(planes[3], planes[4], kernel_size=1, bias=False, padding=0),
            nn.BatchNorm2d(planes[4]))
        self.ChannelGate = ChannelGate(640)


    def forward(self, x):
        b, c, h, w, u, v = x.shape
        x = x.view(b, c, h * w, u * v)

        x = self.conv1x1_in(x)  # [80, 640, hw, 25] -> [80, 64, hw, 25]

        c = x.shape[1]
        x = x.view(b, c, h * w, u, v)
        x = self.conv1(x)  # [80, 64, hw, 5, 5] --> [80, 64, hw, 3, 3]
        x = self.conv2(x)  # [80, 64, hw, 3, 3] --> [80, 64, hw, 1, 1]

        c = x.shape[1]
        x = x.view(b, c, h, w)
        x = self.conv1x1_out(x)  # [80, 64, h, w] --> [80, 640, h, w]t
        return x

class SelfCorrelationComputation(nn.Module):
    def __init__(self, kernel_size=(5, 5), padding=2):
        super(SelfCorrelationComputation, self).__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.relu(x)
        x = F.normalize(x, dim=1, p=2)
        identity = x

        x = self.unfold(x)  # b, cuv, h, w
        x = x.view(b, c, self.kernel_size[0], self.kernel_size[1], h, w)
        x = x * identity.unsqueeze(2).unsqueeze(2)  # b, c, u, v, h, w * b, c, 1, 1, h, w
        x = x.permute(0, 1, 4, 5, 2, 3).contiguous()  # b, c, h, w, u, v
        return x
    
class mySelfCorrelationComputation(nn.Module):
    def __init__(self, kernel_size=(5, 5), padding=2):
        super(mySelfCorrelationComputation, self).__init__()
        planes =[640, 64, 64, 640]
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=False)

        self.conv1x1_in = nn.Sequential(nn.Conv2d(planes[0], planes[1], kernel_size=1, bias=False, padding=0),
                                        nn.BatchNorm2d(planes[1]),
                                        nn.ReLU(inplace=True))
        self.embeddingFea = nn.Sequential(nn.Conv2d(1664, 640,
                                                     kernel_size=1, bias=False, padding=0),
                                           nn.BatchNorm2d(640),
                                           nn.ReLU(inplace=True))
        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(640, 640, kernel_size=1, bias=False, padding=0),
            nn.BatchNorm2d(640))

    def forward(self, x):

        x = self.conv1x1_in(x)
        b, c, h, w = x.shape

        x0 = self.relu(x)
        x = x0
        x = F.normalize(x, dim=1, p=2)
        identity = x

        x = self.unfold(x)  # 提取出滑动的局部区域块，这里滑动窗口大小为5*5，步长为1
        # b, cuv, h, w  （80,640*5*5,5,5)
        x = x.view(b, c, self.kernel_size[0], self.kernel_size[1], h, w)  # b, c, u, v, h, w
        x = x * identity.unsqueeze(2).unsqueeze(2)  # 通过unsqueeze增维使identity和x变为同维度  公式（1）
        # b, c, u, v, h, w * b, c, 1, 1, h, w
        x = x.view(b, -1, h, w)
        # x = x.permute(0, 1, 4, 5, 2, 3).contiguous()  # b, c, h, w, u, v
        # torch.contiguous()方法首先拷贝了一份张量在内存中的地址，然后将地址按照形状改变后的张量的语义进行排列
        # x = x.mean(dim=[-1, -2])
        feature_gs = featureL2Norm(x)

        # concatenate
        feature_cat = torch.cat([identity, feature_gs], 1)

        # embed
        feature_embd = self.embeddingFea(feature_cat)
        feature_embd = self.conv1x1_out(feature_embd)
        return feature_embd

class SelfCorrelationComputation1(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SelfCorrelationComputation1, self).__init__()

        self.d_model = d_model
        self.d_k = d_model//h
        self.d_v = d_model//h
        self.h = h

        self.fc_o = nn.Linear(h * self.d_v, d_model)
        self.dropout=nn.Dropout(dropout)



        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self,x, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        x = x.permute(0, 2, 3, 1).contiguous()  # （10，5，5，640）
        B, H, W, C = x.shape
        N = H * W
        x = x.view(B,N,C)
        queries = x
        keys = x
        values = x
        b_s, nq = queries.shape[:2]   # (50,49)
        nk = keys.shape[1]

        q = queries.view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k) (50,8,49,64)
        k = keys.view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)  (50,8,64,49)
        v = values.view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v) (50,8,49,64)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)   #(50,8,49,49) # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v) (50,49,512)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        out = out.view(B,C,H,W)
        return out





class SelfCorrelationComputation2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(SelfCorrelationComputation2, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.agg = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels))

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)
        out = self.agg(out)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)





class SelfCorrelationComputation3(nn.Module):  # nonl
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True):
        super(SelfCorrelationComputation3, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

class SelfCorrelationComputation4(nn.Module):  # se
    def __init__(self, channel, reduction=16):
        super(SelfCorrelationComputation4, self).__init__()
        hdim = 64
        self.conv1x1_in = nn.Sequential(nn.Conv2d(channel, hdim, kernel_size=1, bias=False, padding=0),
                                        nn.BatchNorm2d(hdim),
                                        nn.ReLU(inplace=False))
        self.conv1x1_out = nn.Sequential(nn.Conv2d(hdim, channel, kernel_size=1, bias=False, padding=0),
                                        nn.BatchNorm2d(channel),
                                        nn.ReLU(inplace=False))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(hdim, hdim // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(hdim // reduction, hdim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1x1_in(x)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x = x * y.expand_as(x)
        x = self.conv1x1_out(x)
        return x



class SelfCorrelationComputation5(nn.Module):  # Gam
    def __init__(self, in_channels, out_channels, rate=4):
        super(SelfCorrelationComputation5, self).__init__()

        self.fc1 = nn.Conv2d(in_channels, in_channels // rate, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // rate, in_channels, 1, bias=False)

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)  # 15 25 640
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)  # 15 5 5 640
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)   # 15，640，5，5
        # x_channel_att = self.fc2(self.relu1(self.fc1(x)))

        x = x * x_channel_att  # 15，640，5，5

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out




def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


class SelfCorrelationComputation6(nn.Module):  # local
    def __init__(self, in_planes, out_planes, kernel_att=5, head=1, kernel_conv=3, stride=1, dilation=1):
        super(SelfCorrelationComputation6, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1,
                                  stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w = q.shape
        h_out, w_out = h // self.stride, w // self.stride

        # ### att
        # ## positional encoding
        pe = self.conv_p(position(h, w, x.is_cuda))

        q_att = q.view(b * self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b * self.head, self.head_dim, h, w)
        v_att = v.view(b * self.head, self.head_dim, h, w)

        if self.stride > 1:
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe

        unfold_k = self.unfold(self.pad_att(k_att)).view(b * self.head, self.head_dim,
                                                         self.kernel_att * self.kernel_att, h_out,
                                                         w_out)  # b*head, head_dim, k_att^2, h_out, w_out
        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att * self.kernel_att, h_out,
                                                        w_out)  # 1, head_dim, k_att^2, h_out, w_out

        att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(
            1)  # (b*head, head_dim, 1, h_out, w_out) * (b*head, head_dim, k_att^2, h_out, w_out) -> (b*head, k_att^2, h_out, w_out)
        att = self.softmax(att)

        out_att = self.unfold(self.pad_att(v_att)).view(b * self.head, self.head_dim, self.kernel_att * self.kernel_att,
                                                        h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)

        ## conv
        f_all = self.fc(torch.cat(
            [q.view(b, self.head, self.head_dim, h * w), k.view(b, self.head, self.head_dim, h * w),
             v.view(b, self.head, self.head_dim, h * w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

        out_conv = self.dep_conv(f_conv)

        return self.rate1 * out_att + self.rate2 * out_conv


class SelfCorrelationComputation10(nn.Module):  # nat
    def __init__(self, dim, kernel_size=3, num_heads=1,planes=[640, 64, 64, 64, 640],
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 mode=1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        self.win_size = kernel_size // 2
        self.mid_cell = kernel_size - 1
        self.rpb_size = 2 * kernel_size - 1

        self.relu = nn.ReLU(inplace=True)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode = mode
        self.rpb = nn.Parameter(torch.zeros(num_heads, self.rpb_size, self.rpb_size))
        trunc_normal_(self.rpb, std=.02)
        # RPB implementation by @qwopqwop200
        self.idx_h = torch.arange(0, kernel_size)
        self.idx_w = torch.arange(0, kernel_size)
        self.idx_k = ((self.idx_h.unsqueeze(-1) * self.rpb_size) + self.idx_w).view(-1)
        warnings.warn("This is the legacy version of NAT -- it uses unfold+pad to produce NAT, and is highly inefficient.")
        self.bn1 = nn.BatchNorm2d(640)
        self.conv1x1_in = nn.Sequential(nn.Conv2d(planes[0], planes[1], kernel_size=1, bias=False, padding=0),
                                        nn.BatchNorm2d(planes[1]),
                                        nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(planes[1], planes[2], kernel_size=3, bias=False, padding=0),
                                        nn.BatchNorm2d(planes[2]),
                                        nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(planes[2], planes[3],kernel_size=3, bias=False, padding=0),
                                   nn.BatchNorm2d(planes[3]),
                                   nn.ReLU(inplace=True))
        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(planes[3], planes[4], kernel_size=1, bias=False, padding=0),
            nn.BatchNorm2d(planes[4]))


    def apply_pb(self, attn, height, width):

        num_repeat_h = torch.ones(self.kernel_size,dtype=torch.long)
        num_repeat_w = torch.ones(self.kernel_size,dtype=torch.long)
        num_repeat_h[self.kernel_size//2] = height - (self.kernel_size-1)
        num_repeat_w[self.kernel_size//2] = width - (self.kernel_size-1)
        bias_hw = (self.idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * (2*self.kernel_size-1)) + self.idx_w.repeat_interleave(num_repeat_w)
        bias_idx = bias_hw.unsqueeze(-1) + self.idx_k
        # Index flip
        # Our RPB indexing in the kernel is in a different order, so we flip these indices to ensure weights match.
        bias_idx = torch.flip(bias_idx.reshape(-1, self.kernel_size**2), [0])
        return attn + self.rpb.flatten(1, 2)[:, bias_idx].reshape(self.num_heads, height * width, 1, self.kernel_size ** 2).transpose(0, 1)

    def forward(self, x):
        x = self.relu(x)
        x = F.normalize(x, dim=1, p=2)  # （10，640，5，5）
        x = x.permute(0, 2, 3, 1).contiguous()  # （10，5，5，640）
        # x = self.norm1(x)
        B, H, W, C = x.shape
        N = H * W
        num_tokens = int(self.kernel_size ** 2)  # 49
        pad_l = pad_t = pad_r = pad_b = 0
        Ho, Wo = H, W
        if N <= num_tokens:
            if self.kernel_size > W:
                pad_r = self.kernel_size - W
            if self.kernel_size > H:
                pad_b = self.kernel_size - H
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))  # （10，7，7，640）
            B, H, W, C = x.shape
            N = H * W
            assert N == num_tokens, f"Something went wrong. {N} should equal {H} x {W}!"
        x = self.qkv(x).reshape(B, H, W, 3 * C)  # (80,7,7,1920)
        q, x = x[:, :, :, :C], x[:, :, :, C:]
        q = q.reshape(B, N, self.num_heads, C // self.num_heads, 1).transpose(3, 4) * self.scale  # (80,25,1,1,640)
        pd = self.kernel_size - 1
        pdr = pd // 2

        if self.mode == 0:
            x = x.permute(0, 3, 1, 2).flatten(0, 1)
            x = x.unfold(1, self.kernel_size, 1).unfold(2, self.kernel_size, 1).permute(0, 3, 4, 1, 2)
            x = pad(x, (pdr, pdr, pdr, pdr, 0, 0), 'replicate')
            x = x.reshape(B, 2, self.num_heads, C // self.num_heads, num_tokens, N)
            x = x.permute(1, 0, 5, 2, 4, 3)
        elif self.mode == 1:
            Hr, Wr = H - pd, W - pd
            x = unfold(x.permute(0, 3, 1, 2),
                       kernel_size=(self.kernel_size, self.kernel_size),
                       stride=(1, 1),
                       padding=(0, 0)).reshape(B, 2 * C * num_tokens, Hr, Wr)
            x = pad(x, (pdr, pdr, pdr, pdr), 'replicate').reshape(
                B, 2, self.num_heads, C // self.num_heads, num_tokens, N)
            x = x.permute(1, 0, 5, 2, 4, 3)  # (2,80,25,1,9,640)
        else:
            raise NotImplementedError(f'Mode {self.mode} not implemented for NeighborhoodAttention2D.')
        k, v = x[0], x[1]

        attn = (q @ k.transpose(-2, -1))  # B x N x H x 1 x num_tokens
        attn = self.apply_pb(attn, H, W)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)  # B x N x H x 1 x C # (80,25,1,1,640)
        x = x.reshape(B, H, W, C)  # (10，7，7，640)
        if pad_r or pad_b:
            x = x[:, :Ho, :Wo, :]
        x = self.proj_drop(self.proj(x))
        x = x.permute(0, 3, 1, 2).contiguous()  # （10，640，5，5）
        return x


class SelfCorrelationComputation8(nn.Module):  # NAM
    def __init__(self, channels=640):
        super(SelfCorrelationComputation8, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x
        x = self.bn2(x)
        # 式2的计算，即Mc的计算
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual  #

        return x

class SelfCorrelationComputation9(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SelfCorrelationComputation9, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class SelfCorrelationComputation12(nn.Module): # cbam
    def __init__(self, channel, ratio=8, kernel_size=3):
        super(SelfCorrelationComputation12, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x



def generate_spatial_descriptor(data, kernel_size):
    '''
    Applies self local similarity with fixed sliding window.
    Args:
        data: featuren map, variable of shape (b,c,h,w)
        kernel_size: width/heigh of local window, int
    Returns:
        output: global spatial map, variable of shape (b,c,h,w)
    '''

    padding = int(kernel_size // 2)  # 5.7//2 = 2.0, 5.2//2 = 2.0
    b, c, h, w = data.shape
    p2d = _quadruple(padding)  # (pad_l,pad_r,pad_t,pad_b)
    data_padded = Func.pad(data, p2d, 'constant', 0)  # output variable
    assert data_padded.shape == (
    b, c, (h + 2 * padding), (w + 2 * padding)), 'Error: data_padded shape{} wrong!'.format(data_padded.shape)

    output = torch.zeros(size=[b, kernel_size * kernel_size, h, w], requires_grad=data.requires_grad)  # 80,25,5,5
    if data.is_cuda:
        output = output.cuda(data.get_device())

    for hi in range(h):
        for wj in range(w):
            q = data[:, :, hi, wj].contiguous()  # (b,c)
            i = hi + padding  # h index in datapadded
            j = wj + padding  # w index in datapadded

            hs = i - padding  # 0,5,0,5
            he = i + padding + 1
            ws = j - padding
            we = j + padding + 1
            patch = data_padded[:, :, hs:he, ws:we].contiguous()  # (b,c,k,k)
            assert (patch.shape == (b, c, kernel_size, kernel_size))
            hk, wk = kernel_size, kernel_size

            # reshape features for matrix multiplication
            feature_a = q.view(b, c, 1 * 1).transpose(1, 2)  # (b,1,c) input is not contigous
            feature_b = patch.view(b, c, hk * wk)  # (b,c,L)

            # perform matrix mult.
            feature_mul = torch.bmm(feature_a, feature_b)  # (b,1,L)
            assert (feature_mul.shape == (b, 1, hk * wk))
            # indexed [batch,row_A,col_A,row_B,col_B]
            correlation_tensor = feature_mul.unsqueeze(1)  # (b,L)
            output[:, :, hi, wj] = correlation_tensor.squeeze(1).squeeze(1)

    return output


def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature, norm)

class channelatt(nn.Module):  # se
    def __init__(self, channel, reduction=25):
        super(channelatt, self).__init__()
        hdim = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(hdim, hdim // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(hdim // reduction, hdim, bias=False),
            nn.Sigmoid()
        )

        self.ChannelGate = ChannelGate(640)
        self.ChannelAttention = ChannelAttention(640)
        self.SpatialAttention = SpatialAttention()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x = x * y.expand_as(x)
        return x
class SpatialContextEncoder(torch.nn.Module):
    '''
    Spatial Context Encoder.
    Author: Shuaiyi Huang
    Input:
        x: feature of shape (b,c,h,w)
    Output:
        feature_embd: context-aware semantic feature of shape (b,c+k**2,h,w), where k is the kernel size of spatial descriptor
    '''

    def __init__(self, planes=None, kernel_size=None):
        super(SpatialContextEncoder, self).__init__()
        self.kernel_size = kernel_size
        self.conv1x1_in = nn.Sequential(nn.Conv2d(planes[0], planes[1], kernel_size=1, bias=False, padding=0),
                                        nn.BatchNorm2d(planes[1]),
                                        nn.ReLU(inplace=True))
        self.embeddingFea1 = nn.Sequential(nn.Conv2d(1664, planes[0],
                                                    kernel_size=1, bias=False, padding=0),
                                          nn.BatchNorm2d(planes[0]),
                                          nn.ReLU(inplace=True))
        self.embeddingFea2 = nn.Sequential(nn.Conv2d(640, planes[2],
                                                    kernel_size=1, bias=False, padding=0),
                                          nn.BatchNorm2d(planes[2]),
                                          nn.ReLU(inplace=True))
        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(planes[2], planes[3], kernel_size=1, bias=False, padding=0),
            nn.BatchNorm2d(planes[3]))
        self.channels = channelatt(640)
        self.ChannelGate = ChannelGate(640)
        self.kernel_size1 = (5, 5)
        self.unfold = nn.Unfold(kernel_size=(5, 5), padding=2)
        self.relu = nn.ReLU(inplace=True)

        print('SpatialContextEncoder initialization: input_dim {},hidden_dim {}'.format(planes[1], planes[2]))
        return

    def forward(self, x,ide):
        ide = self.conv1x1_in(ide)
        x = self.conv1x1_in(x)
        # feature_gs = generate_spatial_descriptor(x, kernel_size=self.kernel_size)
        # Add L2norm
        feature_gs = featureL2Norm(x)

        # concatenate
        feature_cat = torch.cat([ide, feature_gs], 1)
        feature_embd = self.embeddingFea(feature_cat)
        # embed
        feature_embd1 = self.embeddingFea1(feature_cat)
        feature_embd = self.embeddingFea2(feature_embd1)
        # channel expansion
        feature_embd = self.conv1x1_out(feature_embd)

        return feature_embd

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes=640, ratio=10):
        super(ChannelAttention, self).__init__()
        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        avg_out = self.fc2(self.relu1(self.fc1(avg_pool)))
        max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        max_out = self.fc2(self.relu1(self.fc1(max_pool)))
        out = avg_out + max_out
        return self.sigmoid(out)
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
class match_block(nn.Module):
    def __init__(self, inplanes):
        super(match_block, self).__init__()
        self.in_channels = inplanes
        self.inter_channels = None
        self.ChannelAttention = ChannelAttention(self.in_channels)
        self.SpatialAttention = SpatialAttention()
    def forward(self, spt, qry):

        way = spt.shape[0]
        num_qry = qry.shape[0]
        spt = F.normalize(spt, p=2, dim=1, eps=1e-8)
        qry = F.normalize(qry, p=2, dim=1, eps=1e-8)
        # way , C , H_s , W_s --> num_qry * way, C , H_s , W_s
        # num_qry , C , H_q , W_q --> num_qry * way,C ,H_q , W_q
        spt = spt.unsqueeze(0).repeat(num_qry, 1, 1, 1, 1)
        spt = spt.view(-1, 640,5,5) # num_qry * way, C , H_s , W_s
        qry = qry.unsqueeze(1).repeat(1, way, 1, 1, 1)
        qry = qry.view(-1, 640,5,5) # num_qry * way,C ,H_q , W_q
        c_weight1 = self.ChannelAttention(spt)
        c_weight2 = self.ChannelAttention(qry)
        xq = qry * c_weight1
        xs = spt * c_weight2
        xq0 = self.SpatialAttention(xq)
        xs0 = self.SpatialAttention(xs)
        x1 = xq * xq0 + qry
        x2 = xs * xs0 + spt
        return x2, x1

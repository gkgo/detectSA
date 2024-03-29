import torch.nn as nn

# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).


import torch.nn.functional as F
import torch
import torch.nn as nn

def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature, norm)

class mySelfCorrelationComputation(nn.Module):
    def __init__(self,channel, kernel_size=(5, 5), padding=2):
        super(mySelfCorrelationComputation, self).__init__()
        planes =[640, 64, 64, 640]
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=False)

        self.conv1x1_in = nn.Sequential(nn.Conv2d(channel, planes[1], kernel_size=1, bias=False, padding=0),
                                        nn.BatchNorm2d(planes[1]),
                                        nn.ReLU(inplace=True))
        self.embeddingFea = nn.Sequential(nn.Conv2d(1664, channel,
                                                     kernel_size=1, bias=False, padding=0),
                                           nn.BatchNorm2d(channel),
                                           nn.ReLU(inplace=True))
        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, bias=False, padding=0),
            nn.BatchNorm2d(channel))

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



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        return out


class ResNet(nn.Module):

    def __init__(self, args, block=BasicBlock):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.args = args
        self.layer1 = self._make_layer(block, 64, stride=2)
        self.layer2 = self._make_layer(block, 160, stride=2)
        self.layer3 = self._make_layer(block, 320, stride=2)
        self.layer4 = self._make_layer(block, 640, stride=2)
        # self.scr_module = SqueezeExcitation(channel=640)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.scr_module1 = mySelfCorrelationComputation(channel=160, kernel_size=(5, 5), padding=2)
        self.scr_module2 = mySelfCorrelationComputation(channel=320, kernel_size=(5, 5), padding=2)
        self.scr_module = mySelfCorrelationComputation(channel=640,kernel_size=(5, 5), padding=2)
        self.relu = nn.LeakyReLU(0.1)
        self.maxpool = nn.MaxPool2d(1)
        # self.scr_module = cbam_block(channel=640)
        # self.scr_module = SqueezeExcitation(channel=640)
        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(1120, 640, kernel_size=1, bias=False, padding=0),
            nn.BatchNorm2d(640))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # id = x
        # x = self.scr_module(x)
        # x = id + x
#____________________________________________________
        out1 = self.layer1(x)


        out2 = self.layer2(out1)
        # out2_s = self.scr_module1(out2)
        # out2 = out2 + out2_s


        out3 = self.layer3(out2)
        # out3_s = self.scr_module2(out3)
        # out3 = out3 + out3_s


        out4 = self.layer4(out3)
        # out4_s = self.scr_module(out4)
        # out4 = out4 + out4_s


        # ___________________________________________________________
        # out2 = F.avg_pool2d(out2, out2.size()[2:])
        # out3 = F.avg_pool2d(out3, out3.size()[2:])
        # out4 = F.avg_pool2d(out4, out4.size()[2:])

        # out2 = F.layer_norm(out2, out2.size()[1:])
        # out3 = F.layer_norm(out3, out3.size()[1:])
        # out4 = F.layer_norm(out4, out4.size()[1:])

        # out = torch.cat([out4, out3, out2], 1)
        # out = self.conv1x1_out(out)
        # x = self.relu(out4)
        # x = self.maxpool(out)

        return out4

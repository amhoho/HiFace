# -*- coding: UTF-8 -*-
# ----------------------------------------------------------
# @Author   : Etpoem
# @Time     : 2020/7/9 20:34
# @Desc     : 
# ----------------------------------------------------------
import torch
from torch import nn
from pathlib import Path
from collections import namedtuple


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    """A named tuple describing a ResNet block"""


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class BottleneckIR(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth)
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


def get_block(in_channel, depth, num_units, stride=2):

    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


class BackBone(nn.Module):
    def __init__(self):
        super(BackBone, self).__init__()
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(),
            Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512)
        )
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    BottleneckIR(bottleneck.in_channel,
                                 bottleneck.depth,
                                 bottleneck.stride)
                )
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return x


class FaceRec(object):
    def __init__(self):
        self.device = torch.device('cuda')
        self.model = BackBone()
        state_dict = torch.load(Path(__file__).parent / 'face_rec.pth')
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        del state_dict

    def get_feature(self, image):
        """
        获取人脸特征向量
        :param image:       tensor -- [1, 3, 112, 112]  aligned face
        :return:
                feature : ndarray 512D (has been normed)
        """
        with torch.no_grad():
            output = self.model(image)[0]
            feature = output / torch.norm(output)
            feature = feature.cpu().numpy()
        return feature


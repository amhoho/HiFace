# -*- coding: UTF-8 -*-
# -------------------------------------------------------------
# @Author   :   Etpoem
# @Time     :   2020/9/28 9:42
# @Desc     :   
# -------------------------------------------------------------
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, conv1x1
from pathlib import Path


class ResnetAg(nn.Module):
    def __init__(self, block=BasicBlock, layers=[1, 1, 1, 1]):
        super(ResnetAg, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        replace_stride_with_dilation = [False, False, False]
        self._norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 512, bias=True)
        self.age_fc = nn.Linear(512, 81, bias=True)
        self.gender_fc = nn.Linear(512, 2, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        age = self.age_fc(x)
        gender = self.gender_fc(x)

        return age, gender


class AgeGenderModel(object):
    def __init__(self):
        self.gender_type = ['female', 'male']
        self.model = ResnetAg()
        self.device = torch.device('cuda')
        state_dict = torch.load(Path(__file__).parent / 'age_gender_resnet9_93.pth')
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        del state_dict

    def predict(self, image):
        """
        :param image:   tensor -- 经过矫正的人脸，size [1, 3, 112, 112]
        :return:
                int     -- age
                float   -- age score
                int     -- gender, {0: female, 1: male}
                float   -- gender score
        """
        with torch.no_grad():
            age_out, gender_out = self.model(image)
            age_score, age = torch.max(age_out, 1)
            gender_score, gender = torch.max(gender_out, 1)
            return age.item(), age_score.item(), self.gender_type[gender.item()], gender_score.item()


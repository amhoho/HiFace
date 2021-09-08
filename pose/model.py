# -*- coding: UTF-8 -*-
# -------------------------------------------------------------
# @Author   :   Etpoem
# @Time     :   2020/9/29 16:55
# @Desc     :   
# -------------------------------------------------------------
import torch
import kornia
from torch import nn
from pathlib import Path
from torchvision.models.resnet import BasicBlock, conv1x1


def l2_norm(input_tensor, axis=1):
    norm = torch.norm(input_tensor, 2, axis, True)
    return torch.div(input_tensor, norm)


class ResnetPose(nn.Module):
    def __init__(self, block=BasicBlock, layers=[1, 1, 1, 1]):
        super(ResnetPose, self).__init__()
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
        self.w = nn.Parameter(torch.Tensor(512, 3))
        nn.init.xavier_uniform_(self.w)

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

        return torch.matmul(l2_norm(x, 1), l2_norm(self.w, 0)) * 180


class PoseModel(object):
    def __init__(self):
        self.model = ResnetPose()
        self.device = torch.device('cuda')
        state_dict = torch.load(Path(__file__).parent / 'resnet_pitch_57_yaw_35_roll_43.pth')
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        del state_dict

    def predict(self, image):
        """
        :param image:   tensor -- 经过特定裁剪的人脸区域，size [1, 3, 112, 112]
        :return:
                list -- [pitch, yaw, roll]
        """
        with torch.no_grad():
            pose_out = self.model(image)
            return pose_out[0].cpu().numpy().tolist()


def get_pose_region(src, box):
    """
    根据人脸位置在原图上裁剪出头部姿势检测区域
    :param src:         tensor -- [1, 3, h, w]
    :param box:         list -- [x1, y1, x2, y2]
    :return:
            pose_region: tensor -- [1, 3, 112, 112]
    """
    width = box[2] - box[0]
    height = box[3] - box[1]
    width_border = (height * 1.3 - width) / 2
    x1 = box[0] - width_border
    y1 = box[1] - 0.2 * height
    x2 = box[2] + width_border
    y2 = box[3] + height * 0.1
    new_box = torch.cuda.FloatTensor([[
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ]])
    pose_region = kornia.crop_and_resize(src, new_box, (112, 112))
    return pose_region

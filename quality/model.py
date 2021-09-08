# -*- coding: UTF-8 -*-
# -------------------------------------------------------------
# @Author   :   Etpoem
# @Time     :   2020/9/30 9:14
# @Desc     :   
# -------------------------------------------------------------
import torch
from torchvision.models.resnet import ResNet, BasicBlock
from pathlib import Path


class QualityModel(object):
    def __init__(self):
        self.model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=1)
        self.device = torch.device('cuda')
        state_dict = torch.load(Path(__file__).parent / 'quality_resnet9_val_80.pth')
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        del state_dict

    def predict(self, image):
        """
        :param image:   tensor -- 人脸区域，size [1, 3, 96, 96]
        :return:
                float -- 质量评估得分
        """
        with torch.no_grad():
            out = self.model(image)
            return out[0].item()


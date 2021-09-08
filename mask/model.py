# -*- coding: UTF-8 -*-
# -------------------------------------------------------------
# @Author   :   Etpoem
# @Time     :   2020/9/11 15:26
# @Desc     :   
# -------------------------------------------------------------
import torch
from torchvision.models.resnet import ResNet, BasicBlock
from pathlib import Path


class MaskModel(object):
    def __init__(self):
        self.mask_type = ['none', 'mask']
        self.model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=len(self.mask_type))
        self.device = torch.device('cuda')
        state_dict = torch.load(Path(__file__).parent / 'mask_resnet9_99.pth')
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        del state_dict

    def predict(self, image):
        """
        :param image:       tensor -- 人脸区域，size [1, 3, 96, 96]
        :return:
                str -- 是否戴口罩的描述
                float -- 预测得分
        """
        with torch.no_grad():
            out = self.model(image)
            score, index = torch.max(out, 1)
            return self.mask_type[index.item()], score.item()

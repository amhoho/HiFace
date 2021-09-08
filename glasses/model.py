# -*- coding: UTF-8 -*-
# -------------------------------------------------------------
# @Author   :   Etpoem
# @Time     :   2020/9/11 9:57
# @Desc     :   
# -------------------------------------------------------------
import torch
from torchvision.models.resnet import ResNet, BasicBlock
from pathlib import Path


class GlassesModel(object):
    def __init__(self):
        self.glasses_type = ['none', 'normal', 'dark']
        self.model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=len(self.glasses_type))
        self.device = torch.device('cuda')
        state_dict = torch.load(Path(__file__).parent / 'glasses_resnet9_97.pth')
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        del state_dict

    def predict(self, image):
        """
        :param image:   tensor -- 人脸区域, size [1, 3, 96, 96]
        :return:
                str -- 眼镜类型‘none' or 'normal' or 'dark'
                float -- 预测得分， 大于1则认为可信
        """
        with torch.no_grad():
            out = self.model(image)
            score, index = torch.max(out, 1)
            return self.glasses_type[index.item()], score.item()


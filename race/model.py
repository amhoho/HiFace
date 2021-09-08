# -*- coding: UTF-8 -*-
# -------------------------------------------------------------
# @Author   :   Etpoem
# @Time     :   2020/9/11 10:28
# @Desc     :   
# -------------------------------------------------------------
import torch
from torchvision.models.resnet import ResNet, BasicBlock
from pathlib import Path


class RaceModel(object):
    def __init__(self):
        self.race_type = ['white', 'asian', 'indian', 'black']
        self.model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=len(self.race_type))
        self.device = torch.device('cuda')
        state_dict = torch.load(Path(__file__).parent / 'race_resnet9_90.pth')
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        del state_dict

    def predict(self, image):
        """
        :param image:       tensor -- 经过矫正的人脸，size [1, 3, 112, 112]
        :return:
                str -- 种族类型
                float -- 预测得分，大于1则认为可信
        """
        with torch.no_grad():
            out = self.model(image)
            score, index = torch.max(out, 1)
            index, score = index.item(), score.item()
            if score < 1:
                index = 1
            return self.race_type[index], score


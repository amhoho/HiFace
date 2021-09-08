# -*- coding: UTF-8 -*-
# -------------------------------------------------------------
# @Author   :   Etpoem
# @Time     :   2021/2/20 15:29
# @Desc     :   
# -------------------------------------------------------------
import torch
import torchvision
from pathlib import Path

from .net import RetinaFace
from .model_util import decode_box, decode_landmark, generate_anchors


class FaceDetResnet(object):
    def __init__(self):
        self.device = torch.device('cuda')
        self.model = RetinaFace()
        state_dict = torch.load(Path(__file__).parent / 'resnet50.pth')
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        del state_dict
        # ------------------- 生成先验框 -----------------------------
        # 适用于任意尺寸输入
        self.height_0, self.width_0 = 720, 1280
        self.prior_box_0 = generate_anchors(height=self.height_0, width=self.width_0).to(self.device)
        self.box_scale_0 = torch.cuda.FloatTensor([self.width_0, self.height_0] * 2)
        self.landmark_scale_0 = torch.cuda.FloatTensor([self.width_0, self.height_0] * 5)

        # 适用于电脑端活体验证
        self.height_1, self.width_1 = 720, 1280
        self.prior_box_1 = generate_anchors(height=self.height_1, width=self.width_1).to(self.device)
        self.box_scale_1 = torch.cuda.FloatTensor([self.width_1, self.height_1] * 2)
        self.landmark_scale_1 = torch.cuda.FloatTensor([self.width_1, self.height_1] * 5)

        # 适用于手机端活体验证
        self.height_2, self.width_2 = 960, 720
        self.prior_box_2 = generate_anchors(height=self.height_2, width=self.width_2).to(self.device)
        self.box_scale_2 = torch.cuda.FloatTensor([self.width_2, self.height_2] * 2)
        self.landmark_scale_2 = torch.cuda.FloatTensor([self.width_2, self.height_2] * 5)

        # 适用于半特云口罩检测
        self.height_3, self.width_3 = 640, 480
        self.prior_box_3 = generate_anchors(height=self.height_3, width=self.width_3).to(self.device)
        self.box_scale_3 = torch.cuda.FloatTensor([self.width_3, self.height_3] * 2)
        self.landmark_scale_3 = torch.cuda.FloatTensor([self.width_3, self.height_3] * 5)

        # 适用于身份证头像
        self.height_4, self.width_4 = 126, 102
        self.prior_box_4 = generate_anchors(height=self.height_4, width=self.width_4).to(self.device)
        self.box_scale_4 = torch.cuda.FloatTensor([self.width_4, self.height_4] * 2)
        self.landmark_scale_4 = torch.cuda.FloatTensor([self.width_4, self.height_4] * 5)
        # -----------------------------------------------------------------------------------

        self.norm_parm = torch.cuda.FloatTensor([104, 117, 123])
        self.variance = [0.1, 0.2]
        self.nms_threshold = 0.4

    def detect(self, image, threshold, min_size, size_flag):
        """
        :param image:       tensor -- [h, w, c] RGB format
        :param threshold:   float -- 人脸检测阈值，0到1之间
        :param min_size:    int  -- 最小检测的人脸尺寸
        :param size_flag:   int -- 原图尺寸的大小标识位
        :return:
        """
        if size_flag == 0:
            src_height, src_width, _ = image.shape
            if src_height != self.height_0 or src_width != self.width_0:
                self.height_0 = src_height
                self.width_0 = src_width
                self.prior_box_0 = generate_anchors(self.height_0, self.width_0).to(self.device)
                self.box_scale_0 = torch.cuda.FloatTensor([self.width_0, self.height_0] * 2)
                self.landmark_scale_0 = torch.cuda.FloatTensor([self.width_0, self.height_0] * 5)
            prior_box = self.prior_box_0
            box_scale = self.box_scale_0
            landmark_scale = self.landmark_scale_0
        if size_flag == 1:
            prior_box = self.prior_box_1
            box_scale = self.box_scale_1
            landmark_scale = self.landmark_scale_1
        if size_flag == 2:
            prior_box = self.prior_box_2
            box_scale = self.box_scale_2
            landmark_scale = self.landmark_scale_2
        if size_flag == 3:
            prior_box = self.prior_box_3
            box_scale = self.box_scale_3
            landmark_scale = self.landmark_scale_3
        if size_flag == 4:
            prior_box = self.prior_box_4
            box_scale = self.box_scale_4
            landmark_scale = self.landmark_scale_4

        with torch.no_grad():
            # convert rgb mode to bgr mode
            image = image[:, :, (2, 1, 0)]
            # norm
            image -= self.norm_parm
            # HWC to CHW and expand dimension
            image = image.permute(2, 0, 1).unsqueeze(0)
            # inference
            boxes, scores, landmarks = self.model(image)
            boxes = boxes.squeeze(0)
            scores = scores.squeeze(0)[:, 1]
            landmarks = landmarks.squeeze(0)

            # ignore low score
            index = torch.where(scores >= threshold)
            if len(index) == 0:
                return [], []
            scores = scores[index]
            boxes = boxes[index]
            landmarks = landmarks[index]
            prior_box = prior_box[index]

            # boxes, landmarks 解码
            boxes = decode_box(boxes, prior_box, self.variance)
            boxes = boxes * box_scale
            landmarks = decode_landmark(landmarks, prior_box, self.variance)
            landmarks = landmarks * landmark_scale

            # do nms
            index = torchvision.ops.nms(boxes, scores, self.nms_threshold)
            boxes = boxes[index]
            landmarks = landmarks[index]
            scores = scores[index]

            # order
            order = scores.argsort(descending=True)
            # scores = scores[order]
            boxes = boxes[order]
            landmarks = landmarks[order]

            # filter tiny face
            side = torch.cat((
                (boxes[:, 2] - boxes[:, 0]).unsqueeze(1),
                (boxes[:, 3] - boxes[:, 1]).unsqueeze(1)
            ), dim=1)
            min_side, _ = torch.min(side, dim=1)
            index = torch.where(min_side > min_size)
            if len(index) == 0:
                return [], []
            boxes = boxes[index]
            landmarks = landmarks[index]

            # convert to python list and reshape landmarks to [n, 5, 2] format
            boxes = boxes.cpu().int().numpy().tolist()
            landmarks = landmarks.reshape((-1, 5, 2))
            return boxes, landmarks

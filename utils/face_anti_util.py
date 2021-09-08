# -*- coding: UTF-8 -*-
# -------------------------------------------------------------
# @Author   :   Etpoem
# @Time     :   2020/8/6 9:06
# @Desc     :   
# -------------------------------------------------------------
import torch
import kornia


def get_anti_detect_region(src_image, box, scale):
    """
    根据检测到的人脸框，按照指定scale倍数扩展人脸区域，并返回resize后的人脸区域
    :param src_image:       tensor -- [1, 3, h, w]
    :param box:             list -- [x_min, y_min, x_max, y_max]
    :param scale:           float -- 人脸区域扩展倍数， 2.7 for model_1, 4 for model_2
    :return:
                tensor -- scaled face region that resize to (80, 80)
    """
    src_height = src_image.shape[2]
    src_width = src_image.shape[3]
    box_height = box[3] - box[1]
    box_width = box[2] - box[0]
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    scale = min(src_height / box_height, src_width / box_width, scale)
    long_side = max(box_height, box_width)
    new_side = long_side * scale

    x_min = center_x - new_side / 2
    y_min = center_y - new_side / 2
    x_max = center_x + new_side / 2
    y_max = center_y + new_side / 2
    if x_min < 0:
        x_max -= x_min
        x_min = 0
    if y_min < 0:
        y_max -= y_min
        y_min = 0
    if x_max > src_width:
        x_min -= x_max - src_width
        x_max = src_width
    if y_max > src_height:
        y_min -= y_max - src_height
        y_max = src_height
    tensor_box = torch.cuda.FloatTensor([[
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max]
    ]])
    region = kornia.crop_and_resize(src_image, tensor_box, (80, 80))
    return region


def adjust_box(box):
    """
    检测出的人脸框多为长方形，在此缩减高度，扩展宽度使人脸框成为正方形
    :param box:     list -- [x_min, y_min, x_max, y_max]
    :return:
            list -- new box that adjust to square
    """
    x1, y1, x2, y2 = box
    width = x2 - x1
    new_width = width * 1.1
    x1 = x1 - 0.05 * width
    x2 = x2 + 0.05 * width
    y1 = y2 - new_width
    if y1 < 0:
        y1 = 0
    return [int(x1), int(y1), int(x2), int(y2)]


def face_in_center(box):
    """
    发送过来的图片尺寸为 1280x720，
    判断人脸是否位于中间 700 x 700 的区域内
    :param box:     list -- [x_min, y_min, x_max, y_max]
    :return:
                bool --
    """
    face_x1, face_y1, face_x2, face_y2 = box
    center_x1, center_y1, center_x2, center_y2 = 290, 10, 990, 710
    if face_x1 > center_x1 and face_x2 < center_x2 and face_y1 > center_y1 and face_y2 < center_y2:
        return True
    else:
        return False


def face_in_phone_center(box):
    """
    移动端发送过来的图片短边为 720
    判断人脸是否位于中间
    ：param box:    list -- [x_min, y_min, x_max, y_max]
    :return:
                bool -- 
    """
    face_x1, _, face_x2, _ = box
    center_x1, center_x2 = 20, 700
    if face_x1 > center_x1 and face_x2 < center_x2:
        return True
    else:
        return False

# -*- coding: UTF-8 -*-
# -------------------------------------------------------------
# @Author   :   Etpoem
# @Time     :   2020/7/24 9:53
# @Desc     :   
# -------------------------------------------------------------
import io
import torch
import toml
import kornia
import base64
import numpy as np
from PIL import Image
from PIL import ImageFile
from pathlib import Path

ImageFile.LOAD_TRUNCATED_IMAGES = True
# top_path = Path(__file__).parents[1]
# cfg = toml.load(top_path / 'config.toml')
# # -------------------获取配置参数--------------------
# _DARK_PIXEL = cfg['face']['dark_pixel']
# _DARK_THRESHOLD = cfg['face']['dark_threshold']
# _BLUR_THRESHOLD = cfg['face']['blur_threshold']


# def load_image(image_path):
#     """
#     将图片转换为tensor数据
#     :param image_path:  str or PosixPath --
#     :return:
#             tensor -- [1, 3, h, w]
#     """
#     pil_image = Image.open(image_path).convert('RGB')
#     image = np.asarray(pil_image, 'float32')
#     image = torch.cuda.FloatTensor(image)
#     image = image.permute((2, 0, 1)).unsqueeze(0)
#     return image


# def is_good_face(image, box, min_detect_size):
#     """
#     根据人脸尺寸、朝向、明暗程度、模糊程度判断该人脸的质量
#     :param image:        tensor -- shape of [1, 3, h, w] not normalize
#     :param box:         list -- [x1, y1, x2, y2]
#     :param min_detect_size: int --
#     :return:
#             bool --
#     """
#
#     face_width = box[2] - box[0]
#     face_height = box[3] - box[1]
#     # ------------判断是否为小尺寸的人脸-------------
#     min_size = min(face_width, face_height)
#     if min_size < min_detect_size:
#         return f'人脸尺寸小于{min_detect_size}'
#     x1, y1, x2, y2 = box
#     face_box = torch.cuda.FloatTensor([[
#         [x1, y1],
#         [x2, y1],
#         [x2, y2],
#         [x1, y2]
#     ]])
#     face_region = kornia.crop_and_resize(image, face_box, (70, 70))
#     # -------------判断人脸明暗程度-------------------
#     gray = kornia.color.RgbToGrayscale()
#     gray_face = gray(face_region)
#     dark_mask = torch.lt(gray_face, _DARK_PIXEL)
#     dark_mask_area = torch.sum(dark_mask)
#     dark_ratio = dark_mask_area.item() / (70 * 70)
#     if dark_ratio > _DARK_THRESHOLD:
#         return '该人脸过于暗'
#     # --------------判断人脸模糊程度 ------------------
#     laplace = kornia.filters.Laplacian(5)
#     laplace_out = laplace(gray_face)
#     blur_var = torch.var(laplace_out).item()
#     if blur_var < _BLUR_THRESHOLD:
#         return '该人脸过于模糊'
#     # --------------属于质量好的人脸------------------
#     return 'success'


def get_hat_region(src, box):
    """
    根据人脸位置在原图上裁剪出帽子检测区域
    :param src:         tensor -- [1, 3, h, w]
    :param box:         list -- [x1, y1, x2, y2]
    :return:
            hat_region: tensor -- [1, 3, 112, 112]
    """
    height = box[3] - box[1]
    width = box[2] - box[0]
    x1 = box[0] - 0.15 * width
    y1 = box[1] - 0.35 * height
    x2 = box[2] + 0.15 * width
    y2 = box[3] - 0.5 * height
    new_box = torch.cuda.FloatTensor([[
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ]])
    hat_region = kornia.crop_and_resize(src, new_box, (96, 96))
    return hat_region


def get_glass_mask_region(src, box):
    """
    根据人脸位置在原图上裁剪出眼镜、口罩检测区域
    :param src:         tensor -- [1, 3, h, w]
    :param box:         list -- [x1, y1, x2, y2]
    :return:
                glass_mask_region: tensor -- [1, 3, 96, 96]
    """
    x1, y1, x2, y2 = box
    new_box = torch.cuda.FloatTensor([[
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ]])
    glass_mask_region = kornia.crop_and_resize(src, new_box, (96, 96))
    return glass_mask_region


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


def decode_image(img_string):
    """
    图片解码, 并载入
    :param img_string:  图片base64 编码字符串
    :return:
            nparray -- [h, w, c]
    """
    image = base64.b64decode(img_string)
    image = io.BytesIO(image)
    image = Image.open(image).convert('RGB')
    image = np.asarray(image, 'float32')
    return image

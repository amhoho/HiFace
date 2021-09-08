# -*- coding: UTF-8 -*-
# -------------------------------------------------------------
# @Author   :   Etpoem
# @Time     :   2021/1/15 15:56
# @Desc     :   
# -------------------------------------------------------------
import torch
from math import ceil
from itertools import product


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode_box(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landmark(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landmarks = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                           priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                           priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                           priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                           priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:]), dim=1)
    return landmarks


def generate_anchors(height, width):
    anchors = []
    min_sizes = [[16, 32], [64, 128], [256, 512]]
    steps = [8, 16, 32]
    image_size = (height, width)
    feature_maps = [[ceil(image_size[0] / step), ceil(image_size[1] / step)] for step in steps]
    for k, f in enumerate(feature_maps):
        min_size = min_sizes[k]
        for i, j in product(range(f[0]), range(f[1])):
            for size in min_size:
                s_kx = size / image_size[1]
                s_ky = size / image_size[0]
                dense_cx = [x * steps[k] / image_size[1] for x in [j + 0.5]]
                dense_cy = [y * steps[k] / image_size[0] for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]
    output = torch.Tensor(anchors).view(-1, 4)
    return output

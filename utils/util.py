# -*- coding: UTF-8 -*-
# -------------------------------------------------------------
# @Author   :   Etpoem
# @Time     :   2020/8/6 11:07
# @Desc     :   
# -------------------------------------------------------------
import uuid


def get_request_id():
    """
    返回指定格式uuid
    """
    request_id = uuid.uuid1()
    request_id = ''.join(str(request_id).split('-'))
    return request_id


def score_mapping(score):
    """
    预测可信度的分值映射，映射到[0, 1]
    :param score:   float -- 模型输出的可信值
    :return:
    """
    if score > 5:
        score = 1.0
    elif score > 1:
        score = 0.025 * score + 0.875
    elif score > -1:
        score = 0.4 * score + 0.5
    else:
        score = 0.1
    return score


def distance_to_score(distance):
    """
    将向量的欧式距离映射为相似度值
    :param distance:    float
    :return:
    """
    if distance <= 1.2:
        score = 7.0 / (7.0 + distance)
    elif distance < 1.9:
        score = -1 * distance + 2.05
    elif distance >= 1.9:
        score = 0.10
    return score

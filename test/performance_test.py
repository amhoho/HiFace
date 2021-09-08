# -*- coding: UTF-8 -*-
# -------------------------------------------------------------
# @Author   :   panjf
# @Time     :   2021/2/26 11:34
# @Desc     :   
# -------------------------------------------------------------
import time
import json
import base64
import torch
import requests
import numpy as np
from pathlib import Path
from PIL import Image

from face_det_resnet50.model import FaceDetResnet

BASE_URL = 'http://192.168.96.136:17006'


def add_face_test():
    url = f'{BASE_URL}/v1/db/insert'
    image_path = Path(__file__).parent / 'data/sample_1.jpg'
    with image_path.open('rb') as f:
        image = f.read()
        base64_data = base64.b64encode(image)
    since = time.time()
    for i in range(1000):
        data = {
            'db_name': 'test_p',
            'feature_id': i,
            'image': base64_data.decode('ascii')
        }
        response = requests.post(url=url, json=data)
        response_json = response.json()
        if response_json['error'] != '':
            print(response_json['error'])
        if i % 100 == 0:
            use_time = time.time() - since
            fps = i / use_time
            print(f'add {i} image use time {use_time}s FPS:{fps}')


def face_search_test():
    url = f'{BASE_URL}/v1/db/search'
    image_path = Path(__file__).parent / 'data/org1_720.jpg'
    with image_path.open('rb') as f:
        image = f.read()
        base64_data = base64.b64encode(image)
    since = time.time()
    for i in range(300):
        data = {
            'db_name': 'face_login_test',
            'image': base64_data.decode('ascii'),
            'top': 5,
            'nprobe': 128,
            'detection_threshold': 0.95,
            'min_size': 40
        }
        response = requests.post(url=url, json=data)
        response_json = response.json()
        # print(response_json)
        if response_json['error'] != '':
            print(response_json['error'])
        if i % 50 == 0:
            use_time = time.time() - since
            fps = i / use_time
            print(f'face search {i} image use time {use_time}s FPS:{fps}')


def face_det_view():
    url = f'{BASE_URL}/face/detect'
    image_path = Path(__file__).parent / 'data/bad3.jpg'
    with image_path.open('rb') as f:
        image = f.read()
        base64_data = base64.b64encode(image)

    data = {
        'image': base64_data.decode('ascii'),
        'detection_threshold': 0.95,
        'min_size': 40
    }
    response = requests.post(url=url, json=data)
    response_json = response.json()
    print(response_json)


def face_detect_test():
    url = f'{BASE_URL}/face/detect'
    image_path = Path(__file__).parent / 'data/sample_1.jpg'
    with image_path.open('rb') as f:
        image = f.read()
        base64_data = base64.b64encode(image)
    since = time.time()
    for i in range(1000):
        data = {
            'image': base64_data.decode('ascii'),
            'detection_threshold': 0.95,
            'min_size': 40
        }
        response = requests.post(url=url, json=data)
        response_json = response.json()
        if response_json['error'] != '':
            print(response_json['error'])
        if i % 100 == 0:
            use_time = time.time() - since
            fps = i / use_time
            print(f'detect {i} image use time {use_time}s FPS:{fps}')

    # data = {
    #     'image': base64_data.decode('ascii'),
    #     'detection_threshold': 0.95,
    #     'min_size': 40
    # }
    # response = requests.post(url=url, json=data)
    # response_json = response.json()
    # print(response_json)


def retina_test():
    face_det = FaceDetResnet()
    print('face det model load finish')
    image_path = Path(__file__).parent / 'data/sample_1.jpg'
    image = Image.open(image_path).convert('RGB')
    image = np.asarray(image, np.uint8)

    t_image = torch.cuda.FloatTensor(image)
    # t_det_image = t_image.clone()
    # print(t_image[100, 100, :])
    # boxes, landmarks = face_det.detect(t_det_image, 0.95, 50, 0)
    # print(boxes)
    # print(t_image[100, 100, :])

    since = time.time()
    for i in range(1000):
        # t_image = torch.cuda.FloatTensor(image)
        t_det_image = t_image.clone()
        boxes, landmarks = face_det.detect(t_det_image, 0.95, 50, 0)
        if i % 100 == 0:
            use_time = time.time() - since
            fps = i / use_time
            print(f'add {i} image use time {use_time}s FPS:{fps}')

    # t_image = torch.cuda.FloatTensor(image)
    # boxes, landmarks = face_det.detect(t_image, 0.95, 50, 0)
    # print(boxes)


def face_mask_test():
    url = f'{BASE_URL}/face/mask'
    image_path = Path(__file__).parent / 'data/bad3.jpg'
    with image_path.open('rb') as f:
        image = f.read()
        base64_data = base64.b64encode(image)

    # data = {
    #     'image': base64_data.decode('ascii'),
    #     'threshold': 0.95,
    #     'size': 40
    # }
    # response = requests.post(url=url, json=data)
    # response_json = response.json()
    # print(response_json)

    since = time.time()
    for i in range(300):
        data = {
            'image': base64_data.decode('ascii'),
            'threshold': 0.95,
            'size': 40
        }
        response = requests.post(url=url, json=data)
        response_json = response.json()
        if response_json['error'] != '':
            print(response_json['error'])
        if i % 50 == 0:
            use_time = time.time() - since
            fps = i / use_time
            print(f'mask detect {i} image use time {use_time}s FPS:{fps}')


def face_mask_view():
    url = f'{BASE_URL}/face/mask'
    image_path = Path(__file__).parent / 'data/bad3.jpg'
    with image_path.open('rb') as f:
        image = f.read()
        base64_data = base64.b64encode(image)

    data = {
        'image': base64_data.decode('ascii'),
        'threshold': 0.95,
        'size': 40
    }
    response = requests.post(url=url, json=data)
    response_json = response.json()
    print(response_json)


def face_anti_test():
    url = f'{BASE_URL}/face/anti/detect'
    image_path = Path(__file__).parent / 'data/phone_other.jpg'
    with image_path.open('rb') as f:
        image = f.read()
        base64_data = base64.b64encode(image)

    # data = {
    #     'image': base64_data.decode('ascii'),
    #     'device': 'computer'
    # }
    # response = requests.post(url=url, json=data)
    # print(response.json())

    since = time.time()
    for i in range(300):
        data = {
            'image': base64_data.decode('ascii'),
            'device': 'phone'
        }
        response = requests.post(url=url, json=data)
        response_json = response.json()
        if response_json['error'] != '':
            print(response_json['error'])
        if i % 50 == 0:
            use_time = time.time() - since
            fps = i / use_time
            print(f'anti detect FPS : {fps}')


def face_verify_test():
    url = f'{BASE_URL}/v1/db/face/verify'
    image_path = Path(__file__).parent / 'data/640_face.jpg'
    with image_path.open('rb') as f:
        image = f.read()
        base64_data = base64.b64encode(image)

    # data = {
    #     'image': base64_data.decode('ascii'),
    #     'db_name': 'face_login_test',
    #     'feature_id': 1
    # }
    # response = requests.post(url=url, json=data)
    # print(response.json())

    since = time.time()
    for i in range(300):
        data = {
            'image': base64_data.decode('ascii'),
            'db_name': 'face_login_test',
            'feature_id': 1
        }
        response = requests.post(url=url, json=data)
        response_json = response.json()
        if response_json['error'] != '':
            print(response_json['error'])
        if i % 50 == 0:
            use_time = time.time() - since
            fps = i / use_time
            print(f'face verify FPS : {fps}')


def identity_compare_test():
    url = f'{BASE_URL}/face/identity/compare'
    image_a_path = Path(__file__).parent / 'data/102_face.jpg'
    image_b_path = Path(__file__).parent / 'data/640_face.jpg'
    with image_a_path.open('rb') as f:
        image_a = f.read()
        image_a_data = base64.b64encode(image_a)
    with image_b_path.open('rb') as f:
        image_b = f.read()
        image_b_data = base64.b64encode(image_b)

    data = {
        'image_a': image_a_data.decode('ascii'),
        'image_b': image_b_data.decode('ascii')
    }
    response = requests.post(url=url, json=data)
    print(response.json())

    since = time.time()
    for i in range(300):
        data = {
            'image_a': image_a_data.decode('ascii'),
            'image_b': image_b_data.decode('ascii')
        }
        response = requests.post(url=url, json=data)
        response_json = response.json()
        if response_json['error'] != '':
            print(response_json['error'])
        if i % 50 == 0:
            use_time = time.time() - since
            fps = i / use_time
            print(f'identity compare FPS : {fps}')


if __name__ == "__main__":
    # face_anti_test()
    # add_face_test()
    # face_search_test()
    # face_det_view()
    # face_detect_test()
    # retina_test()
    face_mask_test()
    # face_mask_view()
    # face_verify_test()
    # identity_compare_test()

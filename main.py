# -*- coding: UTF-8 -*-
# -------------------------------------------------------------
# @Author   :   panjf
# @Time     :   2021/2/22 15:05
# @Desc     :   
# -------------------------------------------------------------
import os
import toml
config = toml.load('config.toml')
os.environ["CUDA_VISIBLE_DEVICES"] = config['service']['gpu_device']
import json
import torch
import requests
import numpy as np
from waitress import serve
from flask import Flask, request, jsonify
from marshmallow import ValidationError

# 导入人脸模块
from face_det_resnet50.model import FaceDetResnet
from face_rec.model import FaceRec
from age_gender.model import AgeGenderModel
from expression.model import ExpressionModel
from glasses.model import GlassesModel
from hat.model import HatModel
from mask.model import MaskModel
from pose.model import PoseModel
from quality.model import QualityModel
from race.model import RaceModel
from face_anti.model import FaceAntiM1, FaceAntiM2

# 导入工具模块
from utils.mysql_util import MysqlManager
from utils.face_align import AlignFace
from utils.face_util import get_glass_mask_region, get_pose_region, get_hat_region, decode_image
from utils.face_anti_util import get_anti_detect_region, face_in_center, face_in_phone_center, adjust_box
from utils import Val
from utils.log_util import setup_logger
from utils.util import distance_to_score, score_mapping


logger = setup_logger(name='main')


class FaceService(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.min_width = self.cfg['service']['min_width']
        self.min_height = self.cfg['service']['min_height']
        self.detection_threshold = self.cfg['service']['detection_threshold']
        self.default_login_db = self.cfg['service']['default_login_db']
        self.match_score = self.cfg['service']['match_score']
        self.anti_score = self.cfg['service']['anti_score']
        self.nprobe = self.cfg['faiss']['nprobe']
        self.flask_host = self.cfg['flask']['host']
        self.flask_port = self.cfg['flask']['port']
        self.faiss_host = self.cfg['faiss']['host']
        self.faiss_port = self.cfg['faiss']['port']
        self.faiss_base_url = f'http://{self.faiss_host}:{self.faiss_port}'
        logger.info('配置参数加载完成')

        # ----------------- 人脸模块加载 ------------------------------
        self.face_det = FaceDetResnet()
        logger.info('人脸检测模块加载完成')
        self.face_rec = FaceRec()
        logger.info('人脸特征提取模块加载完成')
        self.face_align = AlignFace()
        logger.info('人脸矫正模块加载完成')
        self.pose_det = PoseModel()
        logger.info('人脸姿势估计模块加载完成')
        self.age_gender_det = AgeGenderModel()
        logger.info('性别、年龄估计模块加载完成')
        self.expresion_det = ExpressionModel()
        logger.info('表情分类模块加载完成')
        self.glass_det = GlassesModel()
        logger.info('眼镜分类模块加载完成')
        self.mask_det = MaskModel()
        logger.info('口罩分类模块加载完成')
        self.hat_det = HatModel()
        logger.info('帽子分类模块加载完成')
        self.race_det = RaceModel()
        logger.info('人种分类模块加载完成')
        self.quality_det = QualityModel()
        logger.info('人脸质量评估模块加载完成')
        self.face_anti_1 = FaceAntiM1()
        self.face_anti_2 = FaceAntiM2()
        logger.info('活体检测模块加载完成')

        self.cbvsp_analyze_on = self.cfg['service']['cbvsp_analyze_on']
        self.pedestrian_db_name = self.cfg['service']['pedestrian_db_name']
        if self.cbvsp_analyze_on:
            self.mysql_manager = MysqlManager()
        logger.info('人脸服务初始化完成')
        logger.info(f'running on {self.flask_host}:{self.flask_port}')

    def flask_run(self):
        app = Flask(__name__)

        @app.route('/v1/db/create', methods=["POST"])
        def db_create():
            url = f'{self.faiss_base_url}/faiss/db/create'
            receive_param = request.get_json()
            ret = requests.post(url=url, json=receive_param).json()
            return jsonify(ret)

        @app.route('/v1/db/remove', methods=["POST"])
        def db_remove():
            url = f'{self.faiss_base_url}/faiss/db/remove'
            receive_param = request.get_json()
            ret = requests.post(url=url, json=receive_param).json()

            return jsonify(ret)

        @app.route('/v1/db/update', methods=["POST"])
        def db_update():
            url = f'{self.faiss_base_url}/faiss/db/update'
            receive_param = request.get_json()
            ret = requests.post(url=url, json=receive_param).json()

            return jsonify(ret)

        @app.route('/v1/db/info', methods=["POST"])
        def db_info():
            url = f'{self.faiss_base_url}/faiss/db/info'
            receive_param = request.get_json()
            ret = requests.post(url=url, json=receive_param).json()

            return jsonify(ret)

        @app.route('/v1/db/insert', methods=["POST"])
        def db_insert():
            ret_info = {
                'error': '',
                'result': []
            }
            # -------------------- 参数校验 ----------------------
            receive_param = request.get_json()
            try:
                receive_param = Val.DbInsert().load(receive_param)
                db_name = receive_param['db_name']
                feature_id = receive_param['feature_id']
                image = receive_param['image']
                info = receive_param['info']
                detection_threshold = receive_param['detection_threshold']
                min_size = receive_param['min_size']
            except ValidationError as err:
                ret_info['error'] = str(err.messages)
                return jsonify(ret_info)
            try:
                image = decode_image(image)
                det_image = torch.cuda.FloatTensor(image)
                boxes, landmarks = self.face_det.detect(det_image, detection_threshold, min_size, size_flag=0)
                if len(boxes) != 0:
                    box = boxes[0]
                    rgb_image = torch.cuda.FloatTensor(image)
                    rgb_image = (rgb_image - 127.5) * 0.0078125
                    rgb_image = rgb_image.permute(2, 0, 1).unsqueeze(0)
                    aligned_face = self.face_align.get(rgb_image, landmarks[0])
                    feature = self.face_rec.get_feature(aligned_face)
                    data = {
                        'db_name': db_name,
                        'feature_id': feature_id,
                        'feature': json.dumps(feature.tolist()),
                        'faceCoord': box,
                        'info': info
                    }
                    url = f'{self.faiss_base_url}/faiss/feature/insert'
                    ret = requests.post(url=url, json=data).json()
                    return jsonify(ret)
                else:
                    ret_info['error'] = '该图片检测不到人脸'
                    return jsonify(ret_info)
            except Exception as e:
                ret_info['error'] = str(e)
                return jsonify(ret_info)

        @app.route('/v1/db/delete', methods=["POST"])
        def face_delete():
            url = f'{self.faiss_base_url}/faiss/feature/delete'
            receive_param = request.get_json()
            ret = requests.post(url=url, json=receive_param).json()
            return jsonify(ret)

        @app.route('/v1/db/delete_by_date', methods=["POST", "DELETE"])
        def face_delete_by_date():
            try:
                url = f'{self.faiss_base_url}/faiss/feature/delete_by_date'
                receive_param = request.get_json()
                ret = requests.post(url=url, json=receive_param).json()
                # 如果有进行行人库同人分析
                if self.cbvsp_analyze_on \
                        and receive_param['db_name'] == self.pedestrian_db_name \
                        and ret['error'] == '':
                    mysql_ret = self.mysql_manager.main_table_delete_by_date(
                        receive_param['begin_time'], receive_param['end_time']
                    )
                    if mysql_ret != 'success':
                        ret['error'] = mysql_ret
                return jsonify(ret)
            except Exception as e:
                ret_info = {
                    'error': str(e),
                    'feature_ids': []
                }
                return jsonify(ret_info)

        @app.route('/v1/db/delete_batch', methods=['POST'])
        def face_delete_batch():
            url = f'{self.faiss_base_url}/faiss/feature/delete_batch'
            receive_param = request.get_json()
            ret = requests.post(url=url, json=receive_param).json()
            return jsonify(ret)

        @app.route('/v1/db/get', methods=["POST"])
        def db_get():
            url = f'{self.faiss_base_url}/faiss/feature/get'
            receive_param = request.get_json()
            ret = requests.post(url=url, json=receive_param).json()
            return jsonify(ret)

        @app.route('/v1/db/set', methods=["POST"])
        def db_set():
            ret_info = {
                'result': [],
                'error': ''
            }
            # -----------------参数校验-----------------
            receive_param = request.get_json()
            try:
                receive_param = Val.DbSet().load(receive_param)
                min_width = receive_param['minWidth']
                min_height = receive_param['minHeight']
                detection_threshold = receive_param['detectionThreshold']
                queue_length = receive_param['queueLength']
                fusion_mode = receive_param['fusion_mode']
            except ValidationError as err:
                ret_info['error'] = str(err.messages)
                return jsonify(ret_info)
            # -------------------------------------------------
            self.min_width = min_width
            self.min_height = min_height
            self.min_detect_size = min(min_width, min_height)
            self.cfg['service']['min_height'] = min_height
            self.cfg['service']['min_width'] = min_width
            self.cfg['service']['detection_threshold'] = detection_threshold
            with open('config.toml', 'w') as f:
                toml.dump(self.cfg, f)
            result = {
                'minWidth': min_width,
                'minHeight': min_height,
                'detectionThreshold': detection_threshold,
                'queueLength': queue_length,
                'fusion_mode': fusion_mode
            }
            ret_info['result'].append(result)
            return jsonify(ret_info)

        @app.route('/v1/db/insert_feature', methods=["POST"])
        def insert_feature():
            url = f'{self.faiss_base_url}/faiss/feature/insert'
            receive_param = request.get_json()
            ret = requests.post(url=url, json=receive_param).json()
            return jsonify(ret)

        @app.route('/v1/db/search', methods=["POST"])
        def db_search():
            ret_info = {
                'error': '',
                'result': []
            }
            # ------------------- 参数校验 ---------------------------
            receive_param = request.get_json()
            try:
                receive_param = Val.DbSearch().load(receive_param)
                db_name = receive_param['db_name']
                image = receive_param['image']
                top = receive_param['top']
                nprobe = receive_param['nprobe']
                detection_threshold = receive_param['detection_threshold']
                min_size = receive_param['min_size']
            except ValidationError as err:
                ret_info['error'] = str(err.messages)
                return jsonify(ret_info)
            # ---------------------------------------------------------
            try:
                image = decode_image(image)
                det_image = torch.cuda.FloatTensor(image)
                boxes, landmarks = self.face_det.detect(det_image, detection_threshold, min_size, size_flag=0)
                if len(boxes) != 0:
                    rgb_image = torch.cuda.FloatTensor(image)
                    rgb_image = (rgb_image - 127.5) * 0.0078125
                    rgb_image = rgb_image.permute(2, 0, 1).unsqueeze(0)
                    aligned_face = self.face_align.get(rgb_image, landmarks[0])
                    feature = self.face_rec.get_feature(aligned_face)
                    data = {
                        'db_name': db_name,
                        'feature': json.dumps(feature.tolist()),
                        'top': top,
                        'nprobe': nprobe
                    }
                    url = f'{self.faiss_base_url}/faiss/feature/search'
                    ret = requests.post(url=url, json=data).json()
                    # 如果开启了行人库同人分析, 去行人库主表获取person_id
                    if self.cbvsp_analyze_on \
                        and receive_param['db_name'] == self.pedestrian_db_name \
                        and ret['error'] == '':
                        feature_ids = [item['feature_id'] for item in ret['result']]
                        person_id_dict = self.mysql_manager.select_person_ids(feature_ids)
                        for item in ret['result']:
                            item['person_id'] = person_id_dict.get(str(item['feature_id']), -1)
                    return jsonify(ret)
                else:
                    ret_info['error'] = '该图片检测不到人脸'
                    return jsonify(ret_info)
            except Exception as e:
                ret_info['error'] = str(e)
                return jsonify(ret_info)

        @app.route('/v1/db/search_feature', methods=["POST"])
        def search_feature():
            try:
                url = f'{self.faiss_base_url}/faiss/feature/search'
                receive_param = request.get_json()
                ret = requests.post(url=url, json=receive_param).json()
                # 如果开启了行人库同人分析, 去行人库主表获取person_id
                if self.cbvsp_analyze_on \
                        and receive_param['db_name'] == self.pedestrian_db_name \
                        and ret['error'] == '':
                    feature_ids = [item['feature_id'] for item in ret['result']]
                    person_id_dict = self.mysql_manager.select_person_ids(feature_ids)
                    for item in ret['result']:
                        item['person_id'] = person_id_dict.get(str(item['feature_id']), -1)
                return jsonify(ret)
            except Exception as e:
                logger.exception('search_feature 发生异常')
                ret_info = {
                    'error': str(e),
                    'result': []
                }
                return jsonify(ret_info)

        @app.route('/face_rec/state', methods=["GET"])
        def state():
            url = f'{self.faiss_base_url}/faiss/state'
            ret = requests.get(url=url).json()
            return jsonify(ret)

        @app.route('/v1/db/face/verify', methods=["POST"])
        def face_verify():
            ret_info = {
                'error': '',
                'result': []
            }
            # -------------------- 参数校验 ---------------------
            receive_param = request.get_json()
            try:
                receive_param = Val.FaceVerify().load(receive_param)
                db_name = receive_param['db_name']
                feature_id = receive_param['feature_id']
                image = receive_param['image']
                detection_threshold = receive_param['detection_threshold']
                min_size = receive_param['min_size']
            except ValidationError as err:
                ret_info['error'] = str(err.messages)
                return jsonify(ret_info)
            # ---------------------------------------------------
            try:
                url = f'{self.faiss_base_url}/faiss/feature/get'
                data = {
                    'db_name': db_name,
                    'feature_id': feature_id
                }
                ret = requests.post(url=url, json=data).json()
                if ret['error'] == '':
                    image = decode_image(image)
                    det_image = torch.cuda.FloatTensor(image)
                    boxes, landmarks = self.face_det.detect(det_image, detection_threshold, min_size, size_flag=0)
                    if len(boxes) != 0:
                        result_info = {}
                        result_info['feature_id'] = feature_id
                        result_info['db_name'] = db_name
                        result_info['info'] = ret['result'][0]['info']
                        result_info['face_coord_0'] = ret['result'][0]['faceCoord']
                        result_info['face_coord_1'] = boxes[0]
                        face_vector_1 = np.asarray(json.loads(ret['result'][0]['feature']), 'float32')
                        rgb_image = torch.cuda.FloatTensor(image)
                        rgb_image = (rgb_image - 127.5) * 0.0078125
                        rgb_image = rgb_image.permute(2, 0, 1).unsqueeze(0)
                        aligned_face = self.face_align.get(rgb_image, landmarks[0])
                        face_vector_2 = self.face_rec.get_feature(aligned_face)
                        distance = np.linalg.norm(face_vector_2 - face_vector_1)
                        similar_score = distance_to_score(distance)
                        result_info['similar_score'] = similar_score
                        if similar_score >= self.match_score:
                            result_info['is_match'] = True
                        else:
                            result_info['is_match'] = False
                        ret_info['result'].append(result_info)
                        return jsonify(ret_info)
                    else:
                        ret_info['error'] = '该图片检测不到人脸'
                        return jsonify(ret_info)
                else:
                    ret_info['error'] = ret['error']
                    return jsonify(ret_info)
            except Exception as e:
                ret_info['error'] = str(e)
                return jsonify(ret_info)

        @app.route('/face/extract_feature', methods=["POST"])
        def extract_face_feature():
            ret_info = {
                'error': '',
                'result': []
            }
            # ----------------------- 参数校验 ---------------------------
            try:
                receive_param = request.get_json()
                receive_param = Val.ExtractFeature().load(receive_param)
                image = receive_param['image']
                detection_threshold = receive_param['detection_threshold']
                min_size = receive_param['min_size']
            except ValidationError as err:
                ret_info['error'] = str(err.messages)
                return jsonify(ret_info)
            try:
                image = decode_image(image)
                det_image = torch.cuda.FloatTensor(image)
                boxes, landmarks = self.face_det.detect(det_image, detection_threshold, min_size, size_flag=0)
                if len(boxes) != 0:
                    rgb_image = torch.cuda.FloatTensor(image)
                    rgb_image = (rgb_image - 127.5) * 0.0078125
                    rgb_image = rgb_image.permute(2, 0, 1).unsqueeze(0)
                    aligned_face = self.face_align.get(rgb_image, landmarks[0])
                    feature = self.face_rec.get_feature(aligned_face)
                    result_info = {
                        'face_coord': boxes[0],
                        'feature': json.dumps(feature.tolist())
                    }
                    ret_info['result'].append(result_info)
                    return jsonify(ret_info)
                else:
                    ret_info['error'] = '该图片检测不到人脸'
                    return jsonify(ret_info)
            except Exception as e:
                ret_info['error'] = str(e)
                return jsonify(ret_info)

        @app.route('/face/anti/detect', methods=["POST"])
        def anti_detect():
            ret_info = {
                'error': '',
                'result': []
            }
            # ------------------------- 参数校验 -----------------------
            try:
                receive_param = request.get_json()
                receive_param = Val.FaceAnti().load(receive_param)
                image = receive_param['image']
                db_name = receive_param['db_name']
                device = receive_param['device']
                feature_id = receive_param['feature_id']
                detection_threshold = receive_param['detection_threshold']
                min_size = receive_param['min_size']
            except ValidationError as err:
                ret_info['error'] = str(err.messages)
                return jsonify(ret_info)
            # -----------------------------------------------------------
            try:
                image = decode_image(image)
                height, width, _ = image.shape
                # 电脑端处理
                if device == 'computer':
                    if (height, width) == (720, 1280):
                        det_image = torch.cuda.FloatTensor(image)
                        boxes, landmarks = self.face_det.detect(det_image, detection_threshold, min_size, size_flag=1)
                        if len(boxes) != 0:
                            box = adjust_box(boxes[0])
                            if face_in_center(box):
                                anti_image = torch.cuda.FloatTensor(image)
                                # convert to BGR mode
                                anti_image = anti_image[:, :, (2, 1, 0)]
                                anti_image = anti_image.permute((2, 0, 1)).unsqueeze(0)
                                region_1 = get_anti_detect_region(anti_image, box, 2.7)
                                region_2 = get_anti_detect_region(anti_image, box, 4)
                                result_1 = self.face_anti_1.predict(region_1)
                                result_2 = self.face_anti_2.predict(region_2)
                                result = result_1 + result_2
                                anti_score, anti = torch.max(result, 1)
                                anti_score = anti_score.item() / 2
                                anti = anti.item()
                                result_info = {
                                    'feature_id': -1,
                                    'similar_score': -1,
                                    'face_coord': box,
                                    'db_name': db_name,
                                    'score': anti_score
                                }
                                # 如果为真人， 进行人脸搜索
                                if anti == 1 and anti_score > self.anti_score:
                                    result_info['real_face'] = True
                                    rgb_image = torch.cuda.FloatTensor(image)
                                    rgb_image = (rgb_image - 127.5) * 0.0078125
                                    rgb_image = rgb_image.permute(2, 0, 1).unsqueeze(0)
                                    aligned_face = self.face_align.get(rgb_image, landmarks[0])
                                    feature = self.face_rec.get_feature(aligned_face)
                                    # 全库搜索
                                    if feature_id == -1:
                                        data = {
                                            'db_name': db_name,
                                            'feature': json.dumps(feature.tolist()),
                                            'top': 1,
                                            'nprobe': self.nprobe
                                        }
                                        url = f'{self.faiss_base_url}/faiss/feature/search_fast'
                                        ret = requests.post(url=url, json=data).json()
                                        similar_score = ret['result'][0]['score']
                                        if similar_score >= self.match_score - 0.01:
                                            result_info['feature_id'] = ret['result'][0]['feature_id']
                                            result_info['similar_score'] = similar_score
                                    # 单人对比
                                    else:
                                        data = {
                                            'db_name': db_name,
                                            'feature_id': feature_id
                                        }
                                        url = f'{self.faiss_base_url}/faiss/feature/get'
                                        ret = requests.post(url=url, json=data).json()
                                        if ret['error'] == '':
                                            feature_0 = np.asarray(json.loads(ret['result'][0]['feature']), 'float32')
                                            distance = np.linalg.norm(feature - feature_0)
                                            similar_score = distance_to_score(distance)
                                            result_info['feature_id'] = feature_id
                                            result_info['similar_score'] = similar_score
                                        else:
                                            ret_info['error'] = ret['error']
                                            return jsonify(ret_info)
                                else:
                                    result_info['real_face'] = False
                                ret_info['result'].append(result_info)
                                return jsonify(ret_info)
                            else:
                                ret_info['error'] = '检测不到人脸'
                                return jsonify(ret_info)
                        else:
                            ret_info['error'] = '检测不到人脸'
                            return jsonify(ret_info)
                    else:
                        ret_info['error'] = '电脑端图片尺寸应为 HxW = 720x1280'
                        return jsonify(ret_info)
                # 手机端处理
                else:
                    if (height, width) == (960, 720):
                        det_image = torch.cuda.FloatTensor(image)
                        boxes, landmarks = self.face_det.detect(det_image, detection_threshold, min_size, size_flag=2)
                        if len(boxes) != 0:
                            box = adjust_box(boxes[0])
                            if face_in_phone_center(box):
                                anti_image = torch.cuda.FloatTensor(image)
                                # convert to BGR mode
                                anti_image = anti_image[:, :, (2, 1, 0)]
                                anti_image = anti_image.permute((2, 0, 1)).unsqueeze(0)
                                region_1 = get_anti_detect_region(anti_image, box, 2.7)
                                region_2 = get_anti_detect_region(anti_image, box, 4)
                                result_1 = self.face_anti_1.predict(region_1)
                                result_2 = self.face_anti_2.predict(region_2)
                                result = result_1 + result_2
                                anti_score, anti = torch.max(result, 1)
                                anti_score = anti_score.item() / 2
                                anti = anti.item()
                                result_info = {
                                    'feature_id': -1,
                                    'similar_score': -1,
                                    'face_coord': box,
                                    'db_name': db_name,
                                    'score': anti_score
                                }
                                # 如果为真人， 进行人脸搜索
                                if anti == 1 and anti_score > self.anti_score:
                                    result_info['real_face'] = True
                                    rgb_image = torch.cuda.FloatTensor(image)
                                    rgb_image = (rgb_image - 127.5) * 0.0078125
                                    rgb_image = rgb_image.permute(2, 0, 1).unsqueeze(0)
                                    aligned_face = self.face_align.get(rgb_image, landmarks[0])
                                    feature = self.face_rec.get_feature(aligned_face)
                                    # 全库搜索
                                    if feature_id == -1:
                                        data = {
                                            'db_name': db_name,
                                            'feature': json.dumps(feature.tolist()),
                                            'top': 1,
                                            'nprobe': self.nprobe
                                        }
                                        url = f'{self.faiss_base_url}/faiss/feature/search_fast'
                                        ret = requests.post(url=url, json=data).json()
                                        similar_score = ret['result'][0]['score']
                                        if similar_score >= self.match_score - 0.01:
                                            result_info['feature_id'] = ret['result'][0]['feature_id']
                                            result_info['similar_score'] = similar_score
                                    # 单人对比
                                    else:
                                        data = {
                                            'db_name': db_name,
                                            'feature_id': feature_id
                                        }
                                        url = f'{self.faiss_base_url}/faiss/feature/get'
                                        ret = requests.post(url=url, json=data).json()
                                        if ret['error'] == '':
                                            feature_0 = np.asarray(json.loads(ret['result'][0]['feature']), 'float32')
                                            distance = np.linalg.norm(feature - feature_0)
                                            similar_score = distance_to_score(distance)
                                            result_info['feature_id'] = feature_id
                                            result_info['similar_score'] = similar_score
                                        else:
                                            ret_info['error'] = ret['error']
                                            return jsonify(ret_info)
                                else:
                                    result_info['real_face'] = False
                                ret_info['result'].append(result_info)
                                return jsonify(ret_info)
                            else:
                                ret_info['error'] = '检测不到人脸'
                                return jsonify(ret_info)
                        else:
                            ret_info['error'] = '检测不到人脸'
                            return jsonify(ret_info)
                    else:
                        ret_info['error'] = '手机端图片尺寸应为 HxW = 960x720'
                        return jsonify(ret_info)
            except Exception as e:
                ret_info['error'] = str(e)
                return jsonify(ret_info)

        @app.route('/face/attribute/detect', methods=["POST"])
        def attribute_detect():
            ret_info = {
                'error': '',
                'result': []
            }
            # --------------------- 参数校验 ----------------------
            try:
                receive_param = request.get_json()
                receive_param = Val.AttributeDetect().load(receive_param)
                image = receive_param['image']
                detection_threshold = receive_param['detection_threshold']
                min_size = receive_param['min_size']
            except ValidationError as err:
                ret_info['error'] = str(err.messages)
                return jsonify(ret_info)
            # ------------------------------------------------------
            try:
                image = decode_image(image)
                det_image = torch.cuda.FloatTensor(image)
                boxes, landmarks = self.face_det.detect(det_image, detection_threshold, min_size, size_flag=0)
                if len(boxes) != 0:
                    rgb_image = torch.cuda.FloatTensor(image)
                    rgb_image = (rgb_image - 127.5) * 0.0078125
                    rgb_image = rgb_image.permute(2, 0, 1).unsqueeze(0)
                    for i, box in enumerate(boxes):
                        result_info = {}
                        # 头部姿势检测
                        pose_region = get_pose_region(rgb_image, box)
                        pitch, yaw, roll = self.pose_det.predict(pose_region)
                        result_info['face_coord'] = box
                        result_info['pitch'] = pitch
                        result_info['yaw'] = yaw
                        result_info['roll'] = roll
                        # 眼镜、口罩、质量 检测
                        face_region = get_glass_mask_region(rgb_image, box)
                        mask, mask_score = self.mask_det.predict(face_region)
                        result_info['mask'] = mask
                        result_info['mask_probability'] = score_mapping(mask_score)
                        glasses, glasses_score = self.glass_det.predict(face_region)
                        result_info['glasses'] = glasses
                        result_info['glasses_probability'] = score_mapping(glasses_score)
                        quality = self.quality_det.predict(face_region)
                        result_info['quality'] = quality
                        # 帽子检测
                        hat_region = get_hat_region(rgb_image, box)
                        hat, hat_score = self.hat_det.predict(hat_region)
                        result_info['hat'] = hat
                        result_info['hat_probability'] = score_mapping(hat_score)

                        aligned_face = self.face_align.get(rgb_image, landmarks[i])
                        # 年龄、性别 检测
                        age, age_score, gender, gender_score = self.age_gender_det.predict(aligned_face)
                        result_info['age'] = age
                        result_info['gender'] = gender
                        result_info['gender_probability'] = score_mapping(gender_score)
                        # 人种检测
                        race, race_score = self.race_det.predict(aligned_face)
                        result_info['race'] = race
                        result_info['race_score'] = race_score
                        # 表情检测
                        expression, expression_score = self.expresion_det.predict(aligned_face)
                        result_info['expression'] = expression
                        result_info['expression_probability'] = expression_score
                        ret_info['result'].append(result_info)
                    return jsonify(ret_info)
                else:
                    ret_info['error'] = '该图片检测不到人脸'
                    return jsonify(ret_info)
            except Exception as e:
                logger.exception('attribute detect 发生异常')
                ret_info['error'] = str(e)
                return jsonify(ret_info)

        @app.route('/face/detect', methods=["POST"])
        def face_detect():
            ret_info = {
                'error': '',
                'result': []
            }
            # ----------------------- 参数校验 ---------------------------
            try:
                receive_param = request.get_json()
                receive_param = Val.ExtractFeature().load(receive_param)
                image = receive_param['image']
                detection_threshold = receive_param['detection_threshold']
                min_size = receive_param['min_size']
            except ValidationError as err:
                ret_info['error'] = str(err.messages)
                return jsonify(ret_info)
            try:
                image = decode_image(image)
                det_image = torch.cuda.FloatTensor(image)
                boxes, landmarks = self.face_det.detect(det_image, detection_threshold, min_size, size_flag=0)
                if len(boxes) != 0:
                    result_info = {
                        'face_coord': boxes
                    }
                    ret_info['result'].append(result_info)
                    return jsonify(ret_info)
                else:
                    ret_info['error'] = '该图片检测不到人脸'
                    return jsonify(ret_info)
            except Exception as e:
                ret_info['error'] = str(e)
                return jsonify(ret_info)

        @app.route('/face/mask', methods=["POST"])
        def face_mask():
            ret_info = {
                'error': '',
                'result': []
            }
            # ------------------- 参数校验 ----------------
            try:
                receive_param = request.get_json()
                receive_param = Val.FaceMask().load(receive_param)
                image = receive_param['image']
                detection_threshold = receive_param['threshold']
                min_size = receive_param['size']
            except ValidationError as err:
                ret_info['error'] = str(err.messages)
                return jsonify(ret_info)
            # -----------------------------------------------
            try:
                image = decode_image(image)
                det_image = torch.cuda.FloatTensor(image)
                boxes, landmarks = self.face_det.detect(det_image, detection_threshold, min_size, size_flag=3)
                if len(boxes) != 0:
                    rgb_image = torch.cuda.FloatTensor(image)
                    rgb_image = (rgb_image - 127.5) * 0.0078125
                    rgb_image = rgb_image.permute(2, 0, 1).unsqueeze(0)
                    for box in boxes:
                        result_info = {}
                        mask_region = get_glass_mask_region(rgb_image, box)
                        mask, mask_score = self.mask_det.predict(mask_region)
                        if mask == 'mask':
                            result_info['mask'] = True
                        else:
                            result_info['mask'] = False
                        result_info['face_coord'] = box
                        result_info['mask_score'] = score_mapping(mask_score)
                        ret_info['result'].append(result_info)
                    return jsonify(ret_info)
                else:
                    ret_info['error'] = '该图片检测不到人脸'
                    return jsonify(ret_info)
            except Exception as e:
                ret_info['error'] = str(e)
                return jsonify(ret_info)

        @app.route('/face/identity/compare', methods=["POST"])
        def identity_compare():
            ret_info = {
                'error': '',
                'result': []
            }
            # --------------------- 参数校验 -------------------
            try:
                receive_param = request.get_json()
                receive_param = Val.IdentityCompare().load(receive_param)
                # image_a 证件头像
                image_a = receive_param['image_a']
                # image_b 拍摄的人脸
                image_b = receive_param['image_b']
                detection_threshold = receive_param['threshold']
                min_size = receive_param['size']
            except ValidationError as err:
                ret_info['error'] = str(err.messages)
                return jsonify(ret_info)
            # -----------------------------------------------------
            try:
                image_a = decode_image(image_a)
                det_image_a = torch.cuda.FloatTensor(image_a)
                boxes_a, landmarks_a = self.face_det.detect(det_image_a, detection_threshold, min_size, size_flag=4)

                image_b = decode_image(image_b)
                det_image_b = torch.cuda.FloatTensor(image_b)
                boxes_b, landmarks_b = self.face_det.detect(det_image_b, detection_threshold, min_size, size_flag=3)

                if len(boxes_a) != 0:
                    if len(boxes_b) != 0:
                        result_info = {}
                        max_face_index_b = 0
                        # 如果image_b 中检测到多张人脸，选取面积最大的人脸
                        if len(boxes_b) > 1:
                            max_area = 0
                            for i, box in enumerate(boxes_b):
                                area = (box[2] - box[0]) * (box[3] - box[1])
                                if area > max_area:
                                    max_area = area
                                    max_face_index_b = i
                        rgb_image_a = torch.cuda.FloatTensor(image_a)
                        rgb_image_a = (rgb_image_a - 127.5) * 0.0078125
                        rgb_image_a = rgb_image_a.permute(2, 0, 1).unsqueeze(0)

                        rgb_image_b = torch.cuda.FloatTensor(image_b)
                        rgb_image_b = (rgb_image_b - 127.5) * 0.0078125
                        rgb_image_b = rgb_image_b.permute(2, 0, 1).unsqueeze(0)

                        aligned_face_a = self.face_align.get(rgb_image_a, landmarks_a[0])
                        aligned_face_b = self.face_align.get(rgb_image_b, landmarks_b[max_face_index_b])
                        feature_a = self.face_rec.get_feature(aligned_face_a)
                        feature_b = self.face_rec.get_feature(aligned_face_b)
                        distance = np.linalg.norm(feature_a - feature_b)
                        similar_score = distance_to_score(distance)
                        result_info['similar_score'] = similar_score
                        result_info['face_coord'] = boxes_b[max_face_index_b]
                        # 对image_b 进行口罩检测
                        mask_region_b = get_glass_mask_region(rgb_image_b, boxes_b[max_face_index_b])
                        mask, _ = self.mask_det.predict(mask_region_b)
                        if mask == "mask":
                            result_info['mask'] = True
                        else:
                            result_info['mask'] = False
                        ret_info['result'].append(result_info)
                        return jsonify(ret_info)
                    else:
                        ret_info['error'] = 'image_b 检测不到人脸'
                        return jsonify(ret_info)
                else:
                    ret_info['error'] = 'image_a 检测不到人脸'
                    return jsonify(ret_info)
            except Exception as e:
                ret_info['error'] = str(e)
                return jsonify(ret_info)

        serve(app=app, host=self.flask_host, port=self.flask_port)


if __name__ == "__main__":
    service = FaceService(cfg=config)
    service.flask_run()

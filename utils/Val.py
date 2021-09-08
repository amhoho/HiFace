# -*- coding: UTF-8 -*-
# -------------------------------------------------------------
# @Author   :   Etpoem
# @Time     :   2020/8/6 11:03
# @Desc     :   
# -------------------------------------------------------------
import toml
from pathlib import Path
from marshmallow import Schema, fields, validate, EXCLUDE, ValidationError

cfg = toml.load(Path(__file__).parents[1] / 'config.toml')


class DbInsert(Schema):
    class Meta:
        unknown = EXCLUDE
    db_name = fields.String(required=True)
    feature_id = fields.Integer(required=True, validate=validate.Range(min=0, max=9223372036854775807))
    image = fields.String(required=True)
    info = fields.String(missing='')
    detection_threshold = fields.Float(missing=cfg['service']['detection_threshold'],
                                       validate=validate.Range(min=0.0, max=1.0))
    min_size = fields.Integer(missing=max(cfg['service']['min_width'], cfg['service']['min_height']),
                              validate=validate.Range(min=30, max=500))


class DbSearch(Schema):
    class Meta:
        unknown = EXCLUDE
    db_name = fields.String(required=True)
    image = fields.String(required=True)
    top = fields.Integer(missing=5)
    nprobe = fields.Integer(missing=cfg['faiss']['nprobe'],
                            validate=validate.Range(min=1, max=cfg['faiss']['nlist']))
    detection_threshold = fields.Float(missing=cfg['service']['detection_threshold'],
                                       validate=validate.Range(min=0.0, max=1.0))
    min_size = fields.Integer(missing=max(cfg['service']['min_width'], cfg['service']['min_height']),
                              validate=validate.Range(min=30, max=500))


class DbSet(Schema):
    class Meta:
        unknown = EXCLUDE
    minWidth = fields.Integer(required=True, validate=validate.Range(min=40, max=300))
    minHeight = fields.Integer(required=True, validate=validate.Range(min=40, max=300))
    detectionThreshold = fields.Float(required=True, validate=validate.Range(min=0.0, max=1.0))
    queueLength = fields.Integer(missing=0)
    fusion_mode = fields.Integer(missing=0)


class FaceAnti(Schema):
    class Meta:
        unknown = EXCLUDE
    image = fields.String(required=True)
    db_name = fields.String(missing=cfg['service']['default_login_db'])
    device = fields.String(missing='computer', validate=validate.OneOf(["computer", "phone"]))
    feature_id = fields.Integer(missing=-1, validate=validate.Range(min=-1, max=9223372036854775807))
    detection_threshold = fields.Float(missing=cfg['service']['detection_threshold'],
                                       validate=validate.Range(min=0.0, max=1.0))
    min_size = fields.Integer(missing=max(cfg['service']['min_width'], cfg['service']['min_height']),
                              validate=validate.Range(min=30, max=500))


class FaceVerify(Schema):
    class Meta:
        unknown = EXCLUDE
    db_name = fields.String(required=True)
    feature_id = fields.Integer(required=True, validate=validate.Range(min=0, max=9223372036854775807))
    image = fields.String(required=True)
    detection_threshold = fields.Float(missing=cfg['service']['detection_threshold'],
                                       validate=validate.Range(min=0.0, max=1.0))
    min_size = fields.Integer(missing=max(cfg['service']['min_width'], cfg['service']['min_height']),
                              validate=validate.Range(min=30, max=500))


class ExtractFeature(Schema):
    class Meta:
        unknown = EXCLUDE
    image = fields.String(required=True)
    detection_threshold = fields.Float(missing=cfg['service']['detection_threshold'],
                                       validate=validate.Range(min=0.0, max=1.0))
    min_size = fields.Integer(missing=max(cfg['service']['min_width'], cfg['service']['min_height']),
                              validate=validate.Range(min=30, max=500))


class AttributeDetect(Schema):
    class Meta:
        unknown = EXCLUDE
    image = fields.String(required=True)
    detection_threshold = fields.Float(missing=cfg['service']['detection_threshold'],
                                       validate=validate.Range(min=0.0, max=1.0))
    min_size = fields.Integer(missing=max(cfg['service']['min_width'], cfg['service']['min_height']),
                              validate=validate.Range(min=30, max=500))


class FaceMask(Schema):
    class Meta:
        unknown = EXCLUDE
    image = fields.String(required=True)
    threshold = fields.Float(missing=cfg['service']['detection_threshold'],
                             validate=validate.Range(min=0.0, max=1.0))
    size = fields.Integer(missing=max(cfg['service']['min_width'], cfg['service']['min_height']),
                          validate=validate.Range(min=30, max=500))


class IdentityCompare(Schema):
    class Meta:
        unknown = EXCLUDE
    image_a = fields.String(required=True)
    image_b = fields.String(required=True)
    threshold = fields.Float(missing=cfg['service']['detection_threshold'],
                             validate=validate.Range(min=0.0, max=1.0))
    size = fields.Integer(missing=max(cfg['service']['min_width'], cfg['service']['min_height']),
                          validate=validate.Range(min=30, max=500))

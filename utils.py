#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from typing import Literal
import math


def toform(y, form: Literal['xywh', 'x1y1x2y2']):
    assert form == 'xywh' or form == 'x1y1x2y2'
    y = tf.cast(y, tf.float64)
    if form == 'xywh':
        x1y1 = y[..., 0:2]
        x2y2 = y[..., 2:4]
        wh = x2y2 - x1y1
        xy = (x1y1 + x2y2)/2
        xywh = tf.concat([xy, wh], axis=-1)
        return xywh
    else:
        wh = y[..., 2:4]
        xy = y[..., 0:2]
        x1y1x2y2 = tf.concat([xy + wh/2, xy - wh/2], axis=-1)
        return x1y1x2y2


def yolo_head(y, anchors, input_shape=(416, 416)):
    '''return xywh, objectness, class of the bounding boxs'''
    anchors_num = anchors.shape[0]
    batch_size = y.shape[0]
    grids_w = y.shape[1]
    grids_h = y.shape[2]
    y = tf.reshape(y, [batch_size, grids_w, grids_h, anchors_num, -1])
    txty = y[..., 0:2]
    twth = y[..., 2:4]
    bwbh = anchors*tf.exp(twth)
    grid_w = input_shape[0] / grids_w
    grid_h = input_shape[1] / grids_h
    sigma_xy = tf.sigmoid(txty)
    grid = tf.meshgrid(range(grids_w), range(grids_h))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
    grid = tf.cast(grid, tf.float32)
    bxby = (grid + sigma_xy) * np.array([grid_w, grid_h])
    pred_box = tf.concat([bxby, bwbh], axis=-1)
    objectness = tf.sigmoid(y[..., 4])
    class_prob = tf.sigmoid(y[..., 5:])
    return pred_box, objectness, class_prob


def bbox_iou(box1, box2, form: Literal['xywh', 'x1y1x2y2'] = 'xywh', eps=1e-7):
    '''compute iou for two bboxs. inputs should be xywh'''
    assert form == 'xywh' or form == 'x1y1x2y2'
    if form == 'x1y1x2y2':
        box1, box2 = toform(box1, 'xywh'), toform(box2, 'xywh')
    x1, y1, w1, h1 = box1[..., :]
    x2, y2, w2, h2 = box2[..., :]
    union_w = tf.math.maximum((x1 + w1), (x2 + w2)) - tf.math.minimum(x1, x2)
    union_h = tf.math.maximum((y1 + h1), (y2 + h2)) - tf.math.minimum(y1, y2)
    W = w2 + w1 - union_w
    H = h2 + h1 - union_h
    Intersection = W*H
    Union = w2*h2 + w1*h1 - Intersection
    return Intersection / (Union + eps)


def bbox_diou(box1, box2, form: Literal['xywh', 'x1y1x2y2'] = 'xywh', eps=1e-7):
    assert form == 'xywh' or form == 'x1y1x2y2'
    if form == 'x1y1x2y2':
        box1, box2 = toform(box1, 'xywh'), toform(box2, 'xywh')
    x1, y1, w1, h1 = box1[..., :]
    x2, y2, w2, h2 = box2[..., :]
    union_w = tf.math.maximum((x1 + w1), (x2 + w2)) - tf.math.minimum(x1, x2)
    union_h = tf.math.maximum((y1 + h1), (y2 + h2)) - tf.math.minimum(y1, y2)
    W = w2 + w1 - union_w
    H = h2 + h1 - union_h
    Intersection = W*H
    Union = w1*h1 + w2*h2 - Intersection
    iou = Intersection / (Union + eps)
    d_center = tf.math.squared_difference(box1[..., :], box2[..., :])
    d_center = tf.math.reduce_sum(d_center, axis=-2)
    d_cover = tf.math.square(union_w)+tf.math.square(union_h)
    d = d_center/d_cover
    return iou-d


def bbox_ciou(box1, box2, form: Literal['xywh', 'x1y1x2y2'] = 'xywh', eps=1e-7):
    assert form == 'xywh' or form == 'x1y1x2y2'
    if form == 'x1y1x2y2':
        box1, box2 = toform(box1, 'xywh'), toform(box2, 'xywh')
    x1, y1, w1, h1 = box1[..., :]
    x2, y2, w2, h2 = box2[..., :]
    union_w = tf.math.maximum((x1 + w1), (x2 + w2)) - tf.math.minimum(x1, x2)
    union_h = tf.math.maximum((y1 + h1), (y2 + h2)) - tf.math.minimum(y1, y2)
    W = w2 + w1 - union_w
    H = h2 + h1 - union_h
    Intersection = W*H
    Union = w1*h1 + w2*h2 - Intersection
    iou = Intersection / (Union + eps)
    d_center = tf.math.squared_difference(box1[..., :], box2[..., :])
    d_center = tf.math.reduce_sum(d_center, axis=-2)
    d_cover = tf.math.square(union_w)+tf.math.square(union_h)
    d = d_center/d_cover
    diou = iou-d
    v = 4*tf.math.square(tf.math.atan(w1/(h1+eps)) -
                         tf.math.atan(w2/(h2+eps)))/(math.pi**2)
    alpha = v/(1-iou)+v+eps
    return diou-alpha*v


def ciou_loss(box1, box2, eps=1e-7):
    return 1 - bbox_ciou(box1, box2)


if __name__ == "__main__":
    box1 = tf.transpose(tf.constant(
        [[1, 3, 0, 0], [1, 3, 1, 1]], dtype=tf.float32))
    box2 = tf.transpose(tf.constant(
        [[1, 1, 1, 1], [1, 1, 1, 1]], dtype=tf.float32))
    print(bbox_ciou(box1, box2))

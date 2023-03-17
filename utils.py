#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


def extract(y, first, interval):
    '''extract data with fixed interval from a tensor'''
    iternum = y.shape[-1] // interval
    out = np.array([y[..., (first+i*interval)] for i in range(iternum)])
    out = np.transpose(out, [1, 2, 3, 0])
    return out


def yolo_head(y, anchors, input_shape=(416, 416)):
    '''return the true position of the bounding boxs'''
    anchors_num = anchors.shape[0]
    one_anchor_length = y.shape[-1] // anchors_num
    tx, ty, tw, th = extract(y, 0, one_anchor_length), extract(y, 1, one_anchor_length), extract(
        y, 2, one_anchor_length), extract(y, 3, one_anchor_length)
    print(tw)
    bw, bh = anchors[..., 0]*tf.exp(tw), anchors[..., 1]*tf.exp(th)
    batch_size = y.shape[0]
    grids_w = y.shape[1]
    grids_h = y.shape[2]
    grid_w = input_shape[0] // grids_w
    grid_h = input_shape[1] // grids_h
    sigma_x = tf.sigmoid(tx)
    sigma_y = tf.sigmoid(ty)
    print(sigma_x.shape)
    out_shape = list(y.shape)
    out_shape[-1] = 3
    out_shape.append(4)
    out = np.zeros(out_shape)
    print(out.shape)
    print(bw.shape)
    out[..., 2] = bw
    out[..., 3] = bh
    for i in range(grids_w):
        for j in range(grids_h):
            bx = grid_w*i + grid_w*sigma_x[:, i, j, :]
            by = grid_h*j + grid_h*sigma_y[:, i, j, :]
            out[:, i, j, :, 0], out[:, i, j, :, 1] = bx, by
    out = out.reshape([batch_size, grids_w, grids_h, -1])
    return out


def bbox_iou(box1, box2, eps=1e-7):
    '''compute iou for two bboxs. inputs should be xywh'''
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    union_w = tf.math.maximum((x1 + w1), (x2 + w2)) - min(x1, x2)
    union_h = tf.math.maximum((y1 + h1), (y2 + h2)) - min(y1, y2)
    W = w2 + w1 - union_w
    H = h2 + h1 - union_h
    Intersection = W*H
    Union = w2*h2 + w1*h1 - Intersection
    return Intersection / (Union + eps)


def loss(p, t, anchors, num_classes, threshold=.5):
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    for i in range(3):
        object_mask = t[i][..., 4:5]
        true_class_probs = t[i][..., 5:]


if __name__ == "__main__":
    predict = np.random.randn(2, 13, 13, 255)
    anchors = np.array([[1, 1], [2, 2], [3, 3]])
    print(yolo_head(predict, anchors).shape)

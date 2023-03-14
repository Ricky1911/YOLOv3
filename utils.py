#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

def yolo_head(y, anchors, input_shape=(416,416)):
    #return the true position of the bounding boxs
    tx, ty, tw, th = y[...,0], y[...,1], y[...,2], y[...,3]
    bw, bh = anchors[...,0]*tf.exp(tw), anchors[...,1]*tf.exp(th)
    grids_w = y.shape[1]
    grids_h = y.shape[2]
    grid_w = input_shape[0]//grids_w
    grid_h = input_shape[1]//grids_h
    sigma_x = tf.sigmoid(tx)
    sigma_y = tf.sigmoid(ty)
    out = np.zeros((..., 4))
    out[...,2] = bw
    out[...,3] = bh
    for i in range(grids_w):
        for j in range(grids_h):
            bx = grid_w*i + grid_w*sigma_x[...,i,j]
            by = grid_h*j + grid_h*sigma_y[...,i,j]
            out[...,i,j,0], out[...,i,j,1] = bx, by
    return out

def bbox_iou(box1, box2, eps=1e-7):
    #xywh
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    union_w = tf.math.maximum((x1 + w1), (x2 + w2)) - min(x1, x2)
    union_h = tf.math.maximum((y1 + h1), (y2 + h2)) - min(y1, y2)
    W = w2 + w1 - union_w
    H = h2 + h1 - union_h
    Intersection = W*H
    Union = w2*h2+w1*h1-Intersection
    return Intersection / (Union +eps)

def loss(p, t, anchors, num_classes, threshold=.5):
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
    
    for i in range(3):
        object_mask = t[i][..., 4:5]
        true_class_probs = t[i][..., 5:]
if __name__ == "__main__":
    box1=tf.constant([0.,0.,1.,1.])
    box2=tf.constant([-.5,0.,1.,1.])
    print(bbox_iou(box1,box2))

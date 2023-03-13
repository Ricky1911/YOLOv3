#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf
def bbox_iou(box1, box2, eps=1e-7):
    #xywh
    x1,y1,w1,h1 = box1
    x2,y2,w2,h2 = box2
    union_w = tf.math.maximum((x1 + w1), (x2 + w2)) - min(x1, x2)
    union_h = tf.math.maximum((y1+h1),(y2+h2))-min(y1,y2)
    W = w2+w1-union_w
    H = h2+h1-union_h
    Intersection = W*H
    Union = w2*h2+w1*h1-Intersection
    return Intersection / Union
if __name__ == "__main__":
    box1=tf.constant([0.,0.,1.,1.])
    box2=tf.constant([-.5,0.,1.,1.])
    print(bbox_iou(box1,box2))

#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : numpy_utils.py
# Author : Yikai Li
# Email  : liyikai98@gmail.com
# Date   : 2020-04-04
#
# Distributed under terms of the MIT license.
import numpy as np


def IoU(gts, predicts, threshold=128):
    """
    to calculate IoU
    :param threshold: numerical, a threshold for gray image to binary image
    :return:  IoU
    """
    intersection = 0.0
    union = 0.0

    for index in range(len(gts)):
        gt_img = (gts[index] >= threshold) * 1
        prob_img = (predicts[index] >= threshold) * 1

        intersection = intersection + np.sum(gt_img * prob_img)
        union = union + np.sum(gt_img) + np.sum(prob_img) - np.sum(gt_img * prob_img)

    iou = np.round(intersection / union, 4)
    return iou


def calc_resize_ratio(old_shape, new_shape):
    return np.array([new_shape[1] / old_shape[1], new_shape[0] / old_shape[0]], dtype=np.float32)


def calc_batch_size(memory_use, unit_size):
    return int(memory_use * 1e9 / (unit_size * 32))


def gen_batches(nr, batch_size):
    batch_starts = np.arange(0, nr, batch_size)
    batch_ends = batch_starts + batch_size
    batch_ends[-1] = nr
    return np.stack([batch_starts, batch_ends], axis=1)


def coordinate_system_transformation(coords, v1, v2):
    '''
    :param coords: shape: (N, 2)
    :param v1: shape: (2)
    :param v2: shape: (2)
    :return:
    '''
    A = np.stack([v1, v2], axis=1)
    return coords * np.linalg.inv(A)


def normal_diff_deg(n1, n2):
    mul = n1 @ n2
    if np.isclose(mul, 1):
        return 0
    return np.arccos(mul) * 180 / np.pi

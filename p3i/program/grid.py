#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : grid.py
# Author : Yikai Li
# Email  : liyikai98@gmail.com
# Date   : 2020-03-22
#
# Distributed under terms of the MIT license.
import sys

import cv2
import torch
import numpy as np

from p3i.utils.auxiliary import CANVAS_CORNER
from p3i.utils.image_editing import reduce_act_size, vis_activation, act2im
from p3i.utils.torch_utils import generate_possible_shifts, find_second_shift_by_angle, mask2ltrb, ltrb2roi

from p3i.utils.numpy_utils import gen_batches, calc_resize_ratio, calc_batch_size


class GridProgram(object):
    def __init__(self, resolution=None, base_point=None, first_shift=None, second_shift=None, loss=None,
                 grid_prog=None):
        if grid_prog is not None:
            self.resolution = grid_prog.resolution
            self.base_point = grid_prog.base_point
            self.first_shift = grid_prog.first_shift
            self.second_shift = grid_prog.second_shift
            self.loss = grid_prog.loss
        else:
            self.resolution = resolution
            self.base_point = base_point.cpu().numpy()
            self.first_shift = first_shift.cpu().numpy()
            self.second_shift = second_shift.cpu().numpy()
            self.loss = loss.cpu().numpy()

    def fit_resolution(self, target_resolution):
        resize_ratio = calc_resize_ratio(self.resolution, target_resolution)
        self.base_point = np.round(self.base_point * resize_ratio).astype(np.int32)
        self.first_shift *= resize_ratio
        self.second_shift *= resize_ratio

    def gen_ij(self, canvas_shape):
        vectors = np.array(CANVAS_CORNER) * np.array(canvas_shape[::-1]) - self.base_point
        canvas_corner_coord = np.linalg.inv(np.stack([self.first_shift, self.second_shift], axis=1)) @ vectors.T
        # first: i; second: j;
        i_min, j_min = np.floor(canvas_corner_coord.min(axis=1)).astype(np.int)
        i_max, j_max = np.ceil(canvas_corner_coord.max(axis=1)).astype(np.int)
        return i_min, i_max, j_min, j_max

    def draw(self, image, color=(255, 255, 0), thickness=2):
        self.fit_resolution(image.shape[:2])
        canvas = image[:, :, :-1].copy()
        i_min, i_max, j_min, j_max = self.gen_ij(canvas.shape[:2])
        i_base_points = self.base_point + np.arange(i_min, i_max)[..., np.newaxis] * self.first_shift
        i_lines = np.concatenate((i_base_points, i_base_points), axis=1)
        i_lines[:, :2] += j_min * self.second_shift
        i_lines[:, 2:] += j_max * self.second_shift
        j_base_points = self.base_point + np.arange(j_min, j_max)[..., np.newaxis] * self.second_shift
        j_lines = np.concatenate((j_base_points, j_base_points), axis=1)
        j_lines[:, :2] += i_min * self.first_shift
        j_lines[:, 2:] += i_max * self.first_shift
        lines = np.round(np.concatenate((i_lines, j_lines))).astype(np.int32)
        for line in lines:
            cv2.line(canvas, (line[0], line[1]), (line[2], line[3]), color=color, thickness=thickness)
        return np.concatenate([canvas, image[:, :, -1:]], axis=2)

    def __str__(self):
        return f"base point: {self.base_point}\n" \
               f"first shift: {self.first_shift}\n" \
               f"second shift: {self.second_shift}\n" \
               f"loss: {self.loss:.2e}"


def program_search(args, activation_, return_type='prog'):
    # extract and resize foreground
    ltrb = mask2ltrb(activation_[-1])
    roi_ = ltrb2roi(ltrb, activation_)
    roi_, resize_ratio = reduce_act_size(roi_, args.roi_width)

    # brute force search
    possible_shifts = generate_possible_shifts(roi_.shape[1:], args.repeat_range, activation_.device)
    losses, possible_shifts = compute_loss(args, roi_, possible_shifts)

    # generate grid program
    sorted_index = torch.argsort(losses)
    sorted_shifts = possible_shifts[sorted_index].type(torch.float32)
    second_index = find_second_shift_by_angle(sorted_shifts)
    if second_index is None:
        return None
    best_loss = losses[sorted_index[0]] + losses[sorted_index[second_index]]

    if return_type == 'loss':
        return best_loss
    return GridProgram(resolution=activation_.shape[1:],
                       base_point=ltrb[:2],
                       first_shift=sorted_shifts[0] / resize_ratio,
                       second_shift=sorted_shifts[second_index] / resize_ratio,
                       loss=best_loss)


def compute_loss(args, activation_, possible_shifts):
    '''
    :param activation_: activation_with_mask gpu tensor shape: (layer_nr, h, w)
    :param possible_shifts: gpu tensor shape: (possible_shift_nr, 2)
    :return: batch loss: shape (bs, 1)
    '''
    # pad canvas
    act_c_, act_h, act_w = activation_.shape
    pad_x, pad_y = act_w // args.repeat_range[0] + 2, act_h // args.repeat_range[0] + 2
    activation_pad_ = torch.zeros((act_c_, act_h + pad_y, act_w + pad_x * 2),
                                  dtype=activation_.dtype, device=activation_.device)
    activation_pad_[:, :act_h, pad_x:pad_x + act_w] = activation_

    # generate canvas index
    y_index, x_index = torch.meshgrid([
        torch.arange(act_h, device=activation_.device),
        torch.arange(pad_x, pad_x + act_w, device=activation_.device)
    ])  # shape: (h, w)
    index = torch.stack([x_index, y_index], dim=2)  # shape: (h, w, 2)

    # batch compute overlap
    possible_shifts = possible_shifts.unsqueeze(1).unsqueeze(1)  # shape: (nr, 1, 1, 2)
    overlaps = torch.zeros(possible_shifts.shape[0], device=activation_.device)
    batches = gen_batches(possible_shifts.shape[0], batch_size=calc_batch_size(args.memory_use, act_h * act_w))
    for batch_start, batch_end in batches:
        index_shift = possible_shifts[batch_start:batch_end] + index  # shape: (bs, h, w, 2)
        batch_mask = activation_pad_[-1, index_shift[..., 1], index_shift[..., 0]]
        diff_mask = batch_mask * activation_[-1] / 255
        overlaps[batch_start:batch_end] = diff_mask.sum(dim=[1, 2])

    # batch compute loss
    possible_shifts = possible_shifts[(overlaps / activation_[-1].sum()) > args.min_overlap]
    losses = torch.zeros(possible_shifts.shape[0], device=activation_.device)
    batches = gen_batches(possible_shifts.shape[0], batch_size=calc_batch_size(args.memory_use, activation_.numel()))
    for batch_start, batch_end in batches:
        index_shift = possible_shifts[batch_start:batch_end] + index  # shape: (bs, h, w, 2)
        batch_activation_ = activation_pad_[:, index_shift[..., 1], index_shift[..., 0]].transpose(0, 1)
        diff = batch_activation_[:, :-1] - activation_[:-1]  # shape: (bs, layer_nr, h, w)
        diff_mask = batch_activation_[:, -1:] * activation_[-1:] / 65025  # shape: (bs, 1, h, w)
        pow_diff = torch.pow(diff * diff_mask, 2)  # shape: (bs, layer_nr, h, w)
        losses[batch_start:batch_end] = torch.sum(pow_diff, dim=[1, 2, 3]) / \
                                        torch.sum(diff_mask, dim=[1, 2, 3])
    return losses, possible_shifts.squeeze(1).squeeze(1)

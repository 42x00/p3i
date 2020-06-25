#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : torch_utils.py
# Author : Yikai Li
# Email  : liyikai98@gmail.com
# Date   : 2020-04-04
#
# Distributed under terms of the MIT license.

import math

import torch


def sample_sphere(v, alpha, num_pts):
    v1 = orth(v)
    v2 = torch.cross(v, v1)
    v, v1, v2 = v.unsqueeze(1), v1.unsqueeze(1), v2.unsqueeze(1)
    indices = torch.arange(num_pts, dtype=torch.float32, device=v.device) + 1.
    phi = torch.acos(1 + (math.cos(alpha) - 1) * indices / num_pts)
    theta = math.pi * (1 + 5 ** 0.5) * indices
    r = torch.sin(phi)
    return (v * torch.cos(phi) + r * (v1 * torch.cos(theta) + v2 * torch.sin(theta))).transpose(0, 1)


def orth(v):
    if torch.abs(v[0]) < torch.abs(v[1]):
        o = torch.tensor([0.0, -v[2], v[1]], dtype=v.dtype, device=v.device)
    else:
        o = torch.tensor([-v[2], 0.0, v[0]], dtype=v.dtype, device=v.device)
    return o / torch.norm(o)


def generate_possible_shifts(roi_shape, repeat_range=(3, 10), device='cpu'):
    dxs, dys = torch.meshgrid([
        torch.arange(-roi_shape[1] // repeat_range[0], roi_shape[1] // repeat_range[0], device=device),
        torch.arange(0, roi_shape[0] // repeat_range[0], device=device)
    ])
    possible_shifts = torch.stack([dxs.flatten(), dys.flatten()], dim=1)
    # ignore minor shift
    select = (torch.abs(possible_shifts[:, 0]) > roi_shape[1] // repeat_range[1]) | \
             (possible_shifts[:, 1] > roi_shape[0] // repeat_range[1])
    possible_shifts = possible_shifts[select]
    return possible_shifts


def find_second_shift_by_angle(sorted_shifts, minimum_angle=44):
    sorted_thetas = torch.atan2(sorted_shifts[:, 1], sorted_shifts[:, 0]) * 180 / math.pi
    sorted_angle = torch.abs(sorted_thetas - sorted_thetas[0])
    select = (sorted_angle > minimum_angle) & (sorted_angle < 180 - minimum_angle)
    select_indexes = select.nonzero()
    if select_indexes.shape[0]:
        return select_indexes[0][0]
    return None


def calc_intersect_point_of_line_and_plane(lines_points, lines_directions, plane_points, plane_normals):
    '''
    :param lines_points: shape: (N, 3)
    :param lines_directions: shape: (N, 3) len > 1 is ok
    :param plane_points: shape: (N, 3)
    :param plane_normals: shape: (N, 3)
    :return:
    '''
    OA = plane_points - lines_points
    OH = (OA * plane_normals).sum(dim=1)
    OD = (lines_directions * plane_normals).sum(dim=1)
    return lines_points + (OH / OD).unsqueeze(1) * lines_directions


def mask2ltrb(mask: torch.Tensor):
    fg_coord = mask.nonzero()
    brtl = torch.cat([fg_coord.max(dim=0)[0], fg_coord.min(dim=0)[0]])
    return brtl[[3, 2, 1, 0]]


def ltrb2roi(ltrb, activation: torch.Tensor):
    ndim = activation.ndimension()
    if ndim == 3:
        return activation[:, ltrb[1]:ltrb[3] + 1, ltrb[0]:ltrb[2] + 1]
    elif ndim == 2:
        return activation[ltrb[1]:ltrb[3] + 1, ltrb[0]:ltrb[2] + 1]
    else:
        return None


def reduce_shape_by_foreground(mask, target_width):
    ltrb = mask2ltrb(mask)
    resize_ratio = target_width / max(ltrb[3] - ltrb[1], ltrb[2] - ltrb[0]).type(torch.float)
    return mask.shape[0] * resize_ratio, mask.shape[1] * resize_ratio

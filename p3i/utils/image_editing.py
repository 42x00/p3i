#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : image_editing.py
# Author : Yikai Li
# Email  : liyikai98@gmail.com
# Date   : 2020-04-04
#
# Distributed under terms of the MIT license.
import cv2
import numpy as np
import torch
from torch.nn.functional import grid_sample

from p3i.utils.torch_utils import calc_intersect_point_of_line_and_plane


def add_mask(image, mask):
    return np.concatenate([image, mask[..., np.newaxis]], axis=2)


def im2rgba(im):
    if im.ndim == 2:
        return cv2.cvtColor(im, cv2.COLOR_GRAY2RGBA)
    assert im.ndim == 3, 'Not an image.'
    if im.shape[-1] == 3:
        return add_mask(im, np.zeros(im.shape[:2]) + 255)
    return im


def im2act(im, act_mode='RGB', model_ckpt=None):
    if act_mode == 'RGB':
        return torch.tensor(im.transpose(2, 0, 1), dtype=torch.float32)
    elif act_mode == 'AlexNet':
        from PIL import Image
        from p3i.models.model_def import get_model_def
        from .image_transforms import PadMultipleOf
        import torchvision.transforms as T
        import numpy as np
        model_def = get_model_def('alexnet')
        model = model_def.get_model(use_gpu=False, ckpt_path=model_ckpt)
        image_transform = T.Compose([
            T.Resize(224),
            PadMultipleOf(32),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        with model_def.hook_model(model) as extractor:
            image = Image.fromarray(im[:, :, :-1]).convert('RGB')
            image = image_transform(image).unsqueeze(0)
            activation = extractor(image)[0][0]
        image_shape = np.array(im.shape[:2])
        resize_f = image_shape.min() / 56
        new_shape = (image_shape / resize_f).astype(np.int32)
        activation_mask = torch.tensor(cv2.resize(im[:, :, -1], (new_shape[1], new_shape[0])), dtype=torch.float32)
        return torch.cat([activation[:, :new_shape[0], :new_shape[1]], activation_mask.unsqueeze(0)], dim=0)


def act2im(act, act_mode='RGB'):
    if act_mode == 'RGB':
        return act.cpu().numpy().transpose(1, 2, 0)


def reduce_act_size(activation_, target_width):
    return resize_act(activation_, (min(target_width, activation_.shape[1]),
                                    min(target_width, activation_.shape[2])))


def resize_act(activation_, target_shape):
    '''
    :param activation_:
    :param target_shape: tuple
    :return:
    '''
    index1, index0 = torch.meshgrid([torch.linspace(-1., 1., target_shape[0], device=activation_.device),
                                     torch.linspace(-1., 1., target_shape[1], device=activation_.device)])
    grids = torch.stack([index0, index1], dim=2)
    resized_activation_ = grid_sample(
        activation_.unsqueeze(0),
        grids.unsqueeze(0)
    ).squeeze(0)
    resized_activation_[torch.isnan(resized_activation_)] = 0.
    return resized_activation_, \
           torch.tensor([target_shape[1] / activation_.shape[2],
                         target_shape[0] / activation_.shape[1]],
                        dtype=torch.float32, device=activation_.device)


def apply_mask(im):
    if im.dtype == np.uint8:
        im = im.astype(np.float32)
    return im[..., :-1] * im[..., -1:] / 255


def vis_image(im):
    if im.shape[-1] == 4:
        im = apply_mask(im)
    if im.max() > 1:
        cv2.imshow('image', im / 255)
    else:
        cv2.imshow('image', im)
    cv2.waitKey(0)


def vis_activation(act, act_mode='RGB'):
    vis_image(act2im(act, act_mode))


def rectify_perspective(image, normal, pad_ratio=1.5):
    activation_ = im2act(im2rgba(image))
    canvas_pcl = canvas(activation_.shape[1:], pad_ratio=pad_ratio)
    plane_pcl = canvas_pcl @ n2rot(torch.tensor(normal))
    plane_pcl[:, :, 2] += 1.
    plane_activation_ = pcl2act(plane_pcl, activation_)
    return act2im(plane_activation_)


def restore_perspective(plane_image, normal, pad_ratio=1.5):
    # preparation
    plane_activation_ = im2act(im2rgba(plane_image))
    origin = torch.zeros(3, dtype=torch.float32).unsqueeze(0)
    plane_center = torch.tensor([0, 0, 1], dtype=torch.float32).unsqueeze(0)
    plane_normal = torch.tensor(normal, dtype=torch.float32).unsqueeze(0)
    rot = n2rot(plane_normal[0])
    new_axis = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32) @ rot

    # generate canvas point cloud
    canvas_h, canvas_w = plane_activation_.shape[1:]
    index1, index0 = torch.meshgrid([torch.linspace(-pad_ratio, pad_ratio, canvas_h),
                                     torch.linspace(-pad_ratio, pad_ratio, canvas_w)])
    canvas_pcl = torch.stack([index0, index1, torch.ones_like(index1)], dim=2).reshape((-1, 3))
    point_nr = canvas_pcl.shape[0]

    # ray tracing
    intersect_points = calc_intersect_point_of_line_and_plane(origin.expand(point_nr, 3),
                                                              canvas_pcl,
                                                              plane_center.expand(point_nr, 3),
                                                              plane_normal.expand(point_nr, 3)) - plane_center
    grids = (intersect_points @ new_axis.transpose(0, 1)).reshape((canvas_h, canvas_w, 2)) / pad_ratio
    activation_ = grid_sample(plane_activation_.unsqueeze(0), grids.unsqueeze(0)).squeeze(0)
    activation_[torch.isnan(activation_)] = 0
    return act2im(activation_)


def remove_pad(image, image_pad):
    image, image_pad = im2rgba(image), im2rgba(image_pad)
    h, w = image.shape[:2]
    pad_h, pad_w = image_pad.shape[:2]
    pad_x, pad_y = (pad_w - w) // 2, (pad_h - h) // 2
    foreground_image = image_pad[pad_y:pad_y + h, pad_x:pad_x + w]
    alpha = (255 - foreground_image[:, :, -1:]) * image[:, :, -1:] / 65025
    return image * alpha + foreground_image * (1 - alpha)


def canvas(canvas_shape, pad_ratio=1.5, device='cpu'):
    h, w = canvas_shape
    index1, index0 = torch.meshgrid([torch.linspace(-pad_ratio, pad_ratio, int(h * pad_ratio), device=device),
                                     torch.linspace(-pad_ratio, pad_ratio, int(w * pad_ratio), device=device)])
    canvas_pcl = torch.stack([index0, index1, torch.zeros_like(index1, device=device)], dim=2)
    return canvas_pcl


def n2rot(n):
    kmat = torch.tensor([
        [0, 0, -n[0]],
        [0, 0, -n[1]],
        [n[0], n[1], 0]
    ], dtype=n.dtype, device=n.device)
    return torch.eye(3, dtype=n.dtype, device=n.device) + kmat + kmat @ kmat / (1 + n[2])


def pcl2act(plane_pcl, activation_):
    '''
    :param plane_pcl: tensor shape: (c_h, c_w, 1)
    :param activation_: tensor shape: (bs, l_nr + 1, c_h, c_w)
    :return:
    '''
    k = 1. / plane_pcl[:, :, 2:]  # shape: (c_h, c_w, 1)
    grids = (plane_pcl * k)[:, :, :2]  # shape: (c_h, c_w, 2)
    plane_activation_ = grid_sample(
        activation_.unsqueeze(0),
        grids.unsqueeze(0)
    ).squeeze(0)
    plane_activation_[torch.isnan(plane_activation_)] = 0.
    return plane_activation_

#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : p3i.py
# Author : Yikai Li
# Email  : liyikai98@gmail.com
# Date   : 2020-03-22
#
# Distributed under terms of the MIT license.
import torch
from tqdm import tqdm

from p3i.program import grid
from p3i.program.grid import GridProgram
from p3i.utils.image_editing import rectify_perspective, restore_perspective, remove_pad, canvas, n2rot, pcl2act, \
    vis_activation, vis_image
from p3i.utils.torch_utils import sample_sphere, reduce_shape_by_foreground

from p3i.utils.auxiliary import MULTIRES


class PerspectivePlaneProgram(GridProgram):
    def __init__(self, grid_prog=None, normal=None, pad_ratio=None):
        super().__init__(grid_prog=grid_prog)
        self.normal = normal.cpu().numpy()
        self.pad_ratio = pad_ratio

    def rectify(self, image):
        return rectify_perspective(image, self.normal, self.pad_ratio)

    def restore(self, image, pad_ratio=None):
        if pad_ratio is None:
            pad_ratio = self.pad_ratio
        return restore_perspective(image, self.normal, pad_ratio)

    def draw(self, image, color=(255, 255, 0), thickness=2):
        rectified_image_ = self.rectify(image)
        rectified_image_with_prog_ = super().draw(rectified_image_, thickness=thickness)
        image_with_prog_pad_ = self.restore(rectified_image_with_prog_)
        image_with_prog_ = remove_pad(image, image_with_prog_pad_)
        return image_with_prog_, rectified_image_with_prog_

    def __str__(self):
        return f'normal: {self.normal}\n' + super().__str__()


def program_search(args, activation_, return_type='prog'):
    # coarse-to-fine search
    possible_normal = torch.tensor([0, 0, 1], dtype=torch.float32, device=activation_.device)
    for res in range(args.coarse_to_fine):
        if args.verbosity:
            print(f'coarse-to-fine search: {res}')
        possible_normal = sample_sphere(possible_normal, MULTIRES[res], args.sample_nr)
        losses = compute_loss(args, activation_, possible_normal, return_type='loss')
        best_loss_index = torch.argmin(losses)
        possible_normal = possible_normal[best_loss_index]

    if return_type == 'loss':
        return losses[best_loss_index]

    # generate perspective plane program
    if args.verbosity:
        print(f'generating program...')
    grid_prog = compute_loss(args, activation_, possible_normal.unsqueeze(0), return_type='prog')
    return PerspectivePlaneProgram(grid_prog=grid_prog,
                                   normal=possible_normal,
                                   pad_ratio=args.pad_ratio)


def compute_loss(args, activation_, possible_normals, return_type='loss'):
    if return_type == 'loss':
        losses = torch.zeros(possible_normals.shape[0], device=activation_.device) + 1e8

    # perspective correction preparation
    reduced_shape = reduce_shape_by_foreground(activation_[-1], args.max_width)
    valid_pixel_nr = activation_[-1].sum() / activation_[-1].numel() * reduced_shape[0] * reduced_shape[1]
    canvas_pcl = canvas(canvas_shape=reduced_shape, pad_ratio=args.pad_ratio, device=activation_.device)

    if args.verbosity == 2 and possible_normals.shape[0] > 1:
        possible_normals = tqdm(possible_normals)
    for normal_index, normal in enumerate(possible_normals):
        # perspective correction
        plane_pcl = canvas_pcl @ n2rot(normal)
        plane_pcl[:, :, 2] += 1.
        plane_activation_ = pcl2act(plane_pcl, activation_)
        if plane_activation_[-1].sum() < valid_pixel_nr:
            continue
        # search after correction
        ret = grid.program_search(args, plane_activation_, return_type=return_type)
        if return_type == 'prog':
            return ret
        if ret is None:
            continue
        losses[normal_index] = ret
    return losses

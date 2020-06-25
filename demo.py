#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : demo.py
# Author : Yikai Li
# Email  : liyikai98@gmail.com
# Date   : 2020-03-21
#
# Distributed under terms of the MIT license.
'''
Usage:
    python demo.py [options] --model_ckpt <model> --input <image> --output_dir <dir>

Arguments:
    <model>                 Path to pre-trained model
    <image>                 Path to image
    <dir>                   Path to output directory

Options:
    --device                Specify GPU device to use [default: cpu]
    --verbosity             Control output (0 = none, 1 = medium, 2 = max)
'''
import cv2

from p3i.config.p3i_options import P3IOptions
from p3i.program import p3i
from p3i.utils.image_editing import im2act, im2rgba
from p3i.utils.vis import visualize_and_dump


def main():
    args = P3IOptions().parse()
    image = cv2.imread(args.input, -1)
    activation_ = im2act(im2rgba(image), args.act_mode, args.model_ckpt)
    p3i_prog = p3i.program_search(args, activation_.to(device=args.device), return_type='prog')
    visualize_and_dump(args, image, p3i_prog)


if __name__ == '__main__':
    main()

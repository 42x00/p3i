#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : vis.py
# Author : Yikai Li
# Email  : liyikai98@gmail.com
# Date   : 2020-04-04
#
# Distributed under terms of the MIT license.
import os
import pickle

import cv2


def visualize_and_dump(args, image, prog):
    if args.verbosity:
        print('-- perspective plane program  --')
        print(prog)
        print('--------------------------------')

    # dump
    os.system(f'mkdir -p {args.output_dir}')
    args.image_name = os.path.basename(args.input).split('.')[0]
    dump_path = os.path.join(args.output_dir, f'{args.image_name}_prog.pkl')
    with open(dump_path, 'wb') as fw:
        pickle.dump(prog, fw, -1)
    if args.verbosity:
        print(f'{dump_path} dumped!')

    # vis
    image_with_prog, rectified_image = prog.draw(image)
    cv2.imwrite(os.path.join(args.output_dir, f'{args.image_name}_prog.png'), image_with_prog)
    cv2.imwrite(os.path.join(args.output_dir, f'{args.image_name}_rectified.png'), rectified_image)

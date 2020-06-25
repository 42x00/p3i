#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : p3i_options.py
# Author : Yikai Li
# Email  : liyikai98@gmail.com
# Date   : 2020-03-22
#
# Distributed under terms of the MIT license.
import argparse


class P3IOptions(object):
    @staticmethod
    def initialize(parser):
        # io
        parser.add_argument('--input', type=str, default=None)
        parser.add_argument('--output_dir', type=str, default='./results')
        parser.add_argument('--model_ckpt', type=str, default=None)
        # hyperparameter
        parser.add_argument('--max_width', type=int, default=256)
        parser.add_argument('--roi_width', type=int, default=48)
        parser.add_argument('--min_overlap', type=float, default=0.44)
        parser.add_argument('--pad_ratio', type=float, default=1.5)
        parser.add_argument('--repeat_range', type=tuple, default=(3, 10))
        parser.add_argument('--sample_nr', type=int, default=256)
        parser.add_argument('--coarse_to_fine', type=int, default=3)
        # auxiliary
        parser.add_argument('--device', type=str, default='cpu')
        parser.add_argument('--act_mode', type=str, default='AlexNet')
        parser.add_argument('--memory_use', type=float, default=6)
        parser.add_argument('--verbosity', type=int, default=2)
        parser.add_argument('--image_name', type=str, default=None)
        return parser

    @staticmethod
    def build_reliance(args):
        if args.device != 'cpu':
            args.device = f'cuda:{int(args.device)}'
        if args.model_ckpt is None:
            args.act_mode = 'RGB'
        if args.verbosity:
            print(f'processing image: {args.input}')
            print(f'use device: {args.device}')
            print(f'activation mode: {args.act_mode}')
        return args

    def parse(self):
        parser = argparse.ArgumentParser()
        parser = self.initialize(parser)
        return self.build_reliance(parser.parse_args())

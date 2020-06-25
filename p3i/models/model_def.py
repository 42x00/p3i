#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : model_def.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/25/2018
#
# This file is part of nimpi.
# Distributed under terms of the MIT license.

import contextlib

__all__ = [
    'get_available_models', 'get_model_def',
    'ModelDef', 'AlexNetModelDef', 'AlexNet5ModelDef', 'AlexNet5Resize256x256ModelDef', 'ResNet34ModelDef',
    'FeatureExtractor', 'HookBasedFeatureExtractor'
]


def get_available_models():
    return ['alexnet', 'alexnet5', 'alexnet5resize256x256']


def get_model_def(name):
    if name == 'alexnet':
        return AlexNetModelDef()
    elif name == 'alexnet5':
        return AlexNet5ModelDef()
    elif name == 'alexnet5resize256x256':
        return AlexNet5Resize256x256ModelDef()
    else:
        raise NotImplementedError('Unknown model name: {}.'.format(name))


class ModelDef(object):
    def __init__(self):
        super().__init__()

    def get_model(self, use_gpu, ckpt_path=None):
        model = self._get_model(ckpt_path)
        model.eval()
        if use_gpu:
            model.cuda()
        return model

    def _get_model(self, ckpt_path=None):
        raise NotImplementedError()


class FeatureExtractor(object):
    pass


class HookBasedFeatureExtractor(FeatureExtractor):
    def __init__(self, model, activations):
        super().__init__()
        self.model = model
        self.activations = activations

    def __call__(self, input):
        self.model(input)
        return self.activations.copy()


class DecoratorBasedFeatureExtractor(FeatureExtractor):
    def __init__(self, model, used_names):
        super().__init__()
        self.model = model
        self.used_names = used_names

    def __call__(self, input):
        output = self.model(input)
        output = {k: v for k, v in output.items() if k in self.used_names}
        assert len(output) == len(self.used_names)
        return output


class AlexNetModelDef(ModelDef):
    input_size = 224

    nr_convs = 1
    conv_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5'][:1]
    conv_dims = [64, 384, 256, 256, 256][:1]
    conv_downsamples = [4, 8, 16, 16, 16][:1]
    peak_window_size = [11, 7, 3, 3, 3][:1]
    disp_alpha = [5.0, 7.0, 15.0, 15.0, 15.0][:1]

    _conv_layer_ids = [0, 3, 6, 8, 10][:1]

    def _get_model(self, ckpt_path=None):
        from .alexnet import alexnet
        assert ckpt_path is not None
        return alexnet(True, ckpt_path=ckpt_path)

    @contextlib.contextmanager
    def hook_model(self, model):
        activations = [None for _ in range(self.nr_convs)]
        handles = [None for _ in range(self.nr_convs)]

        for i in range(self.nr_convs):
            self._set_hook(model, i, activations, handles)

        yield HookBasedFeatureExtractor(model, activations)

        for h in handles:
            h.remove()

    def _set_hook(self, model, i, activations, handles):
        def fetch(self, input, output):
            activations[i] = output.data.clone()

        handles[i] = model.features[self._conv_layer_ids[i]].register_forward_hook(fetch)


class AlexNet5ModelDef(AlexNetModelDef):
    input_size = 256
    nr_convs = 5
    conv_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
    conv_dims = [64, 192, 384, 256, 256]
    conv_downsamples = [4, 8, 16, 16, 16]
    peak_window_size = [11, 7, 3, 3, 3]
    disp_alpha = [5.0, 7.0, 15.0, 15.0, 15.0]

    _conv_layer_ids = [0, 3, 6, 8, 10]


class AlexNet5Resize256x256ModelDef(AlexNet5ModelDef):
    input_size = (256, 256)


class ResNet34ModelDef(ModelDef):
    input_size = 224
    nr_convs = 5
    conv_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
    conv_dims = [64, 256, 512, 1024, 2048]
    conv_downsamples = [2, 4, 8, 16, 32]
    peak_window_size = [8, 6, 4, 2, 1]

    def _get_model(self, ckpt_path=None):
        from .resnet import ResNet34FeatureExtractor
        assert ckpt_path is None
        return ResNet34FeatureExtractor()

    def hook_model(self, model):
        yield DecoratorBasedFeatureExtractor(model, self.conv_names)

#!/usr/bin/env python
# encoding: utf-8

import torch.nn as nn


# input: batch_size * nc * 64 * 64
# output: batch_size * k * 1 * 1
class Encoder(nn.Module):
    def __init__(self, input_size, layer_size, hidden_dim):
        super(Encoder, self).__init__()

        # input is isize
        main = nn.Sequential()

        main.add_module('initial_layer_{0}-{1}'.format(input_size, layer_size),
                        nn.Linear(input_size, layer_size, bias=True))
        main.add_module('initial_relu_{0}'.format(layer_size),
                        nn.LeakyReLU(0.2, inplace=True))  # 0.2 is starting negative slope

        for _ in range(2):  # these are the in between layers, go down to the latent dimension?
            layer_size = layer_size * 2
            main.add_module('middle_{0}-{1}_linear'.format(layer_size // 2, layer_size),
                            nn.Linear(layer_size // 2, layer_size, bias=True))
            main.add_module('middle_{0}_batchnorm'.format(layer_size),
                            nn.BatchNorm1d(layer_size))
            main.add_module('middle_{0}_relu'.format(layer_size),
                            nn.LeakyReLU(0.2, inplace=True))

        main.add_module('final_{0}-{1}_conv'.format(layer_size, hidden_dim),
                        nn.Linear(layer_size, hidden_dim, bias=True))  # final outputs to the latent dimension
        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output


# input: batch_size * k * 1 * 1 because k is the latent dimension
# output: batch_size * nc * image_size * image_size
class Decoder(nn.Module):
    def __init__(self, input_size, layer_size, hidden_dim):
        super(Decoder, self).__init__()
        tisize = 4  # temporary image size
        # assert isize % 16 == 0, "isize has to be a multiple of 16"
        max_layer_size = layer_size * (2 ** 2)
        main = nn.Sequential()
        main.add_module('initial_{0}-{1}_linear'.format(hidden_dim, max_layer_size),
                        nn.Linear(hidden_dim, max_layer_size, bias=True))
        main.add_module('initial_{0}_batchnorm'.format(max_layer_size), nn.BatchNorm1d(max_layer_size))
        main.add_module('initial_{0}_relu'.format(max_layer_size), nn.ReLU(True))

        for _ in range(2):
            max_layer_size = max_layer_size // 2
            main.add_module('pyramid_{0}_{1}_linear'.format(max_layer_size * 2, max_layer_size),
                            nn.Linear(max_layer_size * 2, max_layer_size, bias=True))
            main.add_module('pyramid_{0}_batchnorm'.format(max_layer_size),
                            nn.BatchNorm1d(max_layer_size))
            main.add_module('pyramid_{0}_relu'.format(max_layer_size),
                            nn.ReLU(True))

        main.add_module('final_{0}-{1}_linear'.format(max_layer_size, input_size),
                        nn.Linear(max_layer_size, input_size, bias=True))
        main.add_module('final_{0}_tanh'.format(input_size),
                        nn.Tanh())

        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output


def grad_norm(m, norm_type=2):
    total_norm = 0.0
    for p in m.parameters():
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)

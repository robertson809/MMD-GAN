#!/usr/bin/env python
# encoding: utf-8

import torch.nn as nn


# input: batch_size * nc * 64 * 64
# output: batch_size * k * 1 * 1
class Encoder(nn.Module):
    def __init__(self, isize, input_size, latent_dim=100, layer_size=64): #k is latent dimension, 100 is replaced in actual method call
        super(Encoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        # input is nc x isize x isize
        main = nn.Sequential()
        main.add_module('initial_layer_{0}-{1}'.format(input_size, layer_size),
                        nn.Linear(input_size, layer_size, bias=False))
        main.add_module('initial_relu_{0}'.format(layer_size),
                        nn.LeakyReLU(0.2, inplace=True)) #0.2 is starting negative slope
        csize, cndf = isize / 2, layer_size
        print('isize', isize)
        print('layer_size', layer_size)
        print('csize:', csize)
        print('cndf:', cndf)

        while csize > 4: #these are the inbetween layers, go down to the latent dimension?
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid_{0}-{1}_linear'.format(in_feat, out_feat),
                            nn.Linear(in_feat, out_feat, bias=False))
            main.add_module('pyramid_{0}_batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid_{0}_relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        main.add_module('final_{0}-{1}_conv'.format(cndf, 1),
                        nn.Linear(cndf, latent_dim, bias=False)) #final outputs to the latent dimension
        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output


# input: batch_size * k * 1 * 1 because k is the latent dimension
# output: batch_size * nc * image_size * image_size
class Decoder(nn.Module):
    def __init__(self, isize, nc, latent_dim=100, ngf=64):
        super(Decoder, self).__init__()
        #assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4 #temporary image size
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        main.add_module('initial_{0}-{1}_linear'.format(latent_dim, cngf), nn.Linear(latent_dim, cngf, bias=False))
        main.add_module('initial_{0}_batchnorm'.format(cngf), nn.BatchNorm2d(cngf))
        main.add_module('initial_{0}_relu'.format(cngf), nn.ReLU(True))

        csize = 4
        while csize < isize // 2: #built back up from latent dimension
            main.add_module('pyramid_{0}_{1}_convt'.format(cngf, cngf // 2),
                            nn.Linear(cngf, cngf, bias=False))
            main.add_module('pyramid_{0}_batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid_{0}_relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        main.add_module('final_{0}-{1}_convt'.format(cngf, nc), nn.Linear(cngf, nc, bias=False))
        main.add_module('final_{0}_tanh'.format(nc),
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

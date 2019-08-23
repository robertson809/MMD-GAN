#!/usr/bin/env python
# encoding: utf-8


import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as utils
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import os
import timeit
import click
import time as tm

import util
import numpy as np

import base_module
from mmd import mix_rbf_mmd2

# import sys
# sys.path.append('pydevd-pycharm.egg')
# import pydevd_pycharm
# pydevd_pycharm.settrace('127.0.0.1', port=8200, stdoutToServer=True, stderrToServer=True)

# NetG is a decoder
# input: batch_size * nz * 1 * 1
# output: batch_size * nc * image_size * image_size
#netG = NetG(G_decoder)
class NetG(nn.Module):
    def __init__(self, decoder):
        super(NetG, self).__init__()
        self.decoder = decoder #nn.sequential operator

    def forward(self, input):
        return self.decoder(input)


# NetD is an encoder + decoder
# input: batch_size * nc * image_size * image_size
# f_enc_X: batch_size * k * 1 * 1
# f_dec_X: batch_size * nc * image_size * image_size
class NetD(nn.Module):
    def __init__(self, encoder, decoder):
        super(NetD, self).__init__()
        self.encoder = encoder
        self.decoder = decoder #nn.sequential operator

    def forward(self, input):
        f_enc_X= self.encoder(input)
        f_dec_X = self.decoder(f_enc_X)
        # print('BM 57: type of f_enc_X is ', type(f_enc_X))
        # print('type of f_dec_X is ', type(f_dec_X))

        f_enc_X = f_enc_X.view(input.size(0), -1)
        f_dec_X = f_dec_X.view(input.size(0), -1)
        return f_enc_X, f_dec_X

class ONE_SIDED(nn.Module):
    def __init__(self):
        super(ONE_SIDED, self).__init__()

        main = nn.ReLU()
        self.main = main

    def forward(self, input):
        output = self.main(-input)
        output = -output.mean()
        return output



# Get argument
parser = argparse.ArgumentParser()
parser = util.get_args(parser)
args = parser.parse_args()


if args.experiment is None:
    args.experiment = 'samples'
os.system('mkdir {0}'.format(args.experiment))

if torch.cuda.is_available():
    args.cuda = True
    torch.cuda.set_device(args.gpu_device)
    print("Using GPU device", torch.cuda.current_device())
else:
    raise EnvironmentError("GPU device not available!")

args.manual_seed = 1126
np.random.seed(seed=args.manual_seed)
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed(args.manual_seed)
cudnn.benchmark = True

# Get data
data_michael = np.load(args.datapath).astype(np.float32)
my_tensor = torch.stack([torch.Tensor(row) for row in data_michael])
my_dataset = utils.TensorDataset(my_tensor)
my_dataloader = utils.DataLoader(my_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=int(args.workers))

# construct encoder/decoder modules
D_encoder = base_module.Encoder(args.input_size, args.initial_layer_size, args.hidden_dim) #returns nn.sequential method
#print('line 111 is D_encoder', D_encoder)
D_decoder = base_module.Decoder(args.input_size, args.initial_layer_size, args.hidden_dim) #returns a nn.sequential method
G_decoder = base_module.Decoder(args.input_size, args.initial_layer_size, args.hidden_dim)

netG = NetG(G_decoder)  #returns decoded input
netD = NetD(D_encoder, D_decoder)  #returns tuple of encoded x and decoded x
one_sided = ONE_SIDED()
print("netG:", netG)
print("netD:", netD)
print("oneSide:", one_sided)

netG.apply(base_module.weights_init)
netD.apply(base_module.weights_init)
one_sided.apply(base_module.weights_init)

# sigma for MMD
base = 1.0
sigma_list = [1, 2, 4, 8, 16]
sigma_list = [sigma / base for sigma in sigma_list]

# put variable into cuda device
fixed_noise = torch.cuda.FloatTensor(args.batch_size, args.hidden_dim).normal_(0, 1)
one = torch.cuda.FloatTensor([1])
mone = one * -1
if args.cuda:
    netG.cuda()
    netD.cuda()
    one_sided.cuda()
fixed_noise = Variable(fixed_noise, requires_grad=False)

# setup optimizer
optimizerG = torch.optim.RMSprop(netG.parameters(), lr=args.lr)
optimizerD = torch.optim.RMSprop(netD.parameters(), lr=args.lr)

lambda_MMD = 1.0
lambda_AE_X = 8.0
lambda_AE_Y = 8.0
lambda_rg = 16.0


time = timeit.default_timer()
gen_iterations = 0
for t in range(args.max_iter):
    data_iter = iter(my_dataloader)
    i = 0
    while (i < len(my_dataloader)):
        # ---------------------------
        #        Optimize over NetD
        # ---------------------------
        for p in netD.parameters():
            p.requires_grad = True

        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 100
            Giters = 1
        else:
            Diters = 5
            Giters = 1
        for j in range(Diters):
            if i == len(my_dataloader):
                break

            # clamp parameters of NetD encoder to a cube
            # do not clamp paramters of NetD decoder!!!
            for p in netD.encoder.parameters():
                p.data.clamp_(-0.01, 0.01)

            data = data_iter.next()
            i += 1
            netD.zero_grad()

            x_cpu = data[0] #changed
            x = Variable(x_cpu.cuda())
            batch_size = x.size(0)

            #print('MMD188',netD(x))
            f_enc_X_D, f_dec_X_D = netD(x)

            #print('line 191', batch_size,args.hidden_dim)
            noise = torch.cuda.FloatTensor(batch_size, args.hidden_dim).normal_(0, 1) #give values
                                                #from a normal distribution with mean 0 std 1
            #print('line 194:', noise.size())
            #print('len(noise)', len(noise))
            #print('type(noise)', type(noise))
            noise = torch.cuda.FloatTensor(noise)  # total freeze netG
            y = torch.cuda.FloatTensor(netG(noise).data)

            f_enc_Y_D, f_dec_Y_D = netD(y)

            # compute biased MMD2 and use ReLU to prevent negative value
            mmd2_D = mix_rbf_mmd2(f_enc_X_D, f_enc_Y_D, sigma_list)
            mmd2_D = F.relu(mmd2_D)

            # compute rank hinge loss
            #print('f_enc_X_D:', f_enc_X_D.size())
            #print('f_enc_Y_D:', f_enc_Y_D.size())
            one_side_errD = one_sided(f_enc_X_D.mean(0) - f_enc_Y_D.mean(0))

            # compute L2-loss of AE
            L2_AE_X_D = util.match(x.view(batch_size, -1), f_dec_X_D, 'L2')
            L2_AE_Y_D = util.match(y.view(batch_size, -1), f_dec_Y_D, 'L2')

            errD = torch.sqrt(mmd2_D) + lambda_rg * one_side_errD - lambda_AE_X * L2_AE_X_D - lambda_AE_Y * L2_AE_Y_D
            errD.backward(mone)
            optimizerD.step()

        # ---------------------------
        #        Optimize over NetG
        # ---------------------------
        for p in netD.parameters():
            p.requires_grad = False

        for j in range(Giters):
            if i == len(my_dataloader):
                break

            data = data_iter.next()
            i += 1
            netG.zero_grad()

            x = data[0] #changed
            x = Variable(x_cpu.cuda())
            batch_size = x.size(0)

            f_enc_X, f_dec_X = netD(x)

            noise = torch.cuda.FloatTensor(batch_size, args.hidden_dim).normal_(0, 1)
            noise = torch.cuda.FloatTensor(noise)   #mabye need to do something with volitile call here
            y = netG(noise)

            f_enc_Y, f_dec_Y = netD(y)

            # compute biased MMD2 and use ReLU to prevent negative value
            mmd2_G = mix_rbf_mmd2(f_enc_X, f_enc_Y, sigma_list)
            mmd2_G = F.relu(mmd2_G)

            # compute rank hinge loss
            one_side_errG = one_sided(f_enc_X.mean(0) - f_enc_Y.mean(0))

            errG = torch.sqrt(mmd2_G) + lambda_rg * one_side_errG
            errG.backward(one)
            optimizerG.step()

            gen_iterations += 1

        run_time = (timeit.default_timer() - time) / 60.0
        print('[%3d/%3d][%3d/%3d] [%5d] (%.2f m) MMD2_D %.6f hinge %.6f L2_AE_X %.6f L2_AE_Y %.6f loss_D %.6f Loss_G %.6f f_X %.6f f_Y %.6f |gD| %.4f |gG| %.4f'
              % (t, args.max_iter, i, len(my_dataloader), gen_iterations, run_time,
                 mmd2_D.data.item(), one_side_errD.item(),
                 L2_AE_X_D.item(), L2_AE_Y_D.item(),
                 errD.item(), errG.item(),
                 f_enc_X_D.mean().item(), f_enc_Y_D.mean().item(),
                 base_module.grad_norm(netD), base_module.grad_norm(netG)))

        if gen_iterations % 500 == 0:  #what is this doing? it's saving intermediate images
            y_fixed = netG(fixed_noise)  #keep generating from the same noise I guess
            #y_fixed.data = y_fixed.data.mul(0.5).add(0.5) #no thank you
            #print('full tensor.data', f_dec_X_D.data)
            decoded_sample = f_dec_X_D.data
            # print('size:', f_dec_X_D.size())
            # print('type:', type(f_dec_X))
            torch.save("{0}/dedcoded_sample_at_{1}.txt".format(args.experiment, gen_iterations),
                                                                        './samples/decoded_examples.txt')
            torch.save(f_dec_X_D.data, './samples/decoded_examples.txt')
            torch.save("{0}/dedcoded_sample_at_{1}.txt".format(args.experiment, gen_iterations),
                                                                                './samples/generated_examples.txt')
            torch.save(y_fixed,'./samples/generated_examples.txt')
            #view is reshape, figure out how I need to transform this stuff to save it
            #f_dec_X_D.data = f_dec_X_D.data.mul(0.5).add(0.5) don't need to do that
            #vutils.save_image(y_fixed.data, '{0}/generated_samples_{1}.txt'.format(args.experiment, gen_iterations))
            #vutils.save_image(f_dec_X_D.data, '{0}/decoded_samples_{1}.txt'.format(args.experiment, gen_iterations))

    if t % 50 == 0:
        torch.save(netG.state_dict(), '{0}/netG_iter_{1}.pth'.format(args.experiment, t))
        torch.save(netD.state_dict(), '{0}/netD_iter_{1}.pth'.format(args.experiment, t))
import os
import glob
from path import Path
import random
import math
from math import ceil, floor
from cbam import *
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch
from torch.optim.optimizer import Optimizer, required
from torch.autograd import Variable
from torch import Tensor

from functools import partial
import pdb
from torch.optim import lr_scheduler
from blocks import *
from Config import Config

###############################################################################################
## Define modules
# 1. Utils - Model save & Model load & Scheduler & Normalization & Activation function for Octave convolution
# 2. Attn_Conv for Encoder
# 3. AdaOctConv for Decoder
###############################################################################################
def model_save(save_type, ckpt_dir, model, optim_E, optim_G, optim_D, optim_PD, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    if save_type == 'best':
        torch.save({'netE': model.netE.state_dict(),
                    'netG': model.netG.state_dict(),
                    'netD': model.netD.state_dict(),
                    'netPD': model.netPD.state_dict(),
                    'optim_E': optim_E.state_dict(),
                    'optim_G': optim_G.state_dict(),
                    'optim_D': optim_D.state_dict(),
                    'optim_PD': optim_PD.state_dict()},
                   '%s/model_best_epoch%d.pth' % (ckpt_dir, epoch+1))
    else:
        torch.save({'netE': model.netE.state_dict(),
                    'netG': model.netG.state_dict(),
                    'netD': model.netD.state_dict(),
                    'netPD': model.netPD.state_dict(),
                    'optim_E': optim_E.state_dict(),
                    'optim_G': optim_G.state_dict(),
                    'optim_D': optim_D.state_dict(),
                    'optim_PD': optim_PD.state_dict()},
                   '%s/model_epoch%d.pth' % (ckpt_dir, epoch+1))

## 네트워크 불러오기
def model_load(save_type, ckpt_dir, model, optim_E, optim_G, optim_D, optim_PD):
    if not os.path.exists(ckpt_dir):
        epoch = -1
        return model, optim_E, optim_G, optim_D, optim_PD, epoch
    
    ckpt_path = Path(ckpt_dir)
    if save_type == 'best':
        ckpt_lst = ckpt_path.glob('model_best_epoch*')
        ckpt_lst.sort(key=lambda x: int(x.split('epoch')[1].split('.')[0]))
    else:
        ckpt_lst = ckpt_path.glob('model_epoch*')
        ckpt_lst.sort(key=lambda x: int(x.split('epoch')[1].split('.')[0]))
    model_ckpt = ckpt_lst[-1]

    dict_model = torch.load(model_ckpt)

    model.netE.load_state_dict(dict_model['netE'])
    model.netG.load_state_dict(dict_model['netG'])
    model.netD.load_state_dict(dict_model['netD'])
    model.netPD.load_state_dict(dict_model['netPD'])
    optim_E.load_state_dict(dict_model['optim_E'])
    optim_G.load_state_dict(dict_model['optim_G'])
    optim_D.load_state_dict(dict_model['optim_D'])
    optim_PD.load_state_dict(dict_model['optim_PD'])
    epoch = int(model_ckpt.split('epoch')[1].split('.pth')[0]) - 1

    return model, optim_E, optim_G, optim_D, optim_PD, epoch

def get_scheduler(optimizer, config):
    if config.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + config.n_epoch - config.n_iter) / float(config.n_iter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif config.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_iters, gamma=0.1)
    elif config.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif config.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_iter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', config.lr_policy)
    return scheduler

def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)
    
class Oct_Conv_aftup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pad_type, alpha_in, alpha_out):
        super(Oct_Conv_aftup, self).__init__()
        lf_in = int(in_channels*alpha_in)
        lf_out = int(out_channels*alpha_out)
        hf_in = in_channels - lf_in
        hf_out = out_channels - lf_out

        self.conv_h = nn.Conv2d(in_channels=hf_in, out_channels=hf_out, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type)
        self.conv_l = nn.Conv2d(in_channels=lf_in, out_channels=lf_out, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type)

    def forward(self, x):
        hf, lf = x
        hf = self.conv_h(hf)
        lf = self.conv_l(lf)
        return hf, lf

class Conv_norm(nn.Module):
    def __init__(self, planes, alpha_in=0.25, alpha_out=0.25, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, norm='in'):
        super(Conv_norm, self).__init__()
        #hf_ch = int(num_features * (1 - alpha_in))
        #lf_ch = num_features - hf_ch
        ch = planes 
        ch = planes 
        if norm=='bn':
            self.bn = nn.BatchNorm2d(ch)
        else:
            self.bn = nn.InstanceNorm2d(ch)

    def forward(self, x, alpha_in=0.25, alpha_out=0.25):
        return self.bn(x)
    
class Oct_conv_norm(nn.Module):
    def __init__(self, planes, alpha_in=0.5, alpha_out=0.5, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, norm='in'):
        super(Oct_conv_norm, self).__init__()
        
        hf_ch = int(planes * (1 - alpha_in))
        lf_ch = planes - hf_ch
        # hf_ch = planes 
        # lf_ch = planes 
        
        if norm=='in':
            self.bnh = nn.InstanceNorm2d(hf_ch)
            self.bnl = nn.InstanceNorm2d(lf_ch)
        elif norm=='adain':
            self.bnh = AdaptiveInstanceNorm2d(hf_ch)
            self.bnl = AdaptiveInstanceNorm2d(lf_ch)
        else:
            self.bnh = nn.BatchNorm2d(hf_ch)
            self.bnl = nn.BatchNorm2d(lf_ch)
            
    def forward(self, x):
        hf, lf = x
        return self.bnh(hf), self.bnl(lf)

class Oct_conv_reLU(nn.ReLU):
    def forward(self, x):
        hf, lf = x
        hf = super(Oct_conv_reLU, self).forward(hf)
        lf = super(Oct_conv_reLU, self).forward(lf)
        return hf, lf
    
class Oct_conv_lreLU(nn.LeakyReLU):
    def forward(self, x):
        hf, lf = x
        hf = super(Oct_conv_lreLU, self).forward(hf)
        lf = super(Oct_conv_lreLU, self).forward(lf)
        return hf, lf

class Oct_conv_up(nn.Upsample):
    def forward(self, x):
        hf, lf = x
        hf = super(Oct_conv_up, self).forward(hf)
        lf = super(Oct_conv_up, self).forward(lf)
        return hf, lf

class Oct_conv_avg(nn.AvgPool2d):
    def forward(self, x):
        hf, lf = x
        hf = super(Oct_conv_avg, self).forward(hf)
        lf = super(Oct_conv_avg, self).forward(lf)
        return hf, lf

class Oct_trns_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1, padding_mode='zeros', alpha_in=0.5, alpha_out=0.5):
        super(Oct_trns_conv, self).__init__()

        hf_in = int(in_channels * (1 - alpha_in))
        lf_in = in_channels - hf_in

        hf_out = int(out_channels * (1 - alpha_out))
        lf_out = out_channels - hf_out

        self.up_high = nn.ConvTranspose2d(in_channels=hf_in, out_channels=hf_out, kernel_size=kernel_size, stride=stride
, padding=padding, dilation=dilation, output_padding=output_padding, padding_mode='zeros')
        self.up_low = nn.ConvTranspose2d(in_channels=lf_in, out_channels=lf_out, kernel_size=kernel_size, stride=stride,
 padding=padding, dilation=dilation, output_padding=output_padding, padding_mode='zeros' )

    def forward(self, x):
        hf, lf = x
        return self.up_high(hf), self.up_low(lf)

class Oct_conv_down(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(Oct_conv_down, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        self.avg_pool = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        hf, lf = x
        hf = self.max_pool(hf)
        lf = self.avg_pool(lf)
        return hf, lf

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

###############################################################################################
## Encoder
# Attn_Conv: Attention + Octave Convolution
###############################################################################################
class OctConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, 
                pad_type='reflect', alpha_in=0.5, alpha_out=0.5, type='normal', freq_ratio = None):
        super(OctConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.type = type
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.freq_ratio = freq_ratio

        hf_ch_in = int(in_channels * (1 - self.alpha_in))
        hf_ch_out = int(out_channels * (1 -self. alpha_out))
        lf_ch_in = in_channels - hf_ch_in
        lf_ch_out = out_channels - hf_ch_out

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.is_dw = groups == in_channels

        if type == 'first':
            self.convh = nn.Conv2d(in_channels, hf_ch_out, kernel_size=kernel_size,
                                    stride=stride, padding=padding, padding_mode=pad_type, bias = False)
            self.convl = nn.Conv2d(in_channels, lf_ch_out,
                                   kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type, bias=False)
        elif type == 'last':
            self.convh = nn.Conv2d(hf_ch_in, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type, bias=False)
            self.convl = nn.Conv2d(lf_ch_in, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type, bias=False)
            #self.upsample = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1, padding_mode='zeros')
        else:
            self.L2L = nn.Conv2d(
                lf_ch_in, lf_ch_out,
                kernel_size=kernel_size, stride=stride, padding=padding, groups=math.ceil(alpha_in * groups), padding_mode=pad_type, bias=False
            )
            if self.is_dw:
                self.L2H = None
                self.H2L = None
            else:
                self.L2H = nn.Conv2d(
                    lf_ch_in, hf_ch_out,
                    kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, padding_mode=pad_type, bias=False
                )
                self.H2L = nn.Conv2d(
                    hf_ch_in, lf_ch_out,
                    kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, padding_mode=pad_type, bias=False
                )
            self.H2H = nn.Conv2d(
                hf_ch_in, hf_ch_out,
                kernel_size=kernel_size, stride=stride, padding=padding, groups=math.ceil(groups - alpha_in * groups), padding_mode=pad_type, bias=False
            )

    def forward(self, x):
        if self.type == 'first':
            hf = self.convh(x)
            lf = self.avg_pool(x)
            lf = self.convl(lf)
            return hf, lf
        elif self.type == 'last':
            hf, lf = x
            out_h = self.convh(hf)
            out_l = self.convl(self.upsample(lf))
            output = out_h + out_l
            return output, out_h, out_l
        else:
            hf, lf = x
            if self.is_dw:
                hf, lf = self.H2H(hf), self.L2L(lf)
            else:
                hf, lf = self.H2H(hf) + self.L2H(self.upsample(lf)), self.L2L(lf) + self.H2L(self.avg_pool(hf))
            return hf, lf


###############################################################################################
## Decoder
# AdaOctConv: Adaptive Convolution + Octave Convolution
###############################################################################################
class AdaOctConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups, style_channels, kernel_size,
                 stride, padding, alpha_in, alpha_out, type='normal'):
        super(AdaOctConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.type = type
        
        h_in = int(in_channels * (1 - self.alpha_in))
        h_out = int(out_channels * (1 -self. alpha_out))
        l_in = in_channels - h_in
        l_out = out_channels - h_out
        
        self.out_channels = out_channels if type == 'last' else in_channels

        self.kernelPredictor_h = KernelPredictor(in_channels=h_in,
                                               out_channels=h_in,
                                               n_groups=n_groups,
                                               style_channels=style_channels,
                                               kernel_size=kernel_size)
        self.kernelPredictor_l = KernelPredictor(in_channels=l_in,
                                               out_channels=l_in,
                                               n_groups=n_groups,
                                               style_channels=style_channels,
                                               kernel_size=kernel_size)
        
        self.AdaConv_h = AdaConv2d(in_channels=h_in, out_channels=h_in, n_groups=n_groups)
        self.AdaConv_l = AdaConv2d(in_channels=l_in, out_channels=l_in, n_groups=n_groups)
        
        self.AttnOctConv = Attn_OctConv(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, 
                                        alpha_in=alpha_in, alpha_out=alpha_out, type=type)
        
        self.norm = Oct_conv_norm(planes=in_channels, norm='in', alpha_in=alpha_in)
        self.norm2 = Oct_conv_norm(planes=out_channels, norm='in', alpha_in=alpha_in)
        self.relu = Oct_conv_lreLU(negative_slope=0.2, inplace=False)

    def forward(self, content, style):
        c_hf, c_lf = content
        s_hf, s_lf = style
        h_w_spatial, h_w_pointwise, h_bias = self.kernelPredictor_h(s_hf)
        l_w_spatial, l_w_pointwise, l_bias = self.kernelPredictor_l(s_lf)
        output_h = self.AdaConv_h(c_hf, h_w_spatial, h_w_pointwise, h_bias)
        output_l = self.AdaConv_l(c_lf, l_w_spatial, l_w_pointwise, l_bias)
        output = output_h, output_l
        #output = c_hf, output_l
        
        output = self.norm(output)
        output = self.relu(output)

        output = self.AttnOctConv(output)
        if self.type != 'last':
            output = self.norm2(output)
            output = self.relu(output)

        return output

class KernelPredictor(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups, style_channels, kernel_size):
        super(KernelPredictor, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_groups = n_groups
        self.w_channels = style_channels
        self.kernel_size = kernel_size

        padding = (kernel_size - 1) / 2
        self.spatial = nn.Conv2d(style_channels,
                                 in_channels * out_channels // n_groups,
                                 kernel_size=kernel_size,
                                 padding=(ceil(padding), ceil(padding)),
                                 padding_mode='reflect')
        self.pointwise = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(style_channels,
                      out_channels * out_channels // n_groups,
                      kernel_size=1)
        )
        self.bias = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(style_channels,
                      out_channels,
                      kernel_size=1)
        )

    def forward(self, w):
        w_spatial = self.spatial(w)
        w_spatial = w_spatial.reshape(len(w),
                                      self.out_channels,
                                      self.in_channels // self.n_groups,
                                      self.kernel_size, self.kernel_size)

        w_pointwise = self.pointwise(w)
        w_pointwise = w_pointwise.reshape(len(w),
                                          self.out_channels,
                                          self.out_channels // self.n_groups,
                                          1, 1)
        bias = self.bias(w)
        bias = bias.reshape(len(w),
                            self.out_channels)

        return w_spatial, w_pointwise, bias

class AdaConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, n_groups=None):
        super(AdaConv2d, self).__init__()
        self.n_groups = in_channels if n_groups is None else n_groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        padding = (kernel_size - 1) / 2
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(kernel_size, kernel_size),
                              padding=(ceil(padding), floor(padding)),
                              padding_mode='reflect')

    def forward(self, x, w_spatial, w_pointwise, bias):
        assert len(x) == len(w_spatial) == len(w_pointwise) == len(bias)
        x = F.instance_norm(x)

        ys = []
        for i in range(len(x)):
            y = self.forward_single(x[i:i+1], w_spatial[i], w_pointwise[i], bias[i])
            ys.append(y)
        ys = torch.cat(ys, dim=0)

        ys = self.conv(ys)
        return ys

    def forward_single(self, x, w_spatial, w_pointwise, bias):
        # Only square kernels
        assert w_spatial.size(-1) == w_spatial.size(-2)
        padding = (w_spatial.size(-1) - 1) / 2
        pad = (ceil(padding), floor(padding), ceil(padding), floor(padding))

        x = F.pad(x, pad=pad, mode='reflect')
        x = F.conv2d(x, w_spatial, groups=self.n_groups)
        x = F.conv2d(x, w_pointwise, groups=self.n_groups, bias=bias)
        return x

#############################################################
# Discriminator
#############################################################
def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


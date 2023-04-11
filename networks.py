import torch
from torch import nn
import torch.nn.functional as F
from blocks import *
from blocks_swap import *
import os
import math
from collections import OrderedDict
import numpy as np
import math

#####################################################################################
## Define Networks : Content & Style Encoder, Decoder(=Generator), Discriminator 
#####################################################################################
def init_net(net, init_type='normal', gpu_ids=[], config=None):
    if config.distributed:
        if config.gpu is not None:
                config.device = torch.device('cuda:{}'.format(config.gpu))
                torch.cuda.set_device(config.gpu)
                net.cuda(config.gpu)
                
                net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[config.gpu], broadcast_buffers=False, find_unused_parameters=True) 
                net_without_ddp = net.module
        else:
            config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            net.cuda()

            net = torch.nn.parallel.DistributedDataParallel(net) 
            net_without_ddp = net.module
    else:
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = torch.nn.DataParallel(net).to(config.device)  
        
    return net

def define_network(net_type, alpha_in=0.5, alpha_out=0.5, init_type='normal', gpu_ids=[], config = None):
    net = None

    if net_type == 'Encoder':
        net = Encoder(in_dim=config.input_nc, out_dim=config.feat_nc, alpha_in=alpha_in, alpha_out=alpha_out)
    elif net_type == 'MLP':
        net = MLP(input_dim=4, output_dim=config.feat_nc, dim=256, n_blk=3, norm='none', activ='relu')
    elif net_type == 'Generator':
        net = Decoder(config, in_dim=config.feat_nc, out_dim=config.output_nc, style_kernel=3, alpha_in=alpha_in, alpha_out=alpha_out)
    elif net_type == 'Discriminator':
        net = Discriminator(config=config)
    elif net_type == 'Patch_Dis':
        net = Patch_Discriminator(config)

    return init_net(net, init_type, gpu_ids, config = config)


class Encoder(nn.Module):    
    def __init__(self, in_dim, out_dim, alpha_in=0.5, alpha_out=0.5, norm='in', activ='relu', pad_type='zeros'):    
        super(Encoder, self).__init__()    
        
        # before: 1 x 128 x 128 -> 64 x 128 x 128
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=64, kernel_size=7, stride=1, padding=3)        
        
        # 1st: 128 x 64 x 64
        self.OctConv1_1 = OctConv(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, groups=64, alpha_in=alpha_in, alpha_out=alpha_out, type="first")
        self.OctConv1_2 = OctConv(in_channels=64, out_channels=128, kernel_size=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")

        # 2nd AOIR: 256 x 64 x 64     
        self.OctConv2_1 = OctConv(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, groups=128, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv2_2 = OctConv(in_channels=128, out_channels=256, kernel_size=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")

        # 3rd: 512 x 16 x 16
        #self.attOctConv3 = Attn_OctConv(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")    
        #self.norm3 = Oct_conv_norm(planes = 512, norm = norm)

        self.relu = Oct_conv_lreLU(negative_slope=0.2, inplace=False)

    def forward(self, x):     
        out = self.conv(x)   
        
        out = self.OctConv1_1(out) 
        out = self.relu(out)
        out = self.OctConv1_2(out)
        out = self.relu(out)
        
        out = self.OctConv2_1(out)   
        out = self.relu(out)
        out = self.OctConv2_2(out)
        out = self.relu(out)

        return out


class Decoder(nn.Module):
    def __init__(self, config, in_dim=256, out_dim=3, style_kernel=3, alpha_in=0.5, alpha_out=0.5, pad_type='reflect'):
        super(Decoder, self).__init__()

        group_div = [1, 2, 4, 8]
        norm='in'
        self.up_oct = Oct_conv_up(scale_factor=2)

        # 1st : 256 x 32 x 32
        self.OctConv1_1 = OctConv(in_channels=256, out_channels=256, kernel_size=style_kernel, stride=1, padding=1, groups=256, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv1_2 = OctConv(in_channels=256, out_channels=128, kernel_size=1, stride=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.oct_conv_aftup_1 = Oct_Conv_aftup(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, pad_type=pad_type, alpha_in=alpha_in, alpha_out=alpha_out)

        # 2nd : 128 x 64 x 64
        self.OctConv2_1 = OctConv(in_channels=128, out_channels=128, kernel_size=style_kernel, stride=1, padding=1, groups=128, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv2_2 = OctConv(in_channels=128, out_channels=64, kernel_size=1, stride=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.oct_conv_aftup_2 = Oct_Conv_aftup(64, 64, 3, 1, 1, pad_type, alpha_in, alpha_out)

        # 3rd : 64 x 128 x 128
        self.OctConv3_1 = OctConv(in_channels=64, out_channels=64, kernel_size=style_kernel, stride=1, padding=1, groups=64, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv3_2 = OctConv(in_channels=64, out_channels=32, kernel_size=1, stride=1, alpha_in=alpha_in, alpha_out=alpha_out, type="last")

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=out_dim, kernel_size=1)

    def forward(self, content):
        # 1st OctConv layer
        out = self.OctConv1_1(content)
        out = self.OctConv1_2(out)
        out = self.up_oct(out)
        out = self.oct_conv_aftup_1(out)
        
        # 2nd OctConv layer
        out = self.OctConv2_1(out)
        out = self.OctConv2_2(out)
        out = self.up_oct(out)
        out = self.oct_conv_aftup_2(out)

        # 3rd OctConv layer
        out = self.OctConv3_1(out)
        out, out_high, out_low = self.OctConv3_2(out)

        out = self.conv4(out)
        out_high = self.conv4(out_high)
        out_low = self.conv4(out_low)

        return out, out_high, out_low


class Discriminator(nn.Module):
    def __init__(self, config, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: min(512, int(512 * channel_multiplier)),
            32: min(512, int(512 * channel_multiplier)),
            64: int(256 * channel_multiplier),
            128: int(128 * channel_multiplier),
            256: int(64 * channel_multiplier),
            512: int(32 * channel_multiplier),
            1024: int(16 * channel_multiplier),
        }

        size = config.load_size
        original_size = size

        size = 2 ** int(round(math.log(size, 2)))

        convs = [('0', ConvLayer(config.output_nc, channels[size], 1))]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            layer_name = str(9 - i) if i <= 8 else "%dx%d" % (2 ** i, 2 ** i)
            convs.append((layer_name, ResBlock(in_channel, out_channel, blur_kernel)))

            in_channel = out_channel

        self.convs = nn.Sequential(OrderedDict(convs))

        self.final_conv = ConvLayer(in_channel, channels[4], 3)

        side_length = int(4 * original_size / size)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * (side_length ** 2), channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out

    def get_features(self, input):
        return self.final_conv(self.convs(input))

class Patch_Discriminator(nn.Module):
    def __init__(self, config):
        super(Patch_Discriminator, self).__init__()
        
        self.config = config
        channel_multiplier = self.config.netPatchD_scale_capacity
        size = self.config.patch_size
        
        channels = {
            4: min(self.config.netPatchD_max_nc, int(256 * channel_multiplier)),
            8: min(self.config.netPatchD_max_nc, int(128 * channel_multiplier)),
            16: min(self.config.netPatchD_max_nc, int(64 * channel_multiplier)),
            32: int(32 * channel_multiplier),
            64: int(16 * channel_multiplier),
            128: int(8 * channel_multiplier),
            256: int(4 * channel_multiplier),
        }
        
        log_size = int(math.ceil(math.log(size, 2)))
        in_channel = channels[2 ** log_size]
        
        blur_kernel = [1, 3, 3, 1] if self.config.use_antialias else [1]
        
        # Feature Extraction
        convs = [('0', ConvLayer(config.input_nc, in_channel, 3))]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i-1)]
            layer_name = str(7-i) if i <=6 else "%dx%d" % (2 ** i, 2 ** i)
            convs.append((layer_name, ResBlock(in_channel, out_channel, blur_kernel)))
            in_channel = out_channel
        convs.append(('5', ResBlock(in_channel, self.config.netPatchD_max_nc * 2, downsample=False)))
        convs.append(('6', ConvLayer(self.config.netPatchD_max_nc * 2, self.config.netPatchD_max_nc, 3, activate=False, pad=0)))
        convs.append(("Act", FusedLeakyReLU(self.config.netPatchD_max_nc)))
        conv_layers = OrderedDict(convs)
        self.convs = nn.Sequential(conv_layers)

        # Discriminator
        out_dim = 1
        pairlinear1 = EqualLinear(channels[4] * 2 * 2 * 2, 2048, activation='fused_lrelu')
        pairlinear2 = EqualLinear(2048, 2048, activation='fused_lrelu')
        pairlinear3 = EqualLinear(2048, 1024, activation='fused_lrelu')
        pairlinear4 = EqualLinear(1024, out_dim)
        self.pairlinear = nn.Sequential(pairlinear1, pairlinear2, pairlinear3, pairlinear4)
        
    def get_random_crops(self, x, num_crops=1, crop_window=None):
        target_size = self.config.patch_size
        scale_range = (self.config.patch_min_scale, self.config.patch_max_scale)
        num_crops = self.config.patch_num_crops
        
        B = x.size(0) * num_crops
        flip = torch.round(torch.rand(B, 1, 1, 1, device=x.device)) * 2 - 1.0
        unit_grid_x = torch.linspace(-1.0, 1.0, target_size, device=x.device)[np.newaxis, np.newaxis, :, np.newaxis].repeat(B, target_size, 1, 1)
        unit_grid_y = unit_grid_x.transpose(1, 2)
        unit_grid = torch.cat([unit_grid_x * flip, unit_grid_y], dim=3)
        
        x = x.unsqueeze(1).expand(-1, num_crops, -1, -1, -1).flatten(0, 1)
        scale = torch.rand(B, 1, 1, 2, device=x.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
        offset = (torch.rand(B, 1, 1, 2, device=x.device) * 2 - 1) * (1 - scale)
        sampling_grid = unit_grid * scale + offset
        crop = F.grid_sample(x, sampling_grid, align_corners=False)
        crop = crop.view(B // num_crops, num_crops, crop.size(1), crop.size(2), crop.size(3))

        return crop
                                     
    def extract_features(self, patches, aggregate=False):
        if patches.ndim == 5:
            B, T, C, H, W = patches.size()
            flattened_patches = patches.flatten(0, 1)
        else:
            B, C, H, W = patches.size()
            T = patches.size(1)
            flattened_patches = patches

        #features = self.convs(flattened_patches)
        features_list = []
        conv_feat = flattened_patches
        for conv in self.convs:
            conv_feat = conv(conv_feat)
            features_list.append(conv_feat)

        features = features_list[-1]
        features = features.view(B, T, features.size(1), features.size(2), features.size(3))
        if aggregate:
            features = features.mean(1, keepdim=True).expand(-1, T, -1, -1, -1)
        return features.flatten(0, 1), features_list
    
    def discriminate_features(self, feature1, feature2):
        feature1 = feature1.flatten(1)
        feature2 = feature2.flatten(1)
        out = self.pairlinear(torch.cat([feature1, feature2], dim=1))
        return out
    
    
#####################################################################################
## Define Loss function
#####################################################################################
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgangp':
            self.loss = None
        elif gan_mode == 'swap':
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'swap':
            if target_is_real:
                #loss = F.softplus(-prediction).view(prediction.size(0), -1).mean(dim=1)
                loss = F.softplus(-prediction).mean()
            else:
                #loss = F.softplus(prediction).view(prediction.size(0), -1).mean(dim=1)
                loss = F.softplus(prediction).mean()
        return loss

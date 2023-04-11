import torch
from torch import nn
import torch.nn.functional as F
from blocks import *
from blocks_swap import *
import os
import math
from collections import OrderedDict
import numpy as np

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
        net = Encoder(in_dim=3, out_dim=512, alpha_in=alpha_in, alpha_out=alpha_out)
    elif net_type == 'Style':
        net = StyleEncoder(in_dim=512, out_dim=512, style_channel=512, style_kernel=3, alpha_in=alpha_in, alpha_out=alpha_out)
    elif net_type == 'Generator':
        net = Decoder(in_dim=512, out_dim=3, style_channel=512, alpha_in=alpha_in, alpha_out=alpha_out)
    elif net_type == 'Discriminator':
        net = Discriminator(in_dim=3, ndf=64)
    elif net_type == 'Patch_Dis':
        net = Patch_Discriminator(config)

    return init_net(net, init_type, gpu_ids, config = config)

class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, alpha_in=0.5, alpha_out=0.5, norm='in', activ='relu', pad_type='reflect'):
        super(Encoder, self).__init__()
        """ AOIR = (Attention + Octave convolution) + Instance normalization + ReLU activation function   """

        # before AOIR: 3 x 256 x 256 -> 32 x 256 x 256
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=32, kernel_size=1)

        # 1st AOIR: 64 x 128 x 128
        self.attOctConv1 = Attn_OctConv(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, alpha_in=alpha_in, alpha_out=alpha_out, type="first", use_cbam=True)
        self.norm1 = Oct_conv_norm(planes=64, norm = norm)
        self.relu1 = Oct_conv_lreLU(negative_slope=0.2, inplace=False)

        # 2nd AOIR: 128 x 64 x 64
        self.attOctConv2 = Attn_OctConv(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal", use_cbam=True)
        self.norm2 = Oct_conv_norm(planes = 128, norm = norm)
        self.relu2 = Oct_conv_lreLU(negative_slope=0.2, inplace=False)

        # 3rd AOIR: 256 x 32 x 32
        self.attOctConv3 = Attn_OctConv(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal", use_cbam=True)
        self.norm3 = Oct_conv_norm(planes = 256, norm = norm)
        self.relu3 = Oct_conv_lreLU(negative_slope=0.2, inplace=False)

        # 4th AOIR: 512 x 16 x 16
        self.attOctConv4 = Attn_OctConv(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal", use_cbam=True)
        self.norm4 = Oct_conv_norm(planes = 512, norm = norm)
        self.relu4 = Oct_conv_lreLU(negative_slope=0.2, inplace=False)

    def forward(self, x, freq_weight):
        # before AOIR
        out = self.conv(x)
        
        # 1st AOIR layer
        out = self.attOctConv1(out)
        out = self.norm1(out)
        out = self.relu1(out)
        out = soft_weighting(out, freq_weight)
        
        # 2nd AOIR layer
        out = self.attOctConv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = soft_weighting(out, freq_weight)
        
        # 3rd AOIR layer
        out = self.attOctConv3(out)
        out = self.norm3(out)
        out = self.relu3(out)
        out = soft_weighting(out, freq_weight)

        # 4th AOIR layer
        out = self.attOctConv4(out)
        out = self.norm4(out)
        out = self.relu4(out)
        out = soft_weighting(out, freq_weight)
        
        return out

class StyleEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, style_channel=512, style_kernel=3, alpha_in=0.5, alpha_out=0.5, norm='in', activ='relu', pad_type='reflect'):
        super(StyleEncoder, self).__init__()
        self.style_channel = style_channel
        self.style_kernel = style_kernel

        # 1st AOIR: 512 x 16 x 16 -> 1024 x 8 x 8
        self.attOctConv1 = Attn_OctConv(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, pad_type=pad_type, alpha_in=alpha_in, alpha_out=alpha_out, type="normal", use_cbam=True)
        #self.norm1 = Oct_conv_norm(planes=1024, norm = norm)
        self.relu1 = Oct_conv_lreLU(negative_slope=0.2, inplace=False)
        self.down1 = nn.AvgPool2d(2, 2)

        # 2nd AOIR: 2048 x 4 x 4
        self.attOctConv2 = Attn_OctConv(in_channels=1024, out_channels=2048, kernel_size=3, stride=1, padding=1, pad_type=pad_type, alpha_in=alpha_in, alpha_out=alpha_out, type="normal", use_cbam=True)
        #self.norm2 = Oct_conv_norm(planes = 2048, norm = norm)
        self.relu2 = Oct_conv_lreLU(negative_slope=0.2, inplace=False)
        self.down2 = nn.AvgPool2d(2,2)

        # FC layer
        self.fc_h = nn.Linear(1024 * 4 * 4, style_channel * style_kernel * style_kernel)
        self.fc_l = nn.Linear(1024 * 2 * 2, style_channel * style_kernel * style_kernel)
        
    def forward(self, x, freq_weight):
        # 1st AOIR layer
        out = self.attOctConv1(x)
        #out = self.norm1(out)
        out = self.relu1(out)
        out = self.down1(out[0]), self.down1(out[1])
        out = soft_weighting(out, freq_weight)
        
        # 2nd AOIR layer
        out = self.attOctConv2(out)
        #out = self.norm2(out)
        out = self.relu2(out)
        out = self.down2(out[0]), self.down2(out[1])
        out_h, out_l = soft_weighting(out, freq_weight)

        # Averaging
        #out_h = out_h.mean(dim=(2,3))
        #out_l = out_l.mean(dim=(2,3))
        
        # FC - High
        out_h = out_h.reshape(len(out_h), -1)
        out_h = self.fc_h(out_h)
        out_h = out_h.reshape(len(x[0]), self.style_channel, self.style_kernel, self.style_kernel)
        # FC - Low
        out_l = out_l.reshape(len(out_l), -1)
        out_l = self.fc_l(out_l)
        out_l = out_l.reshape(len(x[1]), self.style_channel, self.style_kernel, self.style_kernel)
        
        out = out_h, out_l
        out = soft_weighting(out, freq_weight)

        return out

class Decoder(nn.Module):
    def __init__(self, in_dim=256, out_dim=1, style_channel=512, alpha_in=0.5, alpha_out=0.5, pad_type='zero'):
        super(Decoder, self).__init__()

        group_div = [1, 2, 4, 8, 16]
        norm='in'
        self.relu = Oct_conv_lreLU(negative_slope=0.2, inplace=False)

        # For Contents only: 256 x 16 x 16 (not change)
        self.ContentConv = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        # 256(=512/2) x 16 x 16 -> 4(=8/2) x 16 x 16
        self.Contentlayer = nn.Conv2d(in_channels=256, out_channels=4, kernel_size=1)

        # 0th : 8 x 16 x 16 -> 512 x 16 x 16
        self.AdaOctConv0 = AdaOctConv(in_channels=8, out_channels=512, n_groups=group_div[0], style_channels=style_channel, kernel_size=
3, stride=1, padding=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal", use_cbam=False)
        self.up0 = Oct_conv_up(scale_factor=2, mode='nearest')

        # 1st : 512 x 16 x 16 -> 256 x 32 x 32
        self.AdaOctConv1 = AdaOctConv(in_channels=512, out_channels=512, n_groups=group_div[1], style_channels=style_channel, kernel_size=3, stride=1, padding=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal", use_cbam=False)
        #self.norm4 = Oct_conv_norm(planes=256, norm=norm)
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.up1 = Oct_conv_up(scale_factor=2, mode='nearest')

        # 2nd : 128 x 64 x 64
        self.AdaOctConv2 = AdaOctConv(in_channels=256, out_channels=256, n_groups=group_div[2], style_channels=style_channel, kernel_size=3, stride=1, padding=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal", use_cbam=False)
        self.conv2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv2_3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.up2 = Oct_conv_up(scale_factor=2, mode='nearest')

        # 3rd : 64 x 128 x 128
        self.AdaOctConv3 = AdaOctConv(in_channels=128, out_channels=128, n_groups=group_div[3], style_channels=style_channel, kernel_size=3, stride=1, padding=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal", use_cbam=False)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.up3 = Oct_conv_up(scale_factor=2, mode='nearest')

        # 4th : 3 x 256 x 256
        self.AdaOctConv4 = AdaOctConv(in_channels=64, out_channels=64, n_groups=group_div[4], style_channels=style_channel, kernel_size=3, stride=1, padding=1, alpha_in=alpha_in, alpha_out=alpha_out, type="last", use_cbam=False) 
        self.relu4 = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

        #self.tanh = nn.Tanh()

    def forward(self, content, style, freq_weight):
        # For content image only
        content_c = self.ContentConv(content[0]), self.ContentConv(content[1])
        content_s = self.Contentlayer(content_c[0]), self.Contentlayer(content_c[1])

        out = self.AdaOctConv0(content_s, style)
        out = soft_weighting(out, freq_weight)
        out = self.relu(out)
        out = self.up0(out)

        # 1st AdaOctConv layer
        out = self.AdaOctConv1(out, style)
        out = soft_weighting(out, freq_weight)
        out = self.conv1(out[0]), self.conv1(out[1])
        out = self.relu(out)
        out = self.up1(out)

        # 2nd AdaOctConv layer
        out = self.AdaOctConv2(out, style)
        out = soft_weighting(out, freq_weight)
        out = self.relu(out)
        out = self.conv2_1(out[0]), self.conv2_1(out[1])
        out = self.relu(out)
        out = self.conv2_2(out[0]), self.conv2_2(out[1])
        out = self.relu(out)
        out = self.conv2_3(out[0]), self.conv2_3(out[1])
        out = self.relu(out)
        out = self.up2(out)
        
        # 3rd AdaOctConv layer
        out = self.AdaOctConv3(out, style)
        out = soft_weighting(out, freq_weight)
        out = self.relu(out)
        out = self.conv3(out[0]), self.conv3(out[1])
        out = self.relu(out)
        out = self.up3(out)

        # 4th AdaOctConv layer        
        out = self.AdaOctConv4(out, style)
        out = self.relu4(out)
        out = self.conv4(out)
        
        return out
    
class Discriminator(nn.Module):
    def __init__(self, in_dim, ndf=64, norm_layer=nn.BatchNorm2d, use_bias=True):
        super(Discriminator, self).__init__()

        # input: 3 x 256 x 256
        sequence = [SpectralNorm(nn.Conv2d(in_dim, ndf, kernel_size=3, stride=1, padding=1)), nn.LeakyReLU(0.2, True),   # 64 x 128 x 128
                    
                    SpectralNorm(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias)), # 128 x 64 x 64
                    #norm_layer(ndf * 2),
                    nn.LeakyReLU(0.2, True),

                    SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias)), # 256 x 32 x 32
                    #norm_layer(ndf * 4),
                    nn.LeakyReLU(0.2, True),

                    SpectralNorm(nn.Conv2d(ndf * 4, ndf * 4, kernel_size=4, stride=2, padding=1)), # 256 x 16 x 16

                    SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)), # 512 x 8 x 8
                    #norm_layer(ndf * 8),
                    nn.LeakyReLU(0.2, True),

                    nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=1)] # 1 x 8 x 8
        
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)
    
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
        convs = [('0', ConvLayer(3, in_channel, 3))]
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

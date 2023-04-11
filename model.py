import torch
from torch import nn
from sklearn.metrics import mean_squared_error
import os
import sys
import networks
import gc
from utils.freq_loss import *
from utils.freq_pixel_loss import *
from utils.freq_fourier_loss import *

from vgg19 import vgg, VGG_loss

class OCCAY(nn.Module):
    def __init__(self, config):
        super(OCCAY, self).__init__()

        self.config = config
        self.device = torch.device('cuda:{}'.format(config.gpu) if torch.cuda.is_available() else 'cpu')

        self.lr = config.lr
        self.lambda_img = config.lambda_img
        self.lambda_pix = config.lambda_pix
        self.lambda_GAN_G = config.lambda_GAN_G
        self.lambda_GAN_D = config.lambda_GAN_D
        self.lambda_PD_G = config.lambda_PD_G
        self.lambda_PD_D = config.lambda_PD_D
        self.lambda_trs = config.lambda_trs
        
        self.alpha_in = config.alpha_in
        self.alpha_out = config.alpha_out
        
        torch.cuda.empty_cache()
        
        # Encoder & Generator
        self.netE = networks.define_network(net_type='Encoder', alpha_in=self.alpha_in, alpha_out=self.alpha_out, 
                                            init_type=self.config.init_type, gpu_ids=self.config.gpu, config = config)
        self.netG = networks.define_network(net_type='Generator', alpha_in=self.alpha_in, alpha_out=self.alpha_out, 
                                            init_type=self.config.init_type, gpu_ids=self.config.gpu, config = config)
        # Two Discriminators
        self.netD = networks.define_network(net_type='Discriminator', init_type=self.config.init_type, gpu_ids=self.config.gpu, config = config)
        self.netPD = networks.define_network(net_type='Patch_Dis', init_type=self.config.init_type, gpu_ids=self.config.gpu, config=config)
        
        # Loss
        self.criterion_L1 = nn.L1Loss()
        self.criterion_GAN = networks.GANLoss(self.config.gan_mode)
        #self.az_loss = nn.BCELoss()
        self.vgg_loss = VGG_loss(config, vgg)

        # Optimizer
        self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.99))
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.99))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.99))
        self.optimizer_PD = torch.optim.Adam(self.netPD.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.99))
        
        # Scheduler
        self.E_scheduler = networks.get_scheduler(self.optimizer_E, config)
        self.G_scheduler = networks.get_scheduler(self.optimizer_G, config)
        self.D_scheduler = networks.get_scheduler(self.optimizer_D, config)
        self.PD_scheduler = networks.get_scheduler(self.optimizer_PD, config)
       
    def swap(self, x):
        shape = x.shape
        assert shape[0] % 2 == 0
        new_shape = [shape[0] // 2, 2] + list(shape[1:])
        x = x.view(*new_shape)
        x = torch.flip(x, [1])
        return x.view(*shape)
        
    def forward(self, data):
        self.real_A = data['real_A'].to(self.device)
        self.real_B = data['real_B'].to(self.device)
       
        # 1. Generate images
        content_A = self.netE(self.real_A)
        self.trs_AtoB, self.trs_AtoB_high, self.trs_AtoB_low = self.netG(content_A)

        # 2. Frequency images
        self.gauss_kernel = get_gaussian_kernel(self.config.gauss_kernel_size).to(self.device)
        real_B_pix_freq = find_fake_freq(self.real_B, self.gauss_kernel)
        if self.config.output_nc == 3:
            self.real_B_low = real_B_pix_freq[:, :3, ...]
            self.real_B_high = real_B_pix_freq[:, 3:6, ...]
        else:
            self.real_B_low = real_B_pix_freq[:, 0, ...].unsqueeze(1)
            self.real_B_high = real_B_pix_freq[:, 1, ...].unsqueeze(1)
        """
        rec_pix_freq = find_fake_freq(self.rec_AtoA, self.gauss_kernel)
        if self.config.output_nc == 3:
            self.rec_low = rec_pix_freq[:, :3, ...]
            self.rec_high = rec_pix_freq[:, 3:6, ...]
        else:
            self.rec_low = rec_pix_freq[:, 0, ...].unsqueeze(1)
            self.rec_high = rec_pix_freq[:, 1, ...].unsqueeze(1)
        """

        # 2. Discriminate images
        ## Real image
        self.D_real_B = self.netD(self.real_B)
        ## Translation
        self.D_fake_AtoB = self.netD(self.trs_AtoB.detach())
        ## Smooth image
        self.D_smooth = self.netD(self.real_B_low.detach())
        
        ## Patch Discriminator
        real_feat = self.netPD.module.extract_features(self.netPD.module.get_random_crops(self.real_B), aggregate=self.config.patch_use_aggregation)[0].detach()
        target_feat, self.target_feat_list = self.netPD.module.extract_features(self.netPD.module.get_random_crops(self.real_B))
        mix_feat, self.mix_feat_list = self.netPD.module.extract_features(self.netPD.module.get_random_crops(self.trs_AtoB))

        self.PD_real = self.netPD.module.discriminate_features(real_feat, target_feat.detach())
        self.PD_fake = self.netPD.module.discriminate_features(real_feat, mix_feat.detach())

        ## Else
        self.mask_h, _ = decide_circle(r=self.config.radius, N=int(self.real_A.shape[0]), C=self.real_A.shape[1], L=self.config.load_size)
        self.mask_1 = torch.ones(self.real_A.shape).to(self.device)
        self.mask_h = self.mask_h.to(self.device)
        
 
    def calc_G_loss(self):
        # L1 loss
        self.G_img = self.criterion_L1(self.real_B, self.trs_AtoB) *self.lambda_img
        
        G_pix_low = self.criterion_L1(self.real_B_low, self.trs_AtoB_low)
        G_pix_high = self.criterion_L1(self.real_B_high, self.trs_AtoB_high)
        self.G_pix = (G_pix_low + G_pix_high) * self.lambda_pix
        
        self.G_trs = (self.G_img + self.G_pix) * self.lambda_trs

        # Perceptual loss
        self.G_percept = self.vgg_loss.perceptual_loss(self.real_B, self.trs_AtoB)

        # Contrastive loss
        #self.G_contrast = self.vgg_loss.contrastive_loss()
        self.G_contrast = 0

        # Adversarial Loss
        self.G_GAN_trs = self.criterion_GAN(self.D_fake_AtoB, True)
        self.G_GAN = self.G_GAN_trs * self.lambda_GAN_G
        
        # Patch GAN Loss
        self.G_PD = self.criterion_GAN(self.PD_fake, True) * self.lambda_PD_G
      
        # Total loss
        self.G_loss = self.G_trs + self.G_percept + self.G_contrast + self.G_GAN + self.G_PD
        
    def calc_D_loss(self):
        # 1. GAN Loss
        D_GAN_real_B = self.criterion_GAN(self.D_real_B, True)
        D_GAN_trs = self.criterion_GAN(self.D_fake_AtoB, False)
        #D_GAN_smooth = self.criterion_GAN(self.D_smooth, False)
        D_GAN_smooth = 0
        self.D_GAN = (D_GAN_real_B + 0.5 * D_GAN_trs + 0.5 * D_GAN_smooth) * self.lambda_GAN_D

        # 2. Patch GAN Loss
        D_PD_real = self.criterion_GAN(self.PD_real, True)
        D_PD_fake = self.criterion_GAN(self.PD_fake, False)
        self.D_PD = (D_PD_real + D_PD_fake) * self.lambda_PD_D

        # Total Loss
        self.D_loss = self.D_GAN + self.D_PD
        

    def train(self, tot_itr, data):
        #self.lambda_GAN_G = (1 + self.lr) * self.lambda_GAN_G
        #self.lambda_PD_G = (1 + self.lr) * self.lambda_PD_G

        # Generator Loss
        self.set_requires_grad([self.netE, self.netG], True)
        self.set_requires_grad([self.netD, self.netPD], False)

        self.forward(data)
        self.calc_G_loss()

        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()
        self.G_loss.backward()
        self.optimizer_E.step()
        self.optimizer_G.step()
        
        # Discriminator Loss
        self.set_requires_grad([self.netE, self.netG], False)
        self.set_requires_grad([self.netD, self.netPD], True)

        self.forward(data)
        self.calc_D_loss()

        self.optimizer_D.zero_grad()
        self.optimizer_PD.zero_grad()
        self.D_loss.backward()
        self.optimizer_D.step()
        self.optimizer_PD.step()

        train_dict = {}
        train_dict['G_img'] = self.G_img
        train_dict['G_pix'] = self.G_pix
        train_dict['G_trs'] = self.G_trs
        train_dict['G_percept'] = self.G_percept
        train_dict['G_GAN'] = self.G_GAN
        train_dict['G_PD'] = self.G_PD
        train_dict['G_loss'] = self.G_loss
        train_dict['D_PD'] = self.D_PD
        train_dict['D_GAN'] = self.D_GAN
        train_dict['D_loss'] = self.D_loss
        
        #train_dict['rec_AtoA'] = self.rec_AtoA
        train_dict['fake_AtoB'] = self.trs_AtoB
        #train_dict['rec_AtoA_low'] = self.rec_AtoA_low
        #train_dict['rec_AtoA_high'] = self.rec_AtoA_high
        train_dict['fake_AtoB_low'] = self.trs_AtoB_low
        train_dict['fake_AtoB_high'] = self.trs_AtoB_high
        
        return train_dict

    def val(self, data):
        with torch.no_grad():
            self.forward(data)
            self.calc_G_loss()
            self.calc_D_loss()
        
            val_dict = {}
            val_dict['G_loss'] = self.G_loss
            val_dict['D_loss'] = self.D_loss
            val_dict['real_A'] = self.real_A
            val_dict['fake_AtoB'] = self.trs_AtoB
            val_dict['real_B'] = self.real_B

        return val_dict

    def test(self, data):
        with torch.no_grad():
            self.forward(data)

            test_dict = {}
            test_dict['fake_AtoB'] = self.trs_AtoB

        return test_dict
    
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


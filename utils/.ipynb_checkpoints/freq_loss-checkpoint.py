import torch.nn.functional as F
import torch
import cv2
import sys
from utils.freq_pixel_loss import *
from utils.freq_fourier_loss import *
from utils.moments_loss import *
from utils.azimuthalAverage import *

sys.path.append("..")

def recon_loss(config, real_image, fake_image, mask_1, gauss_kernel, az_loss):
    ''' weights for spatial domain'''
    w_recon_pix_l1_low = config.w_recon_pix_l1_low
    #w_recon_pix_moment_low = config.w_recon_pix_moment_low
    w_recon_pix_l1_high = config.w_recon_pix_l1_high
    w_recon_pix_moment_high = config.w_recon_pix_moment_high
    ''' weights for frequency domain'''
    w_recon_fft_l1 = config.w_recon_fft_l1
    w_recon_fft_az = config.w_recon_fft_az
    #w_recon_fft_moment = config.w_recon_fft_moment
    cmd_weights = config.CMDweights  # must be in form of list
    ''' preparation '''
    real_img_freq_combined = find_fake_freq(real_image, gauss_kernel)
    fake_img_freq_combined = find_fake_freq(fake_image, gauss_kernel)

    ## reconstruction loss @ spatial domain
    # Loss1 : rec, pix, l1 - RGB 3 channel calc
    loss_rec_pix_l1_low = F.l1_loss(real_img_freq_combined[:, :3, :, :], fake_img_freq_combined[:, :3, :, :])
    loss_rec_pix_l1_high = F.l1_loss(real_img_freq_combined[:, 3:6, :, :], fake_img_freq_combined[:, 3:6, :, :])
    loss_rec_pix_l1 = (w_recon_pix_l1_low * loss_rec_pix_l1_low) + (w_recon_pix_l1_high * loss_rec_pix_l1_high)

    # Loss2 : rec, pix, moment - RGB 3 channel calc
    #rec_pix_moment_low = CMD_loss(torch.sigmoid(real_img_freq_combined[:, :3, :, :]), k=len(cmd_weights), weights=cmd_weights)
    rec_pix_moment_high = CMD_loss(torch.sigmoid(real_img_freq_combined[:, 3:6, :, :]), k=len(cmd_weights), weights=cmd_weights)

    #loss_rec_pix_moment_low = rec_pix_moment_low(torch.sigmoid(fake_img_freq_combined[:, :3, :, :]))
    loss_rec_pix_moment_high = rec_pix_moment_high(torch.sigmoid(fake_img_freq_combined[:, 3:6, :, :]))
    #loss_rec_pix_moment = (w_recon_pix_moment_low * loss_rec_pix_moment_low) + (w_recon_pix_moment_high * loss_rec_pix_moment_high)
    loss_rec_pix_moment = w_recon_pix_moment_high * loss_rec_pix_moment_high

    ## reconstruction loss @ frequency domain
    # Loss3 : rec, fft, l1 - RGB 3 channel calc
    if w_recon_fft_l1 > 0:
        loss_rec_fft_l1 = fft_L1_loss(real_image, fake_image, mask=1, is_color=True)
        loss_rec_fft_l1 = w_recon_fft_l1 * loss_rec_fft_l1
    else:
        loss_rec_fft_l1 = 0

    # Loss4 : rec, fft, az - RGB 3 channel calc
    if w_recon_fft_az > 0:
        azimuthal_real = azimuthal(real_image, mask_1)
        azimuthal_fake = azimuthal(fake_image, mask_1)
        loss_rec_fft_az = az_loss(azimuthal_real, azimuthal_fake)
        loss_rec_fft_az = w_recon_fft_az * loss_rec_fft_az
    else:
        loss_rec_fft_az = 0

    # Loss4 : rec, fft, moment - RGB 3 channel calc
    #real_fft = calc_fft(real_image)
    #fake_fft = calc_fft(fake_image)

    #rec_fft_moment = CMD_loss(torch.sigmoid(real_fft), k=len(cmd_weights), weights=cmd_weights)
    #loss_rec_fft_moment = rec_fft_moment(torch.sigmoid(fake_fft))
    #loss_rec_fft_moment = w_recon_fft_moment * loss_rec_fft_moment

    # Total Recon Loss
    recon_loss = loss_rec_pix_l1 + loss_rec_pix_moment + loss_rec_fft_l1 + loss_rec_fft_az
    return recon_loss

def trans_loss(config, real_image, fake_image, mask_h, gauss_kernel, az_loss):
    ''' weights for spatial domain'''
    w_trans_pix_l1 = config.w_trans_pix_l1
    w_trans_pix_moment = config.w_trans_pix_moment
    w_trans_fft_l1 = config.w_trans_fft_l1
    #w_trans_fft_moment = config.w_trans_fft_moment
    w_trans_fft_az = config.w_trans_fft_az
    cmd_weights = config.CMDweights # must be in form of list

    ''' preparation '''
    real_img_freq_high = find_fake_freq(real_image, gauss_kernel)[:, 3:6, :, :]
    fake_img_freq_high = find_fake_freq(fake_image, gauss_kernel)[:, 3:6, :, :]

    ''' Translation loss'''
    ## translation loss @ spatial domain
    # Loss1 : trans, pix, l1 - gray calc, high freq only
    loss_trans_pix_l1 = F.l1_loss(real_img_freq_high, fake_img_freq_high)

    trans_pix_moment = CMD_loss(torch.sigmoid(real_img_freq_high), k=len(cmd_weights), weights=cmd_weights)
    loss_trans_pix_moment = trans_pix_moment(torch.sigmoid(fake_img_freq_high))

    loss_trans_pix_l1 = w_trans_pix_l1 * loss_trans_pix_l1
    loss_trans_pix_moment = w_trans_pix_moment * loss_trans_pix_moment

    ## translation loss @ frequency domain
    # Loss1 : trans, fft, l1 - gray calc, high freq only
    if w_trans_fft_l1 > 0:
        loss_trans_fft_l1 = fft_L1_loss(real_image, fake_image, mask=mask_h, is_color=True)
        loss_trans_fft_l1 = w_trans_fft_l1 * loss_trans_fft_l1
    else:
        loss_trans_fft_l1 = 0

    # Loss2 : trans, fft, moment - gray calc, high freq only
    #real_image_gray = real_image[:, 0] * 0.299 + real_image[:, 1] * 0.587 + real_image[:, 2] * 0.114
    #fake_image_gray = fake_image[:, 0] * 0.299 + fake_image[:, 1] * 0.587 + fake_image[:, 2] * 0.114
    #gray_real = calc_fft(real_image_gray) * mask_h
    #gray_fake = calc_fft(fake_image_gray) * mask_h
    
    #gray_real = gray_real.unsqueeze(dim=1)
    #gray_fake = gray_fake.unsqueeze(dim=1)

    #trans_fft_moment = CMD_loss(torch.sigmoid(gray_real), k=len(cmd_weights), weights=cmd_weights)
    #loss_trans_fft_moment = trans_fft_moment(torch.sigmoid(gray_fake))

    #loss_trans_fft_moment = w_trans_fft_moment * loss_trans_fft_moment

    # Loss2 : trans, fft, AZ - gray calc, high freq only
    if w_trans_fft_az > 0:
        azimuthal_real = azimuthal(real_image, mask_h)
        azimuthal_fake = azimuthal(fake_image, mask_h)
        loss_trans_fft_az = az_loss(azimuthal_real.detach(), azimuthal_fake.detach())
        loss_trans_fft_az = w_trans_fft_az * loss_trans_fft_az
    else:
        loss_trans_fft_az = 0

    ## Total translation loss
    trans_loss = loss_trans_pix_l1 + loss_trans_pix_moment + loss_trans_fft_l1
    return trans_loss


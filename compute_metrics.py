from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import numpy as np
import glob
import math
from math import exp
from path import Path

######################################
## Calculate metrics : PSNR, SSIM ##
######################################
def PSNR(real_img, fake_img):
    mse = np.mean((real_img - fake_img) ** 2)
    # real_img == fake_img : PSNR 정의될 수 X
    if mse == 0:
        return 100
    # assert np.min(real_img) == np.min(fake_img)
    # assert np.max(real_img) == np.max(fake_img)
    pixel_max = np.max(real_img) - np.min(real_img)
    psnr = 10 * math.log10(pixel_max ** 2 / mse)
    return psnr

def MAE_MSE(real_img, fake_img):
    pix_abs = 0
    pix_sqrt = 0
    #real_img = np.copy(real_img[0])
    for i in range(real_img.shape[0]):
        for j in range(real_img.shape[1]):
            pix_abs += (np.abs(real_img[i, j] - fake_img[i, j])) / 255.0
            pix_sqrt += ((real_img[i, j] - fake_img[i, j]) / 255.0) ** 2
    mae = pix_abs / (real_img.shape[0] * real_img.shape[1])
    mse = pix_sqrt / (real_img.shape[0] * real_img.shape[1])
    return mae, mse

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0)
    # print(_1D_window.shape, _2D_window.shape)  # torch.Size([5, 1]) torch.Size([1, 5, 5])
    #window = Variable(_2D_window.expand(channel, window_size, window_size).contiguous())
    window = torch.Tensor(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def SSIM(real_img, fake_img, size_average=True):
    real_img = torch.reshape(torch.from_numpy(real_img), (1, 1, real_img.shape[-1], real_img.shape[-1])).float()
    fake_img = torch.reshape(torch.from_numpy(fake_img), (1, 1, real_img.shape[-1], real_img.shape[-1])).float()
   
    # real_img shape: 1x64x64
    channel = real_img.shape[0]
    window_size = 5
    # window shape: 1x5x5
    window = create_window(window_size, channel)
    window = torch.reshape(window, (1, 1, 5, 5))

    mu1 = F.conv2d(real_img, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(fake_img, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(real_img * real_img, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(fake_img * fake_img, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(real_img * fake_img, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    

    
def main():
    #org_img_train = np.array(Image.open("/scratch/connectome/conmaster/I2I_translation/OctMRI_DTI/Generated_images/mri_5000_FA_swap_pix_smooth_Loss_PD_256/Train/97_41_B.png").getdata()) 
    #OCCAY_img_train = np.array(Image.open("/scratch/connectome/conmaster/I2I_translation/OctMRI_DTI/Generated_images/mri_5000_FA_swap_pix_smooth_Loss_PD_256/Train/97_41_fake.png").getdata())


    img_path = Path('/scratch/connectome/conmaster/I2I_translation/MRI/DTI_NST/Generated_images/mri_5000_FA_E256_Perceptloss/Test')

    #img_dir = '/scratch/connectome/conmaster/I2I_translation/OctMRI_DTI/Generated_images/mri_5000_FA_swap_pix_smooth_Loss_PD_256/Test'
    #img_path = Path(img_dir)

    psnr = 0.
    mssim = 0.
    t_mae = 0.
    t_mse = 0.
    images = sorted(img_path.glob('*_fake.png'))
    print(len(images))
    for image in images:
        file_n = image.split('_fake')[0]
        B_img = np.array(Image.open(file_n+'_B.png').getdata())
        image = np.array(Image.open(image).getdata())

        image = np.reshape(image, [256, 256])
        B_img = np.reshape(B_img, [256, 256])

        mask = image.copy()
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if B_img[i, j] < 0.2:
                    mask[i, j] = 0.0
                else:
                    mask[i, j] = 1.0

        image = image * mask
        B_img = B_img * mask

        psnr += PSNR(B_img, image)
        mssim += SSIM(B_img, image)
        mae, mse = MAE_MSE(B_img, image)
        t_mae += mae
        t_mse += mse
    print("--")
    psnr = float(psnr/float(len(images)))
    mssim = float(mssim/float(len(images)))
    t_mae = float(t_mae/float(len(images)))
    t_mse = float(t_mse/float(len(images)))
    print("PSNR:", psnr)
    print("SSIM:", mssim)
    print("MAE:", t_mae)
    print("MSE:", t_mse)

    #org_img_test = np.array(Image.open("/scratch/connectome/conmaster/I2I_translation/OctMRI_DTI/Generated_images/mri_5000_FA_E256_Perceptloss/Test/42_sub-NDARINVTX4K8UGF_B.png").getdata())
    #OCCAY_img_test = np.array(Image.open("/scratch/connectome/conmaster/I2I_translation/OctMRI_DTI/Generated_images/mri_5000_FA_E256_Perceptloss/Test/42_sub-NDARINVTX4K8UGF_fake.png").getdata())

    """
    #print(np.min(org_img_test), np.max(OCCAY_img_test))
    #print(org_img_test.shape, OCCAY_img_test.shape)
    org_img_train = np.reshape(org_img_train, [1, 1, 256, 256])
    OCCAY_img_train = np.reshape(OCCAY_img_train, [1, 1, 256, 256])

    org_img_test = np.reshape(org_img_test, [256, 256])
    OCCAY_img_test = np.reshape(OCCAY_img_test, [256, 256])

    train_ssim = SSIM(org_img_train, OCCAY_img_train)
    #train_ssim = 1 - tf.image.ssim(org_img_train, OCCAY_img_train, max_val=1.0)[0]
    print("Train SSIM:", train_ssim)

    test_ssim = SSIM(org_img_test, OCCAY_img_test)
    test_mae, test_mse = MAE_MSE(org_img_test, OCCAY_img_test)
    #test_ssim = 1 - tf.image.ssim(org_img_test, OCCAY_img_test, max_val=1.0)[0]
    print("Test SSIM:", test_ssim)
    print("Test MAE/MSE:", test_mae, test_mse)
   
    
    train_mae, train_mse = MAE_MSE(org_img_train, OCCAY_img_train)
    print("Train MAE/MSE:", train_mae, train_mse)
    test_mae, test_mse = MAE_MSE(org_img_test, OCCAY_img_test)
    print("Test MAE/MSE:", test_mae, test_mse)
    
    
    ###############################################################
    original_FA = np.load('/storage/connectome/GANBERT/data/FA/sub-NDARINVU102M8W0.FA_256.npy')
    #result_FA = Image.open()
    fake_FA = np.asarray(Image.open('/scratch/connectome/conmaster/I2I_translation/OctMRI_DTI/Generated_images/mri_5000_FA_swap_pix_smooth_Loss_PD_256/Test/31_sub-NDARINVU102M8W0_fake.png'))

    print(original_FA.shape, fake_FA.shape)

    mask = fake_FA.copy()
    for i in range(fake_FA.shape[0]):
        for j in range(fake_FA.shape[1]):
            if original_FA[i, j] < 0.5:
                mask[i, j] = 0.0
            else:
                mask[i, j] = 1.0

    filter_FA = fake_FA * mask
    original_filt_FA = original_FA * mask
    print(filter_FA.shape)

    filter_mae, filter_mse = MAE_MSE(original_filt_FA, fake_FA)
    print(filter_mae, filter_mse)
    """

if __name__ == '__main__':
    main()

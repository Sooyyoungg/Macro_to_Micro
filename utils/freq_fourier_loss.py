"""
Description: fourier spectrum loss for GAN
"""
import torch
import pdb

def calc_fft(image):
    '''image is tensor, N*C*H*W'''
    '''output is N*C*H*W*2'''
    fft = torch.view_as_real(torch.fft.fft2(image))
    #fft = torch.view_as_real(torch.fft.fft2(image, dim=(-2, -1), norm=None))
    fft_mag = torch.log(1 + torch.sqrt(fft[..., 0] ** 2 + fft[..., 1] ** 2 + 1e-8))
    return fft_mag

def fft_L1_loss(real_image, fake_image, mask=1, is_color=True):
    criterion_L1 = torch.nn.L1Loss()

    if is_color:    # Color image     # for reconstruciton
        real_fft = calc_fft(real_image)
        fake_fft = calc_fft(fake_image)
        loss = criterion_L1(real_fft, fake_fft)
        return loss
    else:           # Grayscale image # for translation.
        #convert image to grayscale
        real_image_gray = real_image[:, 0] * 0.299 + real_image[:, 1] * 0.587 + real_image[:, 2] * 0.114
        fake_image_gray = fake_image[:, 0] * 0.299 + fake_image[:, 1] * 0.587 + fake_image[:, 2] * 0.114
        
        real_fft = torch.fft.fftshift(calc_fft(real_image_gray))
        fake_fft = torch.fft.fftshift(calc_fft(fake_image_gray))
        loss = criterion_L1(real_fft * mask, fake_fft * mask)
        return loss

def decide_circle(N=4, C=3, L=256, r=96, size=256): # for high freq mask.
    x = torch.ones((N, C, L, L))
    for i in range(L):
        for j in range(L):
            if (i- L/2 + 0.5)**2 + (j- L/2 + 0.5)**2 < r **2:
                x[:,:,i,j]=0
    return x, torch.ones((N, C, L, L)) - x


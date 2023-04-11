"""
Description: pixel loss for GAN
"""

import torch.nn.functional as F
import torch
import cv2
import sys
sys.path.append("..")

def get_gaussian_blur(x, k, stride=1, padding=0):
    res = []
    x = F.pad(x, (padding, padding, padding, padding), mode='constant', value=0)
    for xx in x.split(1,1):
        res.append(F.conv2d(xx, k, stride=stride, padding=0))
    return torch.cat(res,1)

def get_low_freq(im, gauss_kernel):
    padding = (gauss_kernel.shape[-1] - 1) // 2
    low_freq = get_gaussian_blur(im, gauss_kernel, padding=padding)
    return low_freq

def get_gaussian_kernel(size=3):
    kernel = cv2.getGaussianKernel(size, 0).dot(cv2.getGaussianKernel(size, 0).T)
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
    return kernel

def find_fake_freq(im, gauss_kernel, index=None):
    # low freq - RGB, high freq - grayscale
    padding = (gauss_kernel.shape[-1] - 1) // 2
    low_freq = get_gaussian_blur(im, gauss_kernel, padding=padding)
    # CCIR 601 (most digital standard definition formats), Rec.601 luma is as below.
    if im.shape[1] == 3:
        im_gray = im[:, 0, ...] * 0.299 + im[:, 1, ...] * 0.587 + im[:, 2, ...] * 0.114
        im_gray = im_gray.unsqueeze_(dim=1).repeat(1, 3, 1, 1)
        low_gray = get_gaussian_blur(im_gray, gauss_kernel, padding=padding)
        high_freq = im_gray - low_gray
    else:
        high_freq = im - low_freq
    return torch.cat((low_freq, high_freq), 1)


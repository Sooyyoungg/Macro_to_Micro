# from https://github.com/cc-hpc-itwm/UpConv.
import numpy as np
from torch.autograd import Variable
from utils.freq_fourier_loss import *

'''image == magnitude_spectrum'''
def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image = magnitude_spectrum
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fracitonal pixels).

    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof

def azimuthal(config, image, mask=None, is_color=True):
    if config.load_size == 256:
        N = 179
    else:
        N = 88
    if is_color:
        fft_img = torch.fft.fftshift(calc_fft(image))
        fft_img = fft_img * mask
        psd1D_img = np.zeros([image.shape[0], 3, N])
    else:
        image_gray = image[:, 0] * 0.299 + image[:, 1] * 0.587 + image[:, 2] * 0.114
        fft_img = torch.fft.fftshift(calc_fft(image_gray))
        fft_img = fft_img * mask
        psd1D_img = np.zeros([image.shape[0], N])

    for b in range(image.shape[0]):
        if is_color:
            for c in range(image.shape[1]):
                psd1D = azimuthalAverage(fft_img[b,c,:].cpu().detach().numpy())
                psd1D = (psd1D-np.min(psd1D))/(np.max(psd1D)-np.min(psd1D))
                psd1D_img[b,c,:] = psd1D
        else:
            psd1D = azimuthalAverage(fft_img[b].cpu().detach().numpy())
            psd1D = (psd1D-np.min(psd1D))/(np.max(psd1D)-np.min(psd1D))
            psd1D_img[b] = psd1D

    psd1D_img = torch.from_numpy(psd1D_img).float()
    psd1D_img = Variable(psd1D_img, requires_grad=True)

    return psd1D_img

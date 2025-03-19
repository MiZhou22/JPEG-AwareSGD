import numpy as np
import torch

def srgb_gamma2lin(im_in):
    """converts from sRGB to linear color space"""
    thresh = 0.04045
    if torch.is_tensor(im_in):
        im_out = torch.where(im_in <= thresh, im_in / 12.92, ((im_in + 0.055) / 1.055) ** 2.4)
    else:
        im_out = np.where(im_in <= thresh, im_in / 12.92, ((im_in + 0.055) / 1.055) ** 2.4)
    return im_out


def srgb_lin2gamma(im_in):
    """converts from linear to sRGB color space"""
    thresh = 0.0031308
    if torch.is_tensor(im_in):
        im_out = torch.where(im_in <= thresh, 12.92 * im_in, 1.055 * (im_in ** (1 / 2.4)) - 0.055)
    else:
        im_out = np.where(im_in <= thresh, 12.92 * im_in, 1.055 * (im_in ** (1 / 2.4)) - 0.055)
    return im_out
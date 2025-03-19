import cv2
import numpy as np
import torch
import torch.nn.functional

from matplotlib import pyplot as plt

WAVE_LENGTH_COLOR_R, WAVE_LENGTH_COLOR_G, WAVE_LENGTH_COLOR_B = 638*1e-9, 520*1e-9, 450*1e-9
PIXEL_PITCH = 6.4*1e-6


# Pad the image
def image_tensor_padding(image, data_format="nchw", pad_factor=2, pad_var=0):
    """
    Fill the original image with pad_var to pad_factor times its original size
    :param image: tensor image
    :param data_format:HWC or NCHW
    :param pad_factor:
    :param pad_var:
    :return:
    """
    if not torch.is_tensor(image):
        raise Exception("Padding only for tensor")

    img_resolution = image.shape

    if data_format.upper() == "HWC":
        # M is the number of rows, N is the number of columns
        M = img_resolution[0]
        pad_M = int((pad_factor - 1) * M / 2)
        N = img_resolution[1]
        pad_N = int((pad_factor - 1) * N / 2)

        # image = np.pad(image, ((pad_M, pad_M), (pad_N, pad_N), (0, 0)), 'constant', constant_values=(pad_var,
        # pad_var))
        image = torch.nn.functional.pad(image, (0, 0, pad_N, pad_N, pad_M, pad_M), mode="constant", value=pad_var)
    elif data_format.upper() == "NCHW":
        # M is the number of rows, N is the number of columns
        M = img_resolution[-2]
        pad_M = int((pad_factor - 1) * M / 2)
        N = img_resolution[-1]
        pad_N = int((pad_factor - 1) * N / 2)

        # image = np.pad(image, ((0, 0), (0, 0), (pad_M, pad_M), (pad_N, pad_N)), 'constant', constant_values=((None,
        # None), (None, None), (pad_var, pad_var), (pad_var, pad_var)))
        image = torch.nn.functional.pad(image, (pad_N, pad_N, pad_M, pad_M), mode="constant", value=pad_var)

    return image  # same as the input shape


def crop(image,
         data_format="hwc",
         roi_shape=(1080, 1920),
         pad_val=None):
    """
    Crop the image to the size of ROI_Resolution, filling the cropped parts with pad_val
    :param image:           should be HWC, ndarray
    :param data_format:     HWC or NCHW or NDCHW
    :param roi_shape:       default is (1080, 1920)
    :param pad_val:         if pad_val==None, then no padding;
                            else padding the cropping the image to the original shape with value pad_val.
    :return:                return the padded image with shape: original(pad_val != None) or roi_resolution
    """
    data_format = data_format.upper()
    img_resolution = image.shape

    # get the target shape and the image shape
    H_target = roi_shape[0]
    W_target = roi_shape[1]
    if data_format == "HWC":
        H = img_resolution[0]
        W = img_resolution[1]
    elif "CHW" in data_format:
        # NCHW or NDCHW
        H = img_resolution[3] if data_format == "NDCHW" else img_resolution[2]
        W = img_resolution[4] if data_format == "NDCHW" else img_resolution[3]
    else:
        raise Exception("No suitable dataformat!")

    # get the index of the cropped image based on the center position
    cropping_H_start = int((H - H_target) / 2)
    cropping_H_end = int(H / 2 + H_target / 2 - 1)
    cropping_W_start = int((W - W_target) / 2)
    cropping_W_end = int(W / 2 + W_target / 2 - 1)

    # get the output image
    output_image = None
    if pad_val is not None:
        # padding
        output_image = torch.ones_like(image) * pad_val if torch.is_tensor(image) else np.ones_like(image) * pad_val
        if data_format == "HWC":
            output_image[cropping_H_start:cropping_H_end + 1, cropping_W_start:cropping_W_end + 1, :] = \
                image[cropping_H_start:cropping_H_end + 1, cropping_W_start:cropping_W_end + 1, :]
        elif "CHW" in data_format:
            output_image[..., cropping_H_start:cropping_H_end + 1, cropping_W_start:cropping_W_end + 1] = \
                image[..., cropping_H_start:cropping_H_end + 1, cropping_W_start:cropping_W_end + 1]
    else:
        # no padding
        if data_format == "HWC":
            output_image = image[cropping_H_start:cropping_H_end + 1, cropping_W_start:cropping_W_end + 1, ...]
        elif "CHW" in data_format:
            output_image = image[..., cropping_H_start:cropping_H_end + 1, cropping_W_start:cropping_W_end + 1]

    return output_image

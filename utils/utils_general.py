import cv2
import numpy as np
import torch
import os
import re
import torch.nn.functional as F
from matplotlib import pyplot as plt


def impad(target_hw,
          image,
          image_format="NCHW",
          pad_val=0):
    """

    @param target_hw:   target H dim and W dim
    @param image:       image waited to be padded
    @param image_format:image format(HWC, HW, NCHW, NDChW)
    @param pad_val:     constant pad value
    @return:
    """

    # pad based on it is a pytorch tensor or not
    def _pad(_image, pad_width, value):
        if torch.is_tensor(_image):
            _pad_width = tuple(reversed(pad_width))
            __pad_width = tuple((j for i in range(len(_pad_width)) for j in _pad_width[i]))
            output = F.pad(_image, pad=__pad_width, mode="constant", value=value)
        else:
            output = np.pad(_image, pad_width=pad_width, mode="constant", constant_values=value)
        return output

    image_padded = None
    h_target, w_target = target_hw[0], target_hw[1]
    if image_format == "HWC":
        h, w = image.shape[0], image.shape[1]
        h_pad = int(np.floor((h_target - h) / 2))
        w_pad = int(np.floor((w_target - w) / 2))
        image_padded = _pad(image, pad_width=((h_pad, h_pad), (w_pad, w_pad), (0, 0)), value=pad_val)
        image_padded = _pad(image_padded,
                            pad_width=((0, (h_target - h) % 2),
                                       (0, (w_target - w) % 2),
                                       (0, 0)),
                            value=pad_val)
    elif image_format == "HW":
        h, w = image.shape[0], image.shape[1]
        h_pad = int(np.floor((h_target - h) / 2))
        w_pad = int(np.floor((w_target - w) / 2))
        image_padded = _pad(image, pad_width=((h_pad, h_pad), (w_pad, w_pad)), value=pad_val)
        image_padded = _pad(image_padded,
                            pad_width=((0, (h_target - h) % 2),
                                       (0, (w_target - w) % 2)),
                            value=pad_val)
    elif image_format == "NCHW":
        h, w = image.shape[2], image.shape[3]
        h_pad = int(np.floor((h_target - h) / 2))
        w_pad = int(np.floor((w_target - w) / 2))
        image_padded = _pad(image, pad_width=((0, 0), (0, 0), (h_pad, h_pad), (w_pad, w_pad)), value=pad_val)
        image_padded = _pad(image_padded,
                            pad_width=((0, 0),
                                       (0, 0),
                                       (0, (h_target - h) % 2),
                                       (0, (w_target - w) % 2)),
                            value=pad_val)
    elif image_format == "NCDHW":
        h, w = image.shape[3], image.shape[4]
        h_pad = int(np.floor((h_target - h) / 2))
        w_pad = int(np.floor((w_target - w) / 2))
        image_padded = _pad(image, pad_width=((0, 0), (0, 0), (0, 0), (h_pad, h_pad), (w_pad, w_pad)), value=pad_val)
        image_padded = _pad(image_padded, pad_width=((0, 0),
                                                     (0, 0),
                                                     (0, 0),
                                                     (0, (h_target - h) % 2),
                                                     (0, (w_target - w) % 2)),
                            value=pad_val)
    else:
        raise Exception("No matched format!")

    return image_padded


# Read image
def imread(path,
           color="rgb",
           normalization=True,
           pad=False,
           pad_factor=2,
           pad_var=0,
           pad_shape=(1080, 1920)) -> np.ndarray:
    """

    @param path: image path
    @param color: color, rgb or gray
    @param normalization: whether using normalization or not
    @param pad: whether padding the image or not
    @param pad_factor: how much larger
    @param pad_var: padding var
    @param pad_shape: default is (1080, 1920)
    @return: read image in HWC
    """
    image = None
    image_min = None
    image_max = None
    # Read image based on different RGB formats
    if color.upper() in ["RGB", "R", "G", "B"]:
        image = cv2.imread(path)
        h, w = image.shape[0], image.shape[1]
        resize_shape = (int(h * 1920 / w), 1920) if np.abs(h - 1080) / h > np.abs(w - 1920) / w else (
            1080, int(w * 1080 / h))
        image = cv2.resize(image, tuple(reversed(resize_shape)), interpolation=cv2.INTER_NEAREST)
        image = impad(pad_shape, image, "HWC", pad_val=0)
        image = image[:, :, ::-1].copy()  # BGR -> RGB
        if color == "R":
            image = image[:, :, 0, None]
        elif color == "G":
            image = image[:, :, 1, None]
        elif color == "B":
            image = image[:, :, 2, None]

    elif color.upper() == "GRAY":
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        h, w = image.shape[0], image.shape[1]
        resize_shape = (int(h * 1920 / w), 1920) if np.abs(h - 1080) / h > np.abs(w - 1920) / w else (
            1080, int(w * 1080 / h))
        image = cv2.resize(image, tuple(reversed(resize_shape)), interpolation=cv2.INTER_NEAREST)
        image = impad(pad_shape, image, "HW", pad_val=0)
        image = image[:, :, None]
    # image = image.astype("float32")

    if image is None:
        raise Exception("Image is None, Check the code!")

    # Fill the original image with pad_var to be pad_factor times larger
    if pad:
        img_resolution = np.shape(image)

        # M is the number of rows, N is the number of columns
        M = img_resolution[0]
        pad_M = int((pad_factor - 1) * M / 2)
        N = img_resolution[1]
        pad_N = int((pad_factor - 1) * N / 2)

        image = np.pad(image,
                       ((pad_M, pad_M), (pad_N, pad_N), (0, 0)),
                       'constant',
                       constant_values=(pad_var, pad_var))

    # Normalization
    if normalization:
        image_max, image_min = image_channel_max_min(image)
        # image = (image - image_min) / (image_max - image_min + 1e-9)
        dmax = 65535
        if image.dtype == "uint8":
            dmax = 255
        image = image / dmax

    image = image.astype(np.float32)
    return image  # HWC


# Display image
def imshow(image, three_dim=False):
    if three_dim is not True:
        plt.imshow(image, cmap="gray", vmin=0, vmax=1)
        plt.axis('off')
        plt.show()
    else:
        ax = plt.axes(projection='3d')
        x, y = np.meshgrid(np.linspace(0, np.shape(image)[1] - 1, np.shape(image)[1]),
                           np.linspace(0, np.shape(image)[0] - 1, np.shape(image)[0]), indexing='xy')
        ax.plot_surface(x, y, cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY),
                        cmap="rainbow")
        plt.axis('off')
        plt.show()


def image_channel_max_min(image,
                          axis=(0, 1)):
    if torch.is_tensor(image):
        image_max = torch.amax(image, dim=axis, keepdim=True)
        image_min = torch.amin(image, dim=axis, keepdim=True)
    else:
        image_max = np.amax(image, axis=axis, keepdims=True)
        image_min = np.amin(image, axis=axis, keepdims=True)

    return image_max, image_min


def image_normalization(image,
                        axis=(0, 1)):
    image_max, image_min = image_channel_max_min(image, axis)
    image_normalized = (image - image_min) / (image_max - image_min + 10 ** -9)
    return image_normalized
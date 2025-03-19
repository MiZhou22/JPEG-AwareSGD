# Pytorch
import torch
import torch.nn as nn
# Local
from DiffJPEG.modules import compress_jpeg, compress_jpeg_gray, decompress_jpeg, decompress_jpeg_gray
from DiffJPEG.utils import diff_round, quality_to_factor


class DiffJPEG(nn.Module):
    def __init__(self, height, width, differentiable=True, quality=80):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme. 
        '''
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round
        factor = quality_to_factor(quality)
        # self.compress = compress_jpeg(rounding=rounding, factor=factor)
        self.compress_gray = compress_jpeg_gray(rounding=rounding, factor=factor)
        # self.decompress = decompress_jpeg(height, width, rounding=rounding, factor=factor)
        self.decompress_gray = decompress_jpeg_gray(height, width, rounding=rounding, factor=factor)

    def forward(self, x):
        """

        @param x: the images in NCHW
        @return: the images after being feed into the encoder and decoder
        """
        y, quantized_blocks = self.compress_gray(x)
        recovered = self.decompress_gray(y)
        return recovered, quantized_blocks

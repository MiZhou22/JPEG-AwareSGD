import torch

from abc import ABC, abstractmethod
from utils.Angular_Spectrum_Method import Angular_Spectrum_Method


# Abstract base class
class CGH_iterative(ABC):
    def __init__(self,
                 wave_length,
                 slm_resolution,
                 pixel_pitch,
                 device,
                 linear_convolution=True):
        super(CGH_iterative, self).__init__()

        self.device = device
        self.linear_convolution = linear_convolution
        self.wave_length = torch.as_tensor(wave_length, device=self.device)
        self.pixel_pitch = torch.as_tensor(pixel_pitch, device=self.device)
        self.slm_resolution = slm_resolution                                                                    # HW
        self.color_channel_num = self.wave_length.shape[2]                                                      # C
        self.image_resolution = (1, self.color_channel_num, self.slm_resolution[0], self.slm_resolution[1])     # 1CHW
        self.propagator = Angular_Spectrum_Method(wave_length=self.wave_length,
                                                  slm_resolution=self.slm_resolution,
                                                  pixel_pitch=self.pixel_pitch,
                                                  gpu_id=self.device,
                                                  linear_convolution=self.linear_convolution)

    # how to remove the warning from the child class when the signatures are mismatched
    # https://stackoverflow.com/questions/6034662/python-method-overriding-does-signature-matter
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def _3d(self, *args, **kwargs):
        pass

    def _2d(self, *args, **kwargs):
        pass


if __name__ == "__main__":
    pass
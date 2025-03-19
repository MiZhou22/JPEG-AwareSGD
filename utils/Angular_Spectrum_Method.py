import torch.nn.functional

from utils.utils_asm import *

class Angular_Spectrum_Method:
    """
    Angular spectrum method operator implementation: propagation from a pure phase SLM to an image plane at distance z
    """

    def __init__(self,
                 wave_length,  # Wavelength
                 slm_resolution,  # SLM resolution
                 pixel_pitch,  # SLM pixel pitch
                 gpu_id,
                 linear_convolution=True):
        """
        Initialization of the angular spectrum method.
        When using NCHW format, the output is by default on the GPU. When using HWC, the output is on the CPU.

        Note 1: In our implementation of convolution, padding zeros and cropping are computational operations
        without actual physical meaning. Thus, image_resolution still represents the final image size,
        which is also the size of the spatial light modulator (SLM).

        :param wave_length: 1*1*C
        :param slm_resolution: (row, column)
        :param pixel_pitch: scalar
        :param device: torch.device("cuda") or torch.device("cpu")
        :param linear_convolution: Implements linear convolution instead of circular convolution.
                                   Using linear convolution requires padding zeros, followed by cropping at the end.
        """
        # Set parameters
        self.gpu_id = gpu_id
        self.linear_convolution = linear_convolution
        self.pixel_pitch = torch.as_tensor(pixel_pitch, device=self.gpu_id, dtype=torch.float32)
        self.num_color_num = wave_length.shape[2]
        self.slm_resolution = slm_resolution

        # Linear convolution requires zero-padding
        if linear_convolution:
            self.image_resolution = (slm_resolution[0] * 2, slm_resolution[1] * 2, self.num_color_num)
        else:
            self.image_resolution = (slm_resolution[0], slm_resolution[1], self.num_color_num)
        # After zero-padding, set the number of rows and columns
        self.num_rows = torch.as_tensor(self.image_resolution[0], device=self.gpu_id)
        self.num_columns = torch.as_tensor(self.image_resolution[1], device=self.gpu_id)

        # Wavelength
        self.wave_length = torch.as_tensor(wave_length, device=self.gpu_id, dtype=torch.float32).permute(2, 0, 1)  # C*1*1
        self.wave_length = self.wave_length[None, None, ...]  # 1*1*C*1*1

        # Compute fx, fy, and convert to coordinate points, 111HW (NDCHW format)
        self.fx = torch.fft.fftfreq(self.num_rows, self.pixel_pitch).to(self.gpu_id)  # High frequency at the center
        self.fy = torch.fft.fftfreq(self.num_columns, self.pixel_pitch).to(self.gpu_id)
        self.fx, self.fy = torch.meshgrid(self.fx, self.fy, indexing='ij')
        self.fx = self.fx[None, None, None, ...]  # 1*1*1*H*W
        self.fy = self.fy[None, None, None, ...]  # 1*1*1*H*W

    def compute_H(self, z):
        """
        Compute the linear transfer function H matrix, with the same size as the incident wave

        @param z: NDCHW -> ND*1*1*1
        @return: System function H
        """
        # Compute the mask in ASM
        mask = torch.sqrt(torch.square(self.fx) + torch.square(self.fy)) < (1 / self.wave_length)  # 1*1*C*H*W

        # Compute system function
        # If z is a scalar, Python's broadcasting mechanism treats z as (1, 1, 1, 1, 1) in NDCHW format
        H = torch.exp(1j * 2 * torch.pi * z *
                      torch.sqrt(
                          1 / self.wave_length ** 2 - torch.square(self.fx) - torch.square(self.fy))
                      ) * mask  # NDCHW

        # Band-limited ASM - Matsushima et al. (2009)
        height = self.pixel_pitch * self.num_rows
        width = self.pixel_pitch * self.num_columns
        fy_max = 1 / torch.sqrt(torch.square(2 * z / width) + 1) / self.wave_length  # 1*1*C*1*1
        fx_max = 1 / torch.sqrt(torch.square(2 * z / height) + 1) / self.wave_length  # 1*1*C*1*1
        H_filter = (torch.abs(self.fx) < fx_max) & (torch.abs(self.fy) < fy_max)  # 1*1*C*H*W
        H = H * H_filter

        return H

    def __call__(self, u, z):
        """
        @param u: Incident complex amplitude field u (NCHW format)
        @param z: Diffraction distance z (ND*1*1*1)
        @return: Output field
        """
        # Input
        # Is it necessary to use fftshift before fft? No need:
        # https://stackoverflow.com/questions/32166879/do-i-need-to-call-fftshift-before-calling-fft-or-ifft#:~:text=So%2C%20to%20answer%20your%20question,applying%20an%20ifftshift%20or%20fftshift%20.
        self.u = torch.as_tensor(u, device=self.gpu_id)  # NCHW
        if self.linear_convolution:
            self.u = image_tensor_padding(self.u)  # NCHW
        self.U = torch.fft.fft2(self.u, dim=(-2, -1), norm="ortho")  # NCHW
        self.U = self.U[:, None, :, :, :] if len(self.U.shape) == 4 else self.U  # NDCHW

        # System function
        z = torch.as_tensor(z, device=self.gpu_id)  # 1*1*1*batch_size*num_depth
        H = self.compute_H(z)  # NDCHW

        # Output
        self.output_field = torch.fft.ifft2(self.U * H, dim=(-2, -1), norm="ortho")  # NDCHW
        if self.linear_convolution:
            self.output_field = crop(image=self.output_field,
                                     data_format="ndchw",
                                     roi_shape=self.slm_resolution,
                                     pad_val=None)

        return self.output_field  # NDCHW

    def slm_resolution_setter(self, slm_resolution):
        """
        Update SLM resolution
        """
        self.slm_resolution = slm_resolution
        # Linear convolution requires zero-padding
        if self.linear_convolution:
            self.image_resolution = (slm_resolution[0] * 2, slm_resolution[1] * 2, self.num_color_num)
        else:
            self.image_resolution = (slm_resolution[0], slm_resolution[1], self.num_color_num)
        # After zero-padding, set the number of rows and columns
        self.num_rows = torch.as_tensor(self.image_resolution[0], device=self.gpu_id)
        self.num_columns = torch.as_tensor(self.image_resolution[1], device=self.gpu_id)

        # Compute fx, fy, and convert to coordinate points, 111HW (NDCHW format)
        self.fx = torch.fft.fftfreq(self.num_rows, self.pixel_pitch).to(self.gpu_id)  # High frequency at the center
        self.fy = torch.fft.fftfreq(self.num_columns, self.pixel_pitch).to(self.gpu_id)
        self.fx, self.fy = torch.meshgrid(self.fx, self.fy, indexing='ij')
        self.fx = self.fx[None, None, None, ...]  # 1*1*1*H*W
        self.fy = self.fy[None, None, None, ...]  # 1*1*1*H*W

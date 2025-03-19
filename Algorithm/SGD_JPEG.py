"""
phs -jpeg-> phs -> ASM -> Loss backpropagation
"""

import torch
import numpy as np
# from torchvision.io import decode_jpeg, encode_jpeg   not differential require int8 input and output
import tqdm

from DiffJPEG.DiffJPEG import DiffJPEG

from Algorithm.CGH_algorithm import CGH_iterative
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

# Random seed to ensure reproducibility
from utils.utils_asm import crop
from utils.utils_general import image_normalization
from utils.utils_srgb import srgb_lin2gamma


class SGD_JPEG(CGH_iterative):
    """
    Stochastic Gradient Descent algorithm for optimizing phase
    """

    def __init__(self, wave_length, slm_resolution, pixel_pitch, device, linear_convolution=True):
        """

        @param wave_length: 11C
        @param slm_resolution: HW
        @param pixel_pitch: scalar
        @param device: cuda or cpu
        @param linear_convolution:
        """
        # Configuration information
        super().__init__(wave_length, slm_resolution, pixel_pitch, device, linear_convolution)
        self.phase = None

    def __call__(self,
                 a_camera_target,
                 # depth,
                 quality_jpeg: int = 90,
                 iterations: int = 100,
                 optimizer: str = "adam",
                 learning_rate_phase: float = 2e-2,
                 roi_shape=None,
                 initial_phase=None,
                 depth_mid=None,):
        """

        @param a_camera_target: NCHW
        @param depth: scalar
        @param roi_shape:
        @param iterations:
        @param optimizer:
        @param learning_rate_phase:
        @return:
        """
        depth_mid = depth_mid
        poh_hologram = self._2d(a_camera_target,
                                depth_mid,
                                quality_jpeg=quality_jpeg,
                                roi_shape=roi_shape,
                                iterations=iterations,
                                optimizer=optimizer,
                                learning_rate_phase=learning_rate_phase,
                                initial_phase=initial_phase)

        # Return GS phase field
        return poh_hologram

    def _2d(self,
            a_camera_target,
            depth,
            quality_jpeg: int = 100,
            iterations: int = 100,
            optimizer: str = "adam",
            learning_rate_phase: float = 1e-1,
            roi_shape=None,
            initial_phase=None):
        """

        @param a_camera_target: NCHW
        @param depth: scalar
        @param roi_shape:
        @param iterations:
        @param optimizer:
        @param learning_rate_phase:
        @return:
        """
        np.random.seed(1)
        torch.manual_seed(1)

        if np.shape(a_camera_target[0, 0, :, :]) != tuple(self.slm_resolution):
            raise Exception("Target dimensions do not match SLM dimensions!")

        if roi_shape is None:
            self.roi_shape = self.slm_resolution
        else:
            self.roi_shape = roi_shape

        # Randomly initialize phase parameters
        if initial_phase is None:
            initial_phase = torch.rand(self.image_resolution, device=self.device)  # NCHW
        else:
            initial_phase = torch.as_tensor(initial_phase, device=self.device)

        self.phase = initial_phase  # NCHW, -0.5~0.5
        self.phase.requires_grad = True  # updatable
        self.target = torch.as_tensor(a_camera_target, device=self.device, dtype=torch.float32)  # NCHW
        self.roi_shape = roi_shape  # region of interest
        self.target = crop(self.target,
                           data_format="nchw",
                           roi_shape=self.roi_shape)
        self.s = torch.tensor(1., requires_grad=True, device=self.device)  # +s(True), lr_s, cropping image
        parameters = [{'params': self.phase, 'lr': learning_rate_phase},
                      {'params': self.s}]

        # Optimization settings
        self.l2 = torch.nn.MSELoss().to(self.device)
        self.l1 = torch.nn.L1Loss().to(self.device)
        self.optim = None
        if optimizer.upper() == "ADAM":
            self.optim = torch.optim.Adam(parameters, lr=learning_rate_phase)
        elif self.optim is None:
            raise Exception("Optimizer is None!")
        self.z = depth

        jpeg = DiffJPEG(height=self.slm_resolution[0], width=self.slm_resolution[1], differentiable=True, quality=quality_jpeg)
        jpeg.to(device=self.device)
        for i in jpeg.parameters():
            i.requires_grad = False

        psnr_rec_list = []
        ssim_rec_list = []
        # Optimize phase
        t = tqdm.tqdm(range(iterations))
        for i in t:
            # # evaluation
            # psnr_rec, ssim_rec = self.evaluation(jpeg_q=50)
            # psnr_rec_list.append(psnr_rec)
            # ssim_rec_list.append(ssim_rec)

            # Clear gradients
            self.optim.zero_grad()

            # Forward simulation
            # jpeg encode and decode
            phase_warped = self.phase % (2 * torch.pi) / (2 * torch.pi)     # NCHW
            phase_jpeg, quantized_blocks = jpeg(phase_warped)               # NCHW, NK88(K is the number of 8*8 blocks in one image)
            phase_jpeg = phase_jpeg * 2 * torch.pi                          # NCHW
            # self.phase_jpeg = self.phase

            # Complex field on SLM
            u_slm_jpeg = torch.exp(1j * phase_jpeg)                         # NCHW
            u_slm = torch.exp(1j * self.phase)                              # NCHW

            # Propagate to target plane
            rec_field_jpeg = self.propagator(u_slm_jpeg, z=self.z)[:, 0, ...]  # NDCHW -> NCHW
            rec_field = self.propagator(u_slm, z=self.z)[:, 0, ...]  # NDCHW -> NCHW

            rec_jpeg = (torch.abs(rec_field_jpeg))  # A -> Intensity
            rec = (torch.abs(rec_field))  # A -> Intensity

            # Calculate loss function
            # ms-ssim, rec_raw, ffl, reg_holo, bpp
            # loss_msssim = 1 - ms_ssim(self.target, rec_jpeg, data_range=1, size_average=True)
            # loss_ssim = 1 - torch_ssim(self.target, rec_jpeg, data_range=1, size_average=True)
            # loss_rec_raw = self.l2(self.target, self.s * rec)
            # loss_ffl = ffl(self.target, self.s * rec_jpeg)
            loss_rec = self.l2(self.target, self.s * rec_jpeg)
            # loss_tv = self.total_variation_loss(self.target, rec_jpeg, self.l1)
            # loss_reg_holo = self.l1(torch.zeros_like(phase_jpeg, device=self.device), phase_jpeg)
            loss_reg_tv = self.tv_loss(self.phase)
            # loss_res = self.l2(rec_jpeg - rec, torch.zeros_like(phase_jpeg, device=self.device))
            # loss_reg_bpp = self.calculate_bpp(quantized_blocks)
            loss = loss_rec + loss_reg_tv * self.tv_weight(quality=quality_jpeg)

            t.set_postfix(loss=loss.item())
            # print(f"The loss of iteration {i}:" + str(loss.detach().cpu().numpy()))
            # print(
            #     f"MSSSIM loss: {loss_msssim.detach().cpu().numpy():.2}. SSIM loss: {loss_ssim.detach().cpu().numpy():.2}."
            #     f" Rec_raw loss: {loss_rec_raw.detach().cpu().numpy():.2}."
            #     f" FFL loss: {loss_ffl.detach().cpu().numpy():.2}. Rec loss: {loss_rec.detach().cpu().numpy():.2}."
            #     f" reg_holo: {loss_reg_holo.detach().cpu().numpy():.2}.")
            loss.backward()
            self.optim.step()

        # print(self.s)
        # print(self.tv_weight(quality=quality_jpeg))

        self.phase.requires_grad = False
        # self.phase = jpeg(self.phase)

        # import pandas as pd
        # # Create a dictionary with the column names and data
        # perf_dic = {"PSNR": psnr_rec_list, "SSIM": ssim_rec_list}
        # # Save the DataFrame to a CSV file
        # perf_df = pd.DataFrame(data=perf_dic, index=None)
        # perf_df.to_csv("tmp.csv")

        return (self.phase).detach().cpu()  # NCHW 0~2pi

    def tv_weight(self, quality):
        """
        Computes the weight for the TV loss term given the quality level of the JPEG encoding.

        Args:
            quality (float): The quality level of the JPEG encoding.
            k (float): The steepness of the sigmoid curve.
            q0 (float): The quality level at which the weight is halfway between its minimum and maximum values.

        Returns:
            float: The weight for the TV loss term.
        """
        # Compute the weight using a sigmoid function
        weight = 1e-3 + (0 - 1e-3) / (1 + np.exp(-0.60 * (quality - 60)))

        return weight

    def tv_loss(self, img):
        """
        Computes the total variation (TV) loss for a given image.

        Args:
            img (torch.Tensor): The image for which to compute the TV loss.

        Returns:
            torch.Tensor: The TV loss for the given image.
        """
        # Compute the squared differences between adjacent pixels along the x and y axes
        diff_x = (img[:, :, :-1, :-1] - img[:, :, :-1, 1:]).pow(2)
        diff_y = (img[:, :, :-1, :-1] - img[:, :, 1:, :-1]).pow(2)

        # Compute the TV loss as the sum of the squared differences
        tv_loss = diff_x.mean([1, 2, 3]) + diff_y.mean([1, 2, 3])

        return tv_loss

    def calculate_bpp(self, quantized_blocks):
        """
        Calculate the bits per pixel (bpp) of a compressed image based on its quantized representation.

        Args:
            quantized_blocks: A torch tensor of shape (1, num_blocks, 8, 8) representing the quantized 8x8 blocks of the image.

        Returns:
            bpp: The estimated bits per pixel (bpp) of the compressed image.
        """
        # flatten the blocks into a 1D sequence of symbols
        symbols = quantized_blocks.flatten()

        # calculate the probability distribution of the symbols using a softmax function
        temperature = 0.1
        epsilon = 1e-6
        probabilities = torch.softmax((symbols + epsilon) / temperature, dim=0)

        # calculate the entropy of the sequence
        ent = -(probabilities * torch.log2(probabilities)).sum()

        # calculate the bits per pixel (bpp) based on the entropy
        bpp = ent / (8 * 8)

        return bpp

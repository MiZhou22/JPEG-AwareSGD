import torch
import numpy as np
from matplotlib import pyplot as plt

from Algorithm.CGH_algorithm import CGH_iterative
from utils.Angular_Spectrum_Method import Angular_Spectrum_Method
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

# Random seed to ensure reproducibility
from DiffJPEG.DiffJPEG import DiffJPEG
from utils.utils_asm import crop
from utils.utils_general import image_normalization, imread
from utils.utils_srgb import srgb_lin2gamma, srgb_gamma2lin


class SGD(CGH_iterative):
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
                 iterations=100,
                 optimizer="adam",
                 learning_rate_phase=2e-2,
                 roi_shape=None,
                 initial_phase=None,
                 depth_mid=None):
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
            iterations=100,
            optimizer="adam",
            learning_rate_phase=0.001,
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
        parameters = [{'params': self.phase, 'lr': learning_rate_phase}]

        # Scale parameters
        # +s(True), lr_s, cropping image
        self.s = torch.tensor(1., requires_grad=True, device=self.device)
        parameters = parameters + [{'params': self.s, 'lr': 2e-3}]

        # Optimization settings
        self.loss_function = torch.nn.MSELoss().to(self.device)
        self.optim = None
        if optimizer.upper() == "ADAM":
            self.optim = torch.optim.Adam(parameters, lr=learning_rate_phase)
        elif optimizer.upper() == "RMSPROP":
            self.optim = torch.optim.RMSprop(params=parameters, lr=learning_rate_phase)
        elif optimizer.upper() == "SGD":
            self.optim = torch.optim.SGD(params=parameters, lr=learning_rate_phase)
        elif self.optim is None:
            raise Exception("Optimizer is None!")
        self.z = depth

        # Optimize phase
        psnr_rec_list = []
        ssim_rec_list = []
        for i in range(iterations):
            # # evaluation
            # psnr_rec, ssim_rec = self.evaluation(jpeg_q=50)
            # psnr_rec_list.append(psnr_rec)
            # ssim_rec_list.append(ssim_rec)

            # Clear gradients
            self.optim.zero_grad()

            # Forward simulation
            u_slm = torch.exp(1j * self.phase)  # NCHW
            reconstruction_field = self.propagator(u_slm,
                                                   z=self.z)[:, 0, ...]  # NDCHW -> NCHW
            reconstruction_intensity = (torch.abs(reconstruction_field))**2
            reconstruction_intensity = crop(reconstruction_intensity,  # cropping the ROI
                                            data_format="nchw",
                                            roi_shape=self.roi_shape)

            # Calculate loss function
            loss = self.loss_function(srgb_lin2gamma(self.target**2), self.s * srgb_lin2gamma(reconstruction_intensity))

            loss.backward()
            self.optim.step()

        self.phase.requires_grad = False
        return (self.phase).cpu()  # NCHW 0~2pi
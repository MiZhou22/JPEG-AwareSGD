import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append("..")

from Algorithm.SGD_JPEG import SGD_JPEG
from DiffJPEG.DiffJPEG import DiffJPEG
from utils.utils_srgb import srgb_gamma2lin, srgb_lin2gamma
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from Algorithm.SGD import SGD
from utils.utils_general import imshow, image_normalization, imread
from utils.utils_asm import crop

def load_algorithm(algorithm_name: str,
                   propagator):
    algorithm_name = algorithm_name.upper()
    if algorithm_name == "SGD":
        algorithm_class = SGD
    elif algorithm_name == "SGD_JPEG":
        algorithm_class = SGD_JPEG
    else:
        raise "No algorithm selected!"
    algorithm = algorithm_class(wave_length=propagator.wave_length[0, 0, ...].permute(1, 2, 0),
                                slm_resolution=propagator.slm_resolution,
                                pixel_pitch=propagator.pixel_pitch,
                                device=propagator.gpu_id,
                                linear_convolution=propagator.linear_convolution)

    return algorithm

def save_phase(phase,
               algorithm_name: str,
               show_phase: bool = False,
               **cgh_DOEs_parameters) -> float:
    """
    @param algorithm_name:
    @param phase: NCHW -> 0-255
    @param show_phase:
    @return cr
    """

    def calculate_compression_rate(original_image, compressed_image):
        # Get the size of the original, uncompressed image
        original_size = os.path.getsize(original_image)

        # Get the size of the compressed JPEG image
        compressed_size = os.path.getsize(compressed_image)

        # Calculate the compression rate
        compression_rate = original_size / compressed_size

        return compression_rate

    z = cgh_DOEs_parameters["depth_mid"]
    quality = cgh_DOEs_parameters["quality_jpeg"]
    iteration_num = cgh_DOEs_parameters["iterations_image_num"]
    color = cgh_DOEs_parameters["color"]

    # transfer to numpy.ndarray
    if torch.is_tensor(phase):
        phase = phase.numpy().copy()
    else:
        pass

    # save and show the phase
    phase = np.transpose(phase[0, ...], (1, 2, 0))  # HWC
    phase_wrapped_normed = (np.round((phase % (2 * np.pi)) / (2 * np.pi) * 255)).astype(np.uint8)
    if show_phase:
        imshow(phase_wrapped_normed)

    if not iteration_num:
        cv2.imwrite(f"../results/phase/{algorithm_name}_{z}.png", phase_wrapped_normed[..., ::-1])
    else:
        cv2.imwrite(f"../results/phase/{algorithm_name}_{color}_{z}_{iteration_num}.png", phase_wrapped_normed[..., ::-1])

    cv2.imwrite(f"tmp.jpg", phase_wrapped_normed[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    cv2.imwrite(f"tmp.png", phase_wrapped_normed[..., ::-1])
    cr = calculate_compression_rate("tmp.png", "tmp.jpg")
    return cr

def run_cgh_algorithm(algorithm,
                      image: torch.Tensor,
                      **parameters):
    """
    Use cgh_algorithm to compute the hologram from image_path or image array and display it on the slm_display device
    @param algorithm: algorithm object
    @param image: image array NCHW
    @param depth: depth array NCHW
    @param parameters: cgh parameters and DOEs' po_hologram patterns(0-2pi)
    @return:
    """
    # parameters:
    quality_jpeg = parameters["quality_jpeg"]
    iterations_image_num = parameters["iterations_image_num"]
    roi_shape = parameters["roi_shape"]
    depth_mid = parameters["depth_mid"]

    cpx_mid_hologram_phs = None
    # run cgh algorithm
    # Under 2D condition: for GS, HIO, SGD, SGD_GS, SGD_fourierloss, propagate distance is depth_mid
    if isinstance(algorithm, (SGD)):
        po_hologram = algorithm(torch.sqrt(srgb_gamma2lin(image)),  # NCHW
                                iterations=iterations_image_num,
                                roi_shape=roi_shape,
                                depth_mid = depth_mid,)
    elif isinstance(algorithm, (SGD_JPEG)):
        po_hologram = algorithm(torch.sqrt(srgb_gamma2lin(image)),  # NCHW
                                quality_jpeg=quality_jpeg,
                                iterations=iterations_image_num,
                                roi_shape=roi_shape,
                                depth_mid=depth_mid,)
    else:
        raise Exception("No matched algorithm")

    # po_hologram shape: NCHW
    return po_hologram, cpx_mid_hologram_phs  # NCHW


def simulation_run(algorithm_name: str,
                   algorithm,
                   image_path: str,
                   **cgh_DOEs_parameters):
    def calculate_bpp(image: np.ndarray,
                      quality: int) -> float:
        image = np.round(image%(2 * np.pi) / (2 * np.pi)*255).astype(np.uint8)
        # Encode the image as a JPEG image and save it to an in-memory buffer
        _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

        # Calculate the size of the compressed JPEG image (in bits)
        jpeg_size = buffer.nbytes * 8

        # Calculate the bpp value
        return jpeg_size / (image.shape[0] * image.shape[1])

    def calculate_tv(image: torch.Tensor) -> torch.Tensor:
        """
        @param image: NCHW
        @return: tv
        """
        # Compute the squared differences between adjacent pixels along the x and y axes
        diff_x = torch.pow(image[:, :, :-1, :-1] - image[:, :, :-1, 1:], 2)
        diff_y = torch.pow(image[:, :, :-1, :-1] - image[:, :, 1:, :-1], 2)

        # Compute the TV loss as the sum of the squared differences
        return diff_x.mean() + diff_y.mean()

    # parameters
    depth_mid = cgh_DOEs_parameters["depth_mid"]
    quality_jpeg = cgh_DOEs_parameters["quality_jpeg"]
    propagator = cgh_DOEs_parameters["propagator"]
    color = cgh_DOEs_parameters["color"].upper()
    roi_shape = cgh_DOEs_parameters["roi_shape"]
    device = cgh_DOEs_parameters["device"]

    # read, crop
    image = imread(path=image_path, color=color)  # HWC
    image = crop(image=image, data_format="hwc", roi_shape=roi_shape)  # HWC

    # numpy to tensor, HWC to NCHW
    image = np.transpose(image[..., None], (3, 2, 0, 1))  # NCHW
    image = torch.as_tensor(image, device=device)

    # run algorithm
    phase_unwrapped, d_p = run_cgh_algorithm(algorithm,
                                             image,
                                             # depth,
                                             **cgh_DOEs_parameters)  # NCHW

    cr = save_phase(phase_unwrapped,
                    algorithm_name,
                    show_phase=False,
                    **cgh_DOEs_parameters)

    tv = np.round(calculate_tv(phase_unwrapped).numpy(), 2)
    bpp = np.round(calculate_bpp(np.transpose(phase_unwrapped[0, ...].numpy(), (1, 2, 0)), quality_jpeg), 2)
    cr = np.round(cr, 2)

    # Hologram encoding and decoding
    jpeg = DiffJPEG(height=1072, width=1920, differentiable=False, quality=quality_jpeg).to("cpu")
    for p in jpeg.parameters():
        p.requires_grad = False
    phase_wrapped_normed = phase_unwrapped % (2 * torch.pi) / (2 * torch.pi)
    phase_wrapped_normed_jpeg, quantized_blocks = jpeg(phase_wrapped_normed)
    phase_wrapped = phase_wrapped_normed_jpeg * 2 * torch.pi

    # Hologram propagation
    rec_amp = propagator(torch.exp(1j * phase_wrapped), z=depth_mid)[:, 0, ...].abs()   # NCHW

    # amp to int, crop, NCHW to HWC, lin to gamma, normalization
    rec_int = torch.square(rec_amp)                                             # NCHW
    rec_int = crop(image=rec_int, data_format="nchw", roi_shape=roi_shape)
    rec_int = rec_int[0, ...].permute(1, 2, 0).cpu().numpy()                    # HWC
    rec_int = srgb_lin2gamma(rec_int)
    rec_int = image_normalization(rec_int, axis=(0, 1))
    image = image.cpu().numpy()
    image = np.transpose(image[0, ...], (1, 2, 0))                              # HWC

    # Evaluation
    ssim_rec = np.round(ssim(image, rec_int, channel_axis=2, data_range=1), 4)
    psnr_rec = np.round(psnr(image, rec_int, data_range=1), 2)
    # imshow(rec_int)
    # imshow(image)

    # Save the performance
    img_name = image_path.split("/")[-1][:-4]
    img_dir = f"../results/simulation/SGD_JPEG/{int(abs(depth_mid) * 100)}cm/DIV2k/{img_name}/"
    file_name = f"{algorithm_name}_JPEG({quality_jpeg})_{img_name}_{color}_({psnr_rec:.2f}, {ssim_rec:.4f}, {cr:.2f}, {bpp:.2f}, {tv:.2f}).png"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    print(cv2.imwrite(img_dir + file_name, rec_int * 255))
    print("PSNR: " + str(psnr_rec))
    print("SSIM: " + str(ssim_rec))
    print("cr: " + str(cr))
    print("bpp: " + str(bpp))

    return psnr_rec, ssim_rec, cr, bpp, tv

# Import modules from the parent directory
import argparse
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import sys
sys.path.append("")

from utils.utils_algorithm import simulation_run, load_algorithm
from utils.utils_asm import *
from utils.Angular_Spectrum_Method import Angular_Spectrum_Method
from utils import utils_asm


def get_args() -> argparse.Namespace:
    """
    get_args
    @return:
    """
    image_path = "data/DIV2K_test/img/0886_resize.png"

    parser = argparse.ArgumentParser(description="Simulation parameters")
    parser.add_argument("--seed", type=int, default=3407, help="random seed")
    parser.add_argument("--iterations_image_num", type=int, default=1000, help="Total iterations for image hologram")
    parser.add_argument("--quality_jpeg", type=int, default=50, help="Quality for encode&decode and jpeg-aware SGD")
    parser.add_argument("--algorithm_name", type=str, default="SGD_JPEG", help="CGH algorithm")
    parser.add_argument("--color", type=str, default="b", help="The color of image")
    parser.add_argument("--image_path", type=str, default=image_path, help="Image path for CGH algorithm")
    parser.add_argument("--depth_mid", type=float, default=-20 * cm, help="the depth of the middle depth range")
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(), help="gpus to use")

    return parser.parse_args()


if __name__ == "__main__":
    # Units
    nm, um, mm, cm, m = 1e-9, 1e-6, 1e-3, 1e-2, 1e0

    args = get_args()
    seed = args.seed  # Random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # parameters
    depth_mid = args.depth_mid
    iterations_image_num = args.iterations_image_num  # Number of iterations for the algorithm (For IMAGE)
    quality_jpeg = args.quality_jpeg
    slm_resolution = (1072, 1920)  # 1080*1920*c / 500*500 (row, column)
    roi_shape = slm_resolution
    pixel_pitch = PIXEL_PITCH  # pixel pitch

    # Configurations
    image_path = args.image_path
    device = torch.device("cuda")
    linear_convolution = True
    JPEG_compression = True     # JPEG compression after generating holograms
    color = args.color.upper()
    assert color != "RGB"
    if color == "RGB":
        wavelength = np.array([[[utils_asm.WAVE_LENGTH_COLOR_R,
                                 utils_asm.WAVE_LENGTH_COLOR_G,
                                 utils_asm.WAVE_LENGTH_COLOR_B]]])
    elif color == "R":
        wavelength = np.array([[[utils_asm.WAVE_LENGTH_COLOR_R]]])
    elif color == "G":
        wavelength = np.array([[[utils_asm.WAVE_LENGTH_COLOR_G]]])
    elif color == "B":
        wavelength = np.array([[[utils_asm.WAVE_LENGTH_COLOR_B]]])
    else:
        raise Exception("Wave length is not set!")

    # Implementation of Angular Spectrum Method
    print("=>Propagator warming up!")
    propagator = Angular_Spectrum_Method(wave_length=wavelength,
                                         slm_resolution=slm_resolution,
                                         pixel_pitch=pixel_pitch,
                                         gpu_id=device,
                                         linear_convolution=linear_convolution)
    propagator(torch.rand((1, 3, slm_resolution[0], slm_resolution[1])), z=0.5)
    print("Propagator warming up done!")

    # prepare for the cgh parameters and DOEs
    parameters = {
        "iterations_image_num": iterations_image_num,
        "quality_jpeg": quality_jpeg,
        "color": color,
        "roi_shape": roi_shape,
        "propagator": propagator,
        "depth_mid": depth_mid,
        "JPEG_compression": JPEG_compression,
        "device": device,
    }

    algorithm_name = args.algorithm_name.upper()
    algorithm = load_algorithm(algorithm_name, propagator)

    # with torch.no_grad():
    ssim_img, psnr_img, cr, bpp, tv = simulation_run(algorithm_name=algorithm_name,
                                                     algorithm=algorithm,
                                                     image_path=image_path,
                                                     **parameters)

    pass
import numpy as np
import cv2

r_img = cv2.imread('./results/simulation/SGD_JPEG/20cm/DIV2k/0886_resize/SGD_JPEG_JPEG(50)_0886_resize_R_(17.43, 0.4440, 2.40, 2.70, 1.24).png', -1)
g_img = cv2.imread('./results/simulation/SGD_JPEG/20cm/DIV2k/0886_resize/SGD_JPEG_JPEG(50)_0886_resize_G_(19.66, 0.3810, 2.33, 2.79, 1.04).png', -1)
b_img = cv2.imread('./results/simulation/SGD_JPEG/20cm/DIV2k/0886_resize/SGD_JPEG_JPEG(50)_0886_resize_B_(18.93, 0.3193, 2.43, 2.78, 1.51).png', -1)

cv2.imwrite("./results/simulation/SGD_JPEG/20cm/DIV2k/0886_resize/rgb.png", np.stack((b_img, g_img, r_img), axis=-1))
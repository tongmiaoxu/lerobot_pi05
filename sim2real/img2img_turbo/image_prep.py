# Vendored from: https://github.com/GaParmar/img2img-turbo
# Upstream commit: 86f54146590ffb4543c8cf85b5a36657da670924
# Original path in upstream: src/image_prep.py

import numpy as np
from PIL import Image
import cv2


def canny_from_pil(image, low_threshold=100, high_threshold=200):
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    control_image = Image.fromarray(image)
    return control_image

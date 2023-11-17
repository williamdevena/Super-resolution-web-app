import cv2
import numpy as np


def imageFromString(filestr):
    file_bytes = np.fromstring(filestr, np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    return image

def resizeImage(image, rescale_factor):
    width = int(image.shape[1] * rescale_factor)
    height = int(image.shape[0] * rescale_factor)
    dim = (width, height)
    image = cv2.resize(image, dim)

    return image
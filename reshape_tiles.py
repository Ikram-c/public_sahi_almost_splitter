# finish later using https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7/
import numpy as np


def reshape_tiles(image, kernel_size):

    img_height, img_width, channels = image.shape[:2]
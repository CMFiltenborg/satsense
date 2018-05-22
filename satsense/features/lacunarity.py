import cv2

import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from skimage.filters import rank
import skimage.morphology as morp
from skimage.transform import pyramid_reduce, pyramid_expand

from satsense import SatelliteImage, WORLDVIEW3, extract_features, RGB
from satsense.features import Feature, FeatureSet
import numpy as np
import scipy as sp
from skimage.feature import canny as canny_edge
import numba
from numba import jit, prange
import time

from satsense.generators import CellGenerator
from satsense.image import get_grayscale_image, get_rgb_bands


@jit("float64(boolean[:, :], int64)", nopython=True, parallel=True)
def lacunarity(edged_image, box_size):
    """
    Calculate the lacunarity value over an image, following these papers:

    Kit, Oleksandr, and Matthias Lüdeke. "Automated detection of slum area change in Hyderabad, India using multitemporal satellite imagery." ISPRS journal of photogrammetry and remote sensing 83 (2013): 130-137.

    Kit, Oleksandr, Matthias Lüdeke, and Diana Reckien. "Texture-based identification of urban slums in Hyderabad, India using remote sensing data." Applied Geography 32.2 (2012): 660-667.
    """

    # accumulator holds the amount of ones for each position in the image, defined by a sliding window
    #
    accumulator = np.zeros(edged_image.shape)
    for i in prange(edged_image.shape[0] - (box_size)):
        for j in prange(edged_image.shape[1] - (box_size)):
            # sum the binary-box for the amount of 1s in this box
            # box = edged_image[j:j + box_size, j: j + box_size]
            accumulator[i, j] = np.sum(edged_image[j:j + box_size, i: i + box_size])

    accumulator = accumulator.flatten()
    mean_sqrd = np.mean(accumulator) ** 2
    if mean_sqrd == 0:
        return 0.0

    return (np.var(accumulator) / mean_sqrd) + 1


class Lacunarity(Feature):
    def __init__(self, windows=((25, 25),), box_sizes=(10, 20, 30)):
        super(Lacunarity, self)
        self.box_sizes = box_sizes
        self.windows = windows
        self.feature_size = len(self.windows) * len(box_sizes)

    @jit
    def __call__(self, cell):
        result = np.zeros(self.feature_size)
        len_box_sizes = len(self.box_sizes)
        for i, window in enumerate(self.windows):
            win = cell.super_cell(window, padding=True)

            # For every box size we have a feature for this window
            for j in range(len_box_sizes):
                box_size = self.box_sizes[j]
                result[i + j] = lacunarity(win.canny_edged, box_size)

        return result

    def __str__(self):
        return "Lacunarity-{}-{}".format(str(self.windows), str(self.box_sizes))
